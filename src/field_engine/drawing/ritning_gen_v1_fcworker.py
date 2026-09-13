#!/usr/bin/env python3
"""The FreeCAD side of the drawing generator; runs INSIDE the FreeCAD interpreter.

Never run this with the system python. The driver (ritning_gen_v1.py) starts a FreeCAD
console process and points here through the environment variable RITNING_JOB.

What it does, and only that:
  1. reads STEP files (Part.read) and applies the manifest's transform_mm,
  2. builds one TechDraw projection per view (TechDraw::DrawViewPart -> TechDraw.viewPartAsSvg),
     i.e. the OCCT hidden-line removal, the same engine TechDraw's own drawing views use,
  3. measures the geometry (optimalBoundingBox plus cylindrical faces -> holes and diameters),
  4. calibrates the view frame: TechDraw's SVG coordinates are unscaled millimetres centred on
     the projection's own centre; which (u,v) sign convention holds is proved by matching our own
     projection of the body's vertices against the view's own visible vertices. If no candidate
     matches within tolerance, calib.ok is false and the driver refuses to place any dimension
     that depends on the frame (fail-closed).

Input: JSON at RITNING_JOB. Output: JSON at RITNING_JOB_OUT with a per-view SVG fragment, the
calibrated frame and the measured values.

Job keys: parts, views, holes, base_dir; or template_dir + templates to copy FreeCAD's own
ISO 5457 sheet templates out of the FreeCAD installation; or selftest.
"""
import json
import math
import os
import re
import shutil

import FreeCAD  # noqa: F401  (exists only inside the FreeCAD interpreter)
import Part
import TechDraw


def _vec(t):
    return FreeCAD.Vector(float(t[0]), float(t[1]), float(t[2]))


def _cross(a, b):
    return FreeCAD.Vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def template_dir():
    """FreeCAD's own TechDraw ISO template directory inside the installation."""
    return os.path.join(FreeCAD.getResourceDir(), "Mod", "TechDraw", "Templates", "ISO")


def load_shape(step_path, transform):
    sh = Part.Shape()
    sh.read(step_path)
    if transform:
        m = FreeCAD.Matrix(*[float(v) for row in transform for v in row])
        sh = sh.transformGeometry(m)
    return sh


_CYL_LOG = []
_SOLIDS = []


def measure_holes(shape, max_holes=400):
    """Inner cylindrical faces (holes) -> diameter, axis, centre, depth.

    Only full cylinders (u span ~2pi) count; a partial cylinder is a rounding, not a hole.
    Inner versus outer is decided by the material, not by the face orientation: an imported STEP
    may have globally flipped orientation, so the test is a membership question. Is the point just
    inside the mantle in material? Then it is a boss. Material outside and void inside means a
    hole. Both or neither is ambiguous and the face is not measured (fail-closed).
    """
    out = []
    try:
        faces = shape.Faces
    except Exception:
        return out
    for f in faces[: 20000]:
        try:
            surf = f.Surface
            if surf.__class__.__name__ != "Cylinder":
                continue
            u0, u1, v0, v1 = f.ParameterRange
            if (u1 - u0) < (2 * math.pi - 1e-3):
                continue
            r = float(surf.Radius)
            ax = surf.Axis
            ctr = surf.Center
            um = 0.5 * (u0 + u1)
            vm = 0.5 * (v0 + v1)
            p = f.valueAt(um, vm)
            n = f.normalAt(um, vm)
            rad = FreeCAD.Vector(p.x - ctr.x, p.y - ctr.y, p.z - ctr.z)
            axn = FreeCAD.Vector(ax.x, ax.y, ax.z)
            axn.normalize()
            k = rad.dot(axn)
            radial = FreeCAD.Vector(rad.x - axn.x * k, rad.y - axn.y * k, rad.z - axn.z * k)
            if radial.Length < 1e-9:
                continue
            radial.normalize()
            step = max(0.05, min(0.15 * r, 1.0))
            pin = FreeCAD.Vector(p.x - radial.x * step, p.y - radial.y * step, p.z - radial.z * step)
            pout = FreeCAD.Vector(p.x + radial.x * step, p.y + radial.y * step, p.z + radial.z * step)
            try:
                # Shape.isInside on a COMPOUND is not well defined; the question has to be asked
                # per SOLID and OR:ed, otherwise neighbouring parts turn into phantom holes.
                mat_in = any(sd.isInside(pin, 1e-6, True) for sd in _SOLIDS)
                mat_out = any(sd.isInside(pout, 1e-6, True) for sd in _SOLIDS)
            except Exception:
                mat_in = mat_out = None
            inner = (mat_out is True and mat_in is False)
            _CYL_LOG.append({"r": round(r, 4), "inner": bool(inner), "area": round(float(f.Area), 2),
                             "dot_raw": round(float(n.dot(radial)), 3),
                             "mat_in": mat_in, "mat_ut": mat_out})
            if not inner:
                continue
            bb = f.BoundBox
            depth = max(bb.XLength * abs(axn.x), bb.YLength * abs(axn.y), bb.ZLength * abs(axn.z))
            out.append({
                "d_mm": round(2.0 * r, 6),
                "axis": [round(axn.x, 9), round(axn.y, 9), round(axn.z, 9)],
                "center": [round(ctr.x, 6), round(ctr.y, 6), round(ctr.z, 6)],
                "depth_mm": round(float(depth), 6),
                "area_mm2": round(float(f.Area), 4),
            })
            if len(out) >= max_holes:
                break
        except Exception:
            continue
    # dedupe: the two mantle halves of one bore give two faces with identical axis/centre/diameter
    seen = {}
    ded = []
    for h in out:
        key = (round(h["d_mm"], 3), tuple(round(a, 4) for a in h["axis"]),
               tuple(round(c, 3) for c in h["center"]))
        if key in seen:
            ded[seen[key]]["area_mm2"] += h["area_mm2"]
            continue
        seen[key] = len(ded)
        ded.append(dict(h))
    return ded


def svg_path_bbox(svg):
    """Bounding box over the path data's real endpoints.

    Reading all numbers of a d attribute pairwise is wrong as soon as an arc is present: the
    'A rx ry rot large sweep x y' command has SEVEN parameters, so pairwise reading turns the radii
    into coordinates and inflates the view. The commands must be tokenised; only the endpoint (and
    the control points, which lie in the plane) count.
    """
    par = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "A": 7, "Z": 0}
    xs, ys = [], []
    for d in re.findall(r'\sd="([^"]*)"', svg):
        toks = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?', d)
        i = 0
        cx = cy = 0.0
        cmd = None
        while i < len(toks):
            t = toks[i]
            if re.match(r'^[A-Za-z]$', t):
                cmd = t
                i += 1
                if cmd.upper() == "Z":
                    continue
            if cmd is None:
                i += 1
                continue
            n = par[cmd.upper()]
            vals = []
            for k in range(n):
                if i + k >= len(toks) or re.match(r'^[A-Za-z]$', toks[i + k]):
                    vals = None
                    break
                vals.append(float(toks[i + k]))
            if vals is None:
                i += 1
                continue
            i += n
            rel = cmd.islower()
            u = cmd.upper()
            if u == "H":
                cx = cx + vals[0] if rel else vals[0]
            elif u == "V":
                cy = cy + vals[0] if rel else vals[0]
            elif u == "A":
                cx = cx + vals[5] if rel else vals[5]
                cy = cy + vals[6] if rel else vals[6]
            else:
                # M/L/T: (x,y); C: 3 pairs; S/Q: 2 pairs -- the control points lie in the plane
                for k in range(0, n, 2):
                    px = cx + vals[k] if rel else vals[k]
                    py = cy + vals[k + 1] if rel else vals[k + 1]
                    xs.append(px)
                    ys.append(py)
                cx = cx + vals[n - 2] if rel else vals[n - 2]
                cy = cy + vals[n - 1] if rel else vals[n - 1]
                continue
            xs.append(cx)
            ys.append(cy)
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def calibrate_frame(view, shape, svg):
    """Measure (do not guess) the map 3D point -> SVG coordinate for this view.

    TechDraw's own API gives both ends of the map:
      view.projectPoint(P)      -> (u,v) in the projection plane, UNCENTRED
      view.getVisibleVertexes() -> the same points in VIEW coordinates (= the SVG's coordinates)
    The rule, derived and then verified below rather than assumed:
      svg_u =  (pp(P).x - pp(GC).x)
      svg_v = -(pp(P).y - pp(GC).y)        GC = view.getGeometricCenter()
    Verification: every visible view vertex is matched against the nearest predicted 3D vertex and
    the hit fraction is reported. If another sign variant beats the reference, ok is set to False
    and the driver refuses to draw frame-dependent dimensions (fail-closed).
    """
    sbb = svg_path_bbox(svg)
    gc = view.getGeometricCenter()
    pc = view.projectPoint(gc)
    proj = [view.projectPoint(v.Point) for v in shape.Vertexes[:3000]]
    pred = [(q.x - pc.x, -(q.y - pc.y)) for q in proj]
    try:
        vv = [(w.x, w.y) for w in view.getVisibleVertexes()]
    except Exception:
        vv = []

    def traffandel(su, sv, tol=0.05):
        """Fraction of 3D vertices that have a VIEW vertex within tol under the sign choice (su,sv).

        A median distance would not do as a measure: in an assembly most 3D vertices are hidden and
        have no view counterpart, so the median measures the occlusion degree rather than the frame.
        The hit fraction is insensitive to that: hidden vertices lower all four variants equally, so
        the comparison between variants still stands.
        """
        if not vv or not proj:
            return None
        n = 0
        t2 = tol * tol
        for q in proj:
            a0 = su * (q.x - pc.x)
            a1 = sv * (q.y - pc.y)
            for b in vv:
                if (a0 - b[0]) ** 2 + (a1 - b[1]) ** 2 <= t2:
                    n += 1
                    break
        return n / float(len(proj))

    variants = {}
    for su in (1, -1):
        for sv in (1, -1):
            s = traffandel(su, sv)
            if s is not None:
                variants["%+d%+d" % (su, sv)] = round(s, 6)
    # The reference rule is a constant of TechDraw 1.1, derived from its own API (projectPoint is
    # uncentred, y points down on the sheet) and measured on known bodies. Per drawing it is not
    # re-derived but CONTRADICTION-TESTED: if another variant beats the reference -> fail-closed.
    REF = "+1-1"
    out = {
        "regel": "svg=(pp(P).x-pp(GC).x, -(pp(P).y-pp(GC).y)) [TechDraw constant, contradiction-tested]",
        "gc": [round(gc.x, 6), round(gc.y, 6), round(gc.z, 6)],
        "n_vy_vertexer": len(vv),
        "n_3d_vertexer": len(proj),
        "svg_bbox": [round(v, 6) for v in sbb] if sbb else None,
        "varianter_traffandel": variants,
    }
    if variants:
        ref = variants.get(REF, 0.0)
        andra = max([v for k, v in variants.items() if k != REF] or [0.0])
        out["su"] = 1
        out["sv"] = -1
        out["traffandel"] = ref
        out["n_traffar"] = int(round(ref * len(proj)))
        out["basta_alternativ_traffandel"] = andra
        out["motsagd"] = bool(andra > ref + 1e-9)
        out["sarskiljbar"] = bool(ref > andra + 1e-9)
        # The threshold is a COUNT, not a fraction: the fraction falls with the occlusion degree
        # and would then throw away perfectly correct dimensions. The condition is (a) at least 3
        # points where our map hits TechDraw's own view vertex exactly (or all available ones if
        # there are 2), and (b) no other sign variant hits better.
        out["ok"] = bool((out["n_traffar"] >= 3 or (len(proj) >= 2 and ref >= 0.99)) and not out["motsagd"])
    else:
        # no visible view vertex (e.g. a pure body of revolution) -> fall back on a bbox comparison
        # between the predicted points and the SVG's paths
        if sbb and pred:
            pu = [p[0] for p in pred]
            pv = [p[1] for p in pred]
            e = max(abs(min(pu) - sbb[0]), abs(max(pu) - sbb[2]),
                    abs(min(pv) - sbb[1]), abs(max(pv) - sbb[3]))
            out["bbox_residual_mm"] = round(e, 6)
            out["ok"] = bool(e <= 0.5)
            out["fallback"] = "bbox (no visible view vertices)"
        else:
            out["ok"] = False
            out["fallback"] = "no measurable channel"
    return out


def selftest(doc):
    """Calibration proof: an ASYMMETRIC body with sharp geometry is projected and all four sign
    variants are measured. The reference rule must win unambiguously here, otherwise the constant
    in calibrate_frame is wrong, not the drawing."""
    b = Part.makeBox(40, 30, 10)
    hal = Part.makeCylinder(3, 40, FreeCAD.Vector(12, 8, -5), FreeCAD.Vector(0, 0, 1))
    tapp = Part.makeBox(6, 6, 8, FreeCAD.Vector(31, 22, 10))
    sh = b.cut(hal).fuse(tapp)
    obj = doc.addObject("Part::Feature", "Selftest")
    obj.Shape = sh
    doc.recompute()
    page = doc.addObject("TechDraw::DrawPage", "P")
    tpl = doc.addObject("TechDraw::DrawSVGTemplate", "T")
    tpl.Template = os.path.join(template_dir(), "A3_Landscape_blank.svg")
    page.Template = tpl
    ut = {}
    for vid, dv, xv in (("top", (0, 0, -1), (1, 0, 0)), ("front", (0, -1, 0), (1, 0, 0))):
        v = doc.addObject("TechDraw::DrawViewPart", "V" + vid)
        page.addView(v)
        v.Source = [obj]
        v.Direction = _vec(dv)
        v.XDirection = _vec(xv)
        v.Scale = 1.0
        doc.recompute()
        ut[vid] = calibrate_frame(v, sh, TechDraw.viewPartAsSvg(v))
    return ut


def copy_templates(job):
    """Copy the requested sheet templates out of the FreeCAD installation into template_dir.

    The templates ship with FreeCAD; this repository does not carry a copy of them.
    """
    dst = job["template_dir"]
    os.makedirs(dst, exist_ok=True)
    src = template_dir()
    out = {}
    for name in job["templates"]:
        s = os.path.join(src, name)
        if not os.path.exists(s):
            out[name] = {"fel": "not found in %s" % src}
            continue
        d = os.path.join(dst, name)
        shutil.copyfile(s, d)
        out[name] = {"path": d, "size": os.path.getsize(d)}
    return out


def main():
    job = json.load(open(os.environ["RITNING_JOB"]))
    out_path = os.environ["RITNING_JOB_OUT"]
    base = job.get("base_dir", os.getcwd())

    if job.get("templates"):
        json.dump({"templates": copy_templates(job),
                   "freecad_version": FreeCAD.Version()}, open(out_path, "w"))
        return

    doc = FreeCAD.newDocument("ritning")
    if job.get("selftest"):
        json.dump({"selftest": selftest(doc),
                   "freecad_version": FreeCAD.Version()}, open(out_path, "w"))
        return
    shapes = []
    part_meas = []
    for p in job["parts"]:
        sp = p["step"] if os.path.isabs(p["step"]) else os.path.join(base, p["step"])
        try:
            sh = load_shape(sp, p.get("transform"))
        except Exception as e:  # one broken part must not kill the whole drawing
            part_meas.append({"name": p["name"], "fel": "%s: %s" % (type(e).__name__, e)})
            continue
        shapes.append(sh)
        bb = sh.BoundBox
        part_meas.append({
            "name": p["name"],
            "bbox": [round(bb.XMin, 6), round(bb.YMin, 6), round(bb.ZMin, 6),
                     round(bb.XMax, 6), round(bb.YMax, 6), round(bb.ZMax, 6)],
            "volume_mm3": round(float(sh.Volume), 4),
            "n_solids": len(sh.Solids),
        })
    if not shapes:
        json.dump({"fel": "inga_shapes"}, open(out_path, "w"))
        return

    comp = shapes[0] if len(shapes) == 1 else Part.makeCompound(shapes)
    obj = doc.addObject("Part::Feature", "Body")
    obj.Shape = comp
    doc.recompute()

    # Part.Shape.BoundBox is the CONTROL-POINT HULL, not the body's box: for turned surfaces it
    # inflates the dimension. optimalBoundingBox() is the tight box and is the only thing that may
    # be dimensioned; if it is missing, no dimensions are placed.
    bb_pol = comp.BoundBox
    geom = {
        "bbox_pol_mm": [round(bb_pol.XMin, 6), round(bb_pol.YMin, 6), round(bb_pol.ZMin, 6),
                        round(bb_pol.XMax, 6), round(bb_pol.YMax, 6), round(bb_pol.ZMax, 6)],
        "volume_mm3": round(float(comp.Volume), 4),
        "n_solids": len(comp.Solids),
        "n_faces": len(comp.Faces),
    }
    bb = None
    try:
        bb = comp.optimalBoundingBox()
        geom["bbox_metod"] = "Part.Shape.optimalBoundingBox (tight AABB, global frame)"
    except Exception as e:
        geom["bbox_optimal_fel"] = str(e)
        geom["bbox_metod"] = "MISSING -- optimalBoundingBox failed"
    if bb is None:
        geom["bbox_mm"] = None
        geom["L_mm"] = geom["B_mm"] = geom["H_mm"] = None
        bb = bb_pol           # for view placement only, NEVER for dimensions
        geom["mattbara"] = False
    else:
        geom["bbox_mm"] = [round(bb.XMin, 6), round(bb.YMin, 6), round(bb.ZMin, 6),
                           round(bb.XMax, 6), round(bb.YMax, 6), round(bb.ZMax, 6)]
        geom["L_mm"] = round(bb.XLength, 6)
        geom["B_mm"] = round(bb.YLength, 6)
        geom["H_mm"] = round(bb.ZLength, 6)
        geom["mattbara"] = True
        geom["pol_overskatting_mm"] = [round(bb_pol.XLength - bb.XLength, 4),
                                       round(bb_pol.YLength - bb.YLength, 4),
                                       round(bb_pol.ZLength - bb.ZLength, 4)]

    if job.get("holes", True):
        _SOLIDS.extend(comp.Solids)
        geom["holes"] = measure_holes(comp)
        geom["cyl_log"] = _CYL_LOG[:200]
        geom["n_cyl"] = len(_CYL_LOG)
    else:
        geom["holes"] = []

    page = doc.addObject("TechDraw::DrawPage", "Page")
    tpl = doc.addObject("TechDraw::DrawSVGTemplate", "Tpl")
    tpl.Template = os.path.join(template_dir(), "A3_Landscape_blank.svg")
    page.Template = tpl

    views = {}
    for spec in job["views"]:
        v = doc.addObject("TechDraw::DrawViewPart", "V_" + spec["id"])
        page.addView(v)
        v.Source = [obj]
        v.Direction = _vec(spec["dir"])
        v.XDirection = _vec(spec["xdir"])
        v.Scale = 1.0
        v.ScaleType = "Custom"
        v.HardHidden = bool(spec.get("hidden", False))
        v.CoarseView = bool(spec.get("coarse", False))
        v.SmoothVisible = bool(spec.get("smooth", True))
        doc.recompute()
        try:
            svg = TechDraw.viewPartAsSvg(v)
        except Exception as e:
            views[spec["id"]] = {"fel": "%s: %s" % (type(e).__name__, e)}
            continue
        fallback = None
        if svg.count("<path") == 0:
            # EMPTY VIEW: the exact HLR failed silently on this direction. Try the polygonal
            # (CoarseView) one once and RECORD that the view is coarse -- an empty view on a sheet
            # is a lie.
            v.CoarseView = True
            doc.recompute()
            try:
                svg2 = TechDraw.viewPartAsSvg(v)
            except Exception:
                svg2 = ""
            if svg2.count("<path") > 0:
                svg = svg2
                fallback = "coarse (exact HLR gave 0 paths)"
            else:
                fallback = "FAILED: both exact and coarse HLR gave 0 paths"
        calib = calibrate_frame(v, comp, svg)
        gc = v.getGeometricCenter()
        pc = v.projectPoint(gc)

        def to_svg(P, _pc=pc, _v=v):
            q = _v.projectPoint(P)
            return [round(q.x - _pc.x, 6), round(-(q.y - _pc.y), 6)]

        dv = _vec(spec["dir"])
        dv.normalize()
        holes2d = []
        for h in geom.get("holes", []):
            ax = h["axis"]
            if abs(ax[0] * dv.x + ax[1] * dv.y + ax[2] * dv.z) > 0.999:
                c = h["center"]
                uv = to_svg(FreeCAD.Vector(c[0], c[1], c[2]))
                holes2d.append({"d_mm": h["d_mm"], "u": uv[0], "vv": uv[1], "depth_mm": h["depth_mm"]})
        corners = []
        bx = [bb.XMin, bb.XMax]
        by = [bb.YMin, bb.YMax]
        bz = [bb.ZMin, bb.ZMax]
        for xx in bx:
            for yy in by:
                for zz in bz:
                    corners.append(to_svg(FreeCAD.Vector(xx, yy, zz)))
        views[spec["id"]] = {
            "svg": svg,
            "bbox_svg": svg_path_bbox(svg),
            "calib": calib,
            "dir": spec["dir"], "xdir": spec["xdir"],
            "n_paths": svg.count("<path"),
            "hlr_fallback": fallback,
            "holes_2d": holes2d,
            "bbox3d_corners_svg": corners,
        }

    json.dump({
        "freecad_version": FreeCAD.Version(),
        "geom": geom,
        "parts": part_meas,
        "views": views,
    }, open(out_path, "w"))


main()
