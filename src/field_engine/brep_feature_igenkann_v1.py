#!/usr/bin/env python3
"""Feature recognition at B-rep level: from faces to features.

The missing link between mesh -> faces and sketch -> features -> parametric CAD. Classical AAG
subgraph matching on top of brep_aag_v1 -- no training, no weights.

Every class has a mechanical definition, not a guess:

  hole_through   CYLINDER, concavity=INWARD, both end circles lie on an INNER wire of their
                 neighbour face (i.e. the cylinder pierces the neighbour face).
  hole_blind     as above but AT LEAST one end is terminated by a face where the end circle lies on
                 the neighbour's OUTER wire (= a bottom), or by a cone whose apex lies on the axis
                 (drill tip).
  counterbore    two coaxial INWARD cylinders of different radius joined by an annular PLANE
                 perpendicular to the axis (the step).
  countersink    an INWARD CONE coaxial with and adjacent to an INWARD cylinder.
  pocket         a PLANE all of whose edge links are CONCAVE (the bottom face). Depth = the distance
                 to the opening plane (the upper edge ring of the side walls).
  boss           a PLANE host face H with an INNER wire whose neighbour faces lie on the +normal
                 side of H (material rises out of the host). The dual of the hole/pocket opening
                 (-normal side).
  fillet         CYLINDER/TORUS/SPHERE whose edge links are TANGENT to both lateral neighbours
                 (>=2 TANGENT links). INWARD => concave fillet (inside corner), OUTWARD => round.
                 A post-operation in the recipe -- never an extrude of its own.
  chamfer        (a) a CONE with exactly 2 circular CONVEX edges, a coaxial neighbour cylinder and a
                 plane perpendicular to the axis; (b) a PLANE of small area whose normal lies
                 BETWEEN two CONVEX neighbours' normals (n.na>0 and n.nb>0, na.nb<n.na and na.nb<n.nb).
  revolve_body   there is ONE axis A such that EVERY face is rotationally symmetric about A.
  extrude_body   there is ONE direction d such that every face is a PLANE with n||d, a PLANE with
                 n.d=0, or a cone-free side wall with axis||d.
  pattern_*      >=3 holes with an IDENTICAL signature (type, radius, depth) whose centres lie on a
                 circle at equal angular spacing (circular) or on a line at equal spacing (linear).
  rib            a boss whose footprint has length/thickness >= RIB_ASPECT and thickness <= RIB_MAX_T.

I/O: a build123d/OCC solid in, a feature list + a type histogram out; --selftest builds fixtures with
a known operation sequence and compares the recognised histogram against that reference.
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build123d as bd  # noqa: E402
from brep_aag_v1 import (AAG, BRepTools, _add, _cross, _dot, _norm, _scale,  # noqa: E402
                         _sub, _wire_edges, build_aag)
from OCP.TopAbs import TopAbs_EDGE  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

ANG_TOL = 1e-3           # riktnings-parallellitet: |1-|cos|| < ANG_TOL
AXIS_DIST_TOL = 1e-4     # koaxialitet (mm)
FILLET_MAX_R = 25.0      # above this radius it is not a fillet but a load-carrying cylinder (mm)
CHAMFER_MAX_AREA_FRAC = 0.06   # a chamfer plane may hold at most this fraction of the total surface area
CHAMFER_LEG_TOL_FRAC = 1e-6    # above this relative leg difference the band is an asymmetric chamfer
FILLET_LEG_TOL_FRAC = 1e-6     # a circular fillet's two tangent legs are equal; above this it is not one
RIB_ASPECT = 4.0
RIB_MAX_T = 30.0
PATTERN_TOL = 1e-3
FULL_TURN_FRAC = 0.9     # the cylinder face u sweep must be >= this x 2pi to be a HOLE


def _par(a, b) -> bool:
    return abs(abs(_dot(_norm(a), _norm(b))) - 1.0) < ANG_TOL


def _perp(a, b) -> bool:
    return abs(_dot(_norm(a), _norm(b))) < ANG_TOL


def _pt_axis_dist(p, ap, ad) -> float:
    d = _sub(p, ap)
    return math.dist(d, _scale(_norm(ad), _dot(d, _norm(ad))))


def _coaxial(f1: dict, f2: dict) -> bool:
    a1, a2 = f1.get("axis_dir"), f2.get("axis_dir")
    p1, p2 = f1.get("axis_point"), f2.get("axis_point")
    if a1 is None or a2 is None:
        return False
    return _par(a1, a2) and _pt_axis_dist(p2, p1, a1) < AXIS_DIST_TOL


def _broken_edge_line(fa: dict, fb: dict):
    """The line the band broke: the intersection of the planes of fa and fb, as (point, direction).

    The direction is na x nb and one point on it is the solution of the two plane equations inside
    the plane spanned by the two normals (2x2 Gram system). Returns None when the two planes are
    parallel, i.e. there is no edge between them to break.
    """
    na, nb = fa["sample_normal"], fb["sample_normal"]
    if na is None or nb is None:
        return None
    t = _cross(na, nb)
    if math.sqrt(_dot(t, t)) < 1e-9:
        return None
    t = _norm(t)
    ca, cb = _dot(na, fa["sample_point"]), _dot(nb, fb["sample_point"])
    g = _dot(na, nb)
    det = 1.0 - g * g
    if abs(det) < 1e-12:
        return None
    alpha = (ca - g * cb) / det
    beta = (cb - g * ca) / det
    return _add(_scale(na, alpha), _scale(nb, beta)), t


def _band_legs(p0, t, la: dict, lb: dict):
    """Perpendicular distances from the broken-edge line to the two new edges, and the line point
    nearest their common midpoint. Each distance is measured in the face the edge lies in, which is
    exactly the leg the corresponding chamfer/fillet operation consumed."""
    def _dist_to_line(q):
        v = _sub(q, p0)
        return math.dist(v, _scale(t, _dot(v, t)))

    leg_a, leg_b = _dist_to_line(la["mid"]), _dist_to_line(lb["mid"])
    mid = _scale(_add(la["mid"], lb["mid"]), 0.5)
    v = _sub(mid, p0)
    return leg_a, leg_b, _add(p0, _scale(t, _dot(v, t)))


def _planar_chamfer_geometry(f: dict, fa: dict, fb: dict, la: dict, lb: dict) -> dict | None:
    """Measure the leg lengths of a planar chamfer band.

    `f` is the chamfer face, `fa`/`fb` the two planar faces it breaks between and `la`/`lb` the two
    convex edges linking them. Both legs are ALWAYS reported (leg_a/leg_b), together with the
    midpoint and direction of the broken edge. `distance` -- the single scalar a `chamfer` op takes
    -- is only set when the two legs agree to within CHAMFER_LEG_TOL_FRAC; an asymmetric band is
    reported with its two measured legs and `symmetric` False and carries no `distance`, because one
    scalar cannot express two legs. Returns None only when the two planes are parallel (no edge to
    break) or when both legs measure zero.
    """
    line = _broken_edge_line(fa, fb)
    if line is None:
        return None
    p0, t = line
    leg_a, leg_b, edge_mid = _band_legs(p0, t, la, lb)
    if max(leg_a, leg_b) < 1e-9:
        return None
    sym = abs(leg_a - leg_b) / max(leg_a, leg_b) <= CHAMFER_LEG_TOL_FRAC
    out = {"leg_a": leg_a, "leg_b": leg_b, "symmetric": sym,
           "leg_ratio": max(leg_a, leg_b) / min(leg_a, leg_b) if min(leg_a, leg_b) > 1e-12 else None,
           "edge_point": list(edge_mid), "edge_dir": list(t),
           "edge_length": min(la.get("length", 0.0), lb.get("length", 0.0))}
    if sym:
        out["distance"] = 0.5 * (leg_a + leg_b)
    return out


def _edge_fillet_geometry(aag: AAG, f: dict) -> dict | None:
    """Locate a constant-radius fillet band that rounded ONE edge between two planar faces.

    `f` is the fillet face. The band qualifies when exactly two of its links are TANGENT and both
    lead to a PLANE: those two planes are the faces the rounded edge ran between, so the same
    plane-intersection line the chamfer path uses gives the edge that was rounded, and its midpoint
    is what a located selector needs. The two tangent lengths from that line to the two tangent
    edges are equal for a circular fillet by construction, so a band whose measured legs disagree by
    more than FILLET_LEG_TOL_FRAC is not this class and is reported without a location.
    """
    tan = [l for l in aag.links_of(f["idx"]) if l["kind"] == "TANGENT"
           and aag.faces[aag.other(l, f["idx"])]["type"] == "PLANE"]
    if len(tan) != 2:
        return None
    fa, fb = (aag.faces[aag.other(l, f["idx"])] for l in tan)
    line = _broken_edge_line(fa, fb)
    if line is None:
        return None
    p0, t = line
    leg_a, leg_b, edge_mid = _band_legs(p0, t, tan[0], tan[1])
    if max(leg_a, leg_b) < 1e-9:
        return None
    if abs(leg_a - leg_b) / max(leg_a, leg_b) > FILLET_LEG_TOL_FRAC:
        return None
    return {"tangent_leg": 0.5 * (leg_a + leg_b), "edge_point": list(edge_mid), "edge_dir": list(t),
            "edge_length": min(tan[0].get("length", 0.0), tan[1].get("length", 0.0))}


def _edge_on_outer_wire(aag: AAG, face_idx: int, mid_pt, tol=1e-6) -> bool | None:
    """Does the edge (identified by its midpoint) lie on `face_idx`'s OUTER wire?
    True => the neighbour face is TERMINATED by the edge (bottom/end). False => INNER wire, i.e. the edge is a
    HAL i grannytan (genomborrning)."""
    f = aag.face_shape(face_idx)
    try:
        ow = BRepTools.OuterWire_s(f)
    except Exception:
        return None
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.gp import gp_Pnt, gp_Vec
    for e in _wire_edges(ow):
        ac = BRepAdaptor_Curve(e)
        um = 0.5 * (ac.FirstParameter() + ac.LastParameter())
        p, d1 = gp_Pnt(), gp_Vec()
        try:
            ac.D1(um, p, d1)
        except Exception:
            continue
        if math.dist((p.X(), p.Y(), p.Z()), mid_pt) < 1e-6:
            return True
    return False


# --------------------------------------------------------------------- kropps-symmetrier
def _axis_candidates(aag: AAG) -> list[tuple]:
    cands = []
    for f in aag.faces:
        if f["type"] in ("CYLINDER", "CONE", "TORUS"):
            cands.append((f["axis_point"], _norm(f["axis_dir"])))
    for f in aag.faces:
        if f["type"] == "PLANE":
            cands.append((f["origin"], _norm(f["normal"])))
    out = []
    for ap, ad in cands:
        if not any(_par(ad, b) and _pt_axis_dist(ap, bp, b) < 1e-6 for bp, b in out):
            out.append((ap, ad))
    return out


def _is_revolve_about(aag: AAG, ap, ad, skip=frozenset()) -> bool:
    for f in aag.faces:
        if f["idx"] in skip:
            continue
        t = f["type"]
        if t == "PLANE":
            if not _par(f["normal"], ad):
                return False
            continue
        if t in ("CYLINDER", "CONE", "TORUS"):
            if not (_par(f["axis_dir"], ad) and _pt_axis_dist(f["axis_point"], ap, ad) < AXIS_DIST_TOL):
                return False
            continue
        if t == "SPHERE":
            if _pt_axis_dist(f["sphere_center"], ap, ad) >= AXIS_DIST_TOL:
                return False
            continue
        return False
    return True


def _is_extrude_along(aag: AAG, d, skip=frozenset()) -> bool:
    for f in aag.faces:
        if f["idx"] in skip:
            continue
        t = f["type"]
        if t == "PLANE":
            if not (_par(f["normal"], d) or _perp(f["normal"], d)):
                return False
            continue
        if t == "CYLINDER":
            if not _par(f["axis_dir"], d):
                return False
            continue
        return False
    return True


# --------------------------------------------------------------------- huvudigenkannaren
def recognize(shape, aag: AAG | None = None) -> dict:
    aag = aag or build_aag(shape)
    feats: list[dict] = []
    total_area = sum(f["area"] for f in aag.faces)

    # ---- 1. blends (fillet/round): the TANGENT signature. Done FIRST so the hole recogniser can
    #         tell a real bore from a fillet cylinder.
    blend_faces = set()
    for f in aag.faces:
        if f["type"] not in ("CYLINDER", "TORUS", "SPHERE"):
            continue
        ls = aag.links_of(f["idx"])
        ntan = sum(1 for l in ls if l["kind"] == "TANGENT")
        if ntan < 2:
            continue
        r = f.get("radius") if f["type"] != "TORUS" else f.get("minor_radius")
        if r is None or r > FILLET_MAX_R:
            continue
        blend_faces.add(f["idx"])
        feat = {
            "type": "fillet", "faces": [f["idx"]], "surface": f["type"],
            "radius": r, "concavity": f["concavity"],
            "kind": "concave_fillet" if f["concavity"] == "INWARD" else "convex_round",
            "n_tangent_links": ntan, "area": f["area"],
            "post_process": True,
        }
        # Locate the band: which single edge it rounded, so it can be emitted as a fillet op
        # addressed by that edge's midpoint instead of only by type and count. A band that does not
        # run between exactly two planes (a corner patch, a fillet along a curved edge) is reported
        # without a location rather than with a guessed one.
        geom = _edge_fillet_geometry(aag, f)
        if geom is not None:
            feat.update(geom)
        feats.append(feat)

    # ---- 1b. TYPE-INDEPENDENT transition blend: a face whose edge links are ALL TANGENT has no
    #          sharp boundary against anything -- by definition a pure transition face.
    #          MEASURED: the corner transitions where an r1.5 fillet meets an r2 corner radius are
    #          represented by OCC as SurfaceOfRevolution (neither cylinder nor torus) and
    #          would otherwise block the whole extrude_body test.
    for f in aag.faces:
        if f["idx"] in blend_faces:
            continue
        ls = aag.links_of(f["idx"])
        if len(ls) >= 2 and all(l["kind"] == "TANGENT" for l in ls) and f["type"] != "PLANE":
            blend_faces.add(f["idx"])
            feats.append({"type": "fillet", "faces": [f["idx"]], "surface": f["type"],
                          "radius": None, "kind": "blend_transition",
                          "n_tangent_links": len(ls), "area": f["area"], "post_process": True})

    # ---- 2. holes (cylindrical INWARD faces that are NOT blends)
    holes = []
    hole_faces: set[int] = set()
    for f in aag.faces:
        if f["type"] != "CYLINDER" or f["concavity"] != "INWARD" or f["idx"] in blend_faces:
            continue
        # A HOLE SWEEPS THE FULL TURN. A rounded slot corner (a RectangleRounded contour cut
        # through the body) also gives an INWARD cylinder face through both outer faces -- but only a
        # 90-graders svep. MATT: zslid_plattas fonster (r_corner=32) gav 4 falska "genomgaende
        # hole" plus a false pattern_circular before this sweep condition existed.
        if f.get("u_span") is None or f["u_span"] < FULL_TURN_FRAC * 2 * math.pi:
            continue
        ad = _norm(f["axis_dir"])
        ap = f["axis_point"]
        ends = []
        for l in aag.links_of(f["idx"]):
            if l.get("edge_type") != "CIRCLE":
                continue
            nb = aag.other(l, f["idx"])
            nbf = aag.faces[nb]
            onouter = _edge_on_outer_wire(aag, nb, l["mid"])
            ends.append({"link": l, "nb": nb, "nb_type": nbf["type"], "outer": onouter,
                         "t": _dot(_sub(l["circle_center"], ap), ad)})
        if not ends:
            continue
        ts = [e["t"] for e in ends]
        depth = max(ts) - min(ts) if len(ts) >= 2 else None
        bottoms = [e for e in ends if e["outer"] is True]
        # a cone whose apex lies on the axis and terminates the cylinder = drill tip
        tip = [e for e in ends if e["nb_type"] == "CONE"
               and _pt_axis_dist(aag.faces[e["nb"]].get("apex", (1e9, 0, 0)), ap, ad) < 1e-3]
        blind = bool(bottoms) or bool(tip)
        cen = _add(ap, _scale(ad, 0.5 * (max(ts) + min(ts)))) if len(ts) >= 2 else f["sample_point"]
        h = {
            "type": "hole_blind" if blind else "hole_through",
            "faces": [f["idx"]], "radius": f["radius"], "diameter": 2 * f["radius"],
            "axis_dir": ad, "axis_point": ap, "center": cen, "depth": depth,
            "n_ends": len(ends), "n_bottoms": len(bottoms), "drill_tip": bool(tip),
        }
        holes.append(h)
        hole_faces.add(f["idx"])
        feats.append(h)

    # ---- 3. forsankning: koaxiala hal-cylindrar / kon
    for i, h1 in enumerate(holes):
        f1 = aag.faces[h1["faces"][0]]
        for h2 in holes[i + 1:]:
            f2 = aag.faces[h2["faces"][0]]
            if not _coaxial(f1, f2) or abs(f1["radius"] - f2["radius"]) < 1e-6:
                continue
            # the step: an annular PLANE perpendicular to the axis adjoining both
            n1 = {aag.other(l, f1["idx"]) for l in aag.links_of(f1["idx"])}
            n2 = {aag.other(l, f2["idx"]) for l in aag.links_of(f2["idx"])}
            step = [k for k in (n1 & n2) if aag.faces[k]["type"] == "PLANE"
                    and _par(aag.faces[k]["normal"], f1["axis_dir"])]
            if step:
                big, small = (h1, h2) if h1["radius"] > h2["radius"] else (h2, h1)
                feats.append({"type": "counterbore",
                              "faces": big["faces"] + small["faces"] + step,
                              "cbore_diameter": 2 * big["radius"],
                              "hole_diameter": 2 * small["radius"],
                              "axis_dir": big["axis_dir"], "axis_point": big["axis_point"]})
    for f in aag.faces:
        if f["type"] != "CONE" or f["concavity"] != "INWARD":
            continue
        for l in aag.links_of(f["idx"]):
            nb = aag.faces[aag.other(l, f["idx"])]
            if nb["type"] == "CYLINDER" and nb["concavity"] == "INWARD" and _coaxial(f, nb):
                feats.append({"type": "countersink", "faces": [f["idx"], nb["idx"]],
                              "half_angle_deg": f["half_angle_deg"],
                              "hole_diameter": 2 * nb["radius"],
                              "axis_dir": _norm(f["axis_dir"]), "axis_point": f["axis_point"]})
                break

    # ---- 4. chamfer. NO absolute area threshold: a chamfer is defined by BREAKING an edge between
    #         two other faces, not by being "small". That threshold (6 % of total surface area)
    #         failed on a lock nut -- a 2 mm chamfer on a 16 mm high nut takes 7.8 % of the area and
    #         was silently DISCARDED, which in turn destroyed the extrude_body test because the cone
    #         faces could then not be skipped.
    for f in aag.faces:
        ls = aag.links_of(f["idx"])
        if f["type"] == "CONE":
            circ = [l for l in ls if l.get("edge_type") == "CIRCLE"]
            if len(circ) == 2 and all(l["kind"] == "CONVEX" for l in circ):
                nbs = [aag.faces[aag.other(l, f["idx"])] for l in circ]
                types = sorted(n["type"] for n in nbs)
                cyl = [n for n in nbs if n["type"] == "CYLINDER"]
                pln = [n for n in nbs if n["type"] == "PLANE"]
                if types == ["CYLINDER", "PLANE"] and _coaxial(f, cyl[0]) \
                        and _par(pln[0]["normal"], f["axis_dir"]):
                    feats.append({"type": "chamfer", "faces": [f["idx"]], "surface": "CONE",
                                  "half_angle_deg": f["half_angle_deg"], "area": f["area"],
                                  "axial_extent": abs(circ[0]["circle_radius"] - circ[1]["circle_radius"]),
                                  "post_process": True})
                    continue
        if f["type"] == "PLANE":
            cvx = [l for l in ls if l["kind"] == "CONVEX"]
            n = f["sample_normal"]
            if n is None or len(cvx) < 2:
                continue
            hit = None
            hit_pair = None
            asym = None
            asym_pair = None
            for i in range(len(cvx)):
                for j in range(i + 1, len(cvx)):
                    fa = aag.faces[aag.other(cvx[i], f["idx"])]
                    fb = aag.faces[aag.other(cvx[j], f["idx"])]
                    if fa["type"] != "PLANE" or fb["type"] != "PLANE":
                        continue        # chamfer between two PLANES; cone/cylinder neighbours are handled above
                    na, nb = fa["sample_normal"], fb["sample_normal"]
                    if na is None or nb is None:
                        continue
                    bis = _norm(_add(na, nb))
                    if bis == (0.0, 0.0, 0.0):
                        continue
                    ang = math.degrees(math.acos(max(-1, min(1, _dot(n, bis)))))
                    smaller = f["area"] < fa["area"] and f["area"] < fb["area"]
                    # scale-free size rule: the chamfer face is SMALLER than both faces it breaks between
                    if ang < 15.0 and smaller:
                        hit = (ang, math.degrees(math.acos(max(-1, min(1, _dot(na, nb))))))
                        hit_pair = (fa, fb, cvx[i], cvx[j])
                    elif smaller and _dot(n, na) > 0.0 and _dot(n, nb) > 0.0:
                        # UNEQUAL-LEG band. The bisector rule above is a SYMMETRY rule: a 2x4 chamfer
                        # on a right-angled corner sits 18.43 degrees off the bisector and fails it,
                        # so it used to be classified as no feature at all -- which also cost the
                        # part its extrude_body direction, since an unclassified band cannot be
                        # skipped. This branch does not relax that rule (the symmetric hit still wins
                        # whenever it is found); it adds an independent criterion for the same solid
                        # geometry: the band lies BETWEEN the two convex neighbours (positive dot
                        # with both normals, the class definition in this module's docstring) and is
                        # a prism along the edge they intersect in (its normal is perpendicular to
                        # that intersection direction), which is what breaking an edge means.
                        line = _broken_edge_line(fa, fb)
                        if line is not None and abs(_dot(n, line[1])) < 1e-6:
                            asym = (ang, math.degrees(math.acos(max(-1, min(1, _dot(na, nb))))))
                            asym_pair = (fa, fb, cvx[i], cvx[j])
            pick, pair = (hit, hit_pair) if hit else (asym, asym_pair)
            if pick:
                feat = {"type": "chamfer", "faces": [f["idx"]], "surface": "PLANE",
                        "bisector_dev_deg": pick[0], "between_angle_deg": pick[1],
                        "area": f["area"], "post_process": True}
                # Measure the chamfer's own legs and the edge it broke, so the band can be emitted as
                # a chamfer op instead of only being classified. Both legs are always reported; the
                # single `distance` a chamfer op takes is only set when they are equal, so an
                # asymmetric band is reported with its two measured legs and no guessed scalar.
                geom = _planar_chamfer_geometry(f, pair[0], pair[1], pair[2], pair[3])
                if geom is not None:
                    feat.update(geom)
                feats.append(feat)

    # ---- 5. pocket (bottom face with EXCLUSIVELY concave edges)
    for f in aag.faces:
        if f["type"] != "PLANE":
            continue
        ls = aag.links_of(f["idx"])
        if not ls or not all(l["kind"] == "CONCAVE" for l in ls):
            continue
        walls = [aag.other(l, f["idx"]) for l in ls]
        # a blind hole BOTTOM is not a pocket of its own -- it is already the hole end condition
        if all(w in hole_faces for w in walls):
            continue
        n = f["sample_normal"]
        # depth: the longest extent of the walls along -n from the bottom
        depth = 0.0
        for w in walls:
            for l2 in aag.links_of(w):
                depth = max(depth, _dot(_sub(l2["mid"], f["sample_point"]), n))
        feats.append({"type": "pocket", "faces": [f["idx"]], "bottom_face": f["idx"],
                      "walls": walls, "n_walls": len(walls), "depth": depth,
                      "bottom_area": f["area"], "normal": n})

    # ---- 6. boss (host face with an inner wire whose neighbours rise out of the host)
    for f in aag.faces:
        if f["type"] != "PLANE":
            continue
        n = f["sample_normal"]
        if n is None:
            continue
        for w in aag.inner_wires(f["idx"]):
            mids = []
            for e in _wire_edges(w):
                from OCP.BRepAdaptor import BRepAdaptor_Curve
                from OCP.gp import gp_Pnt, gp_Vec
                ac = BRepAdaptor_Curve(e)
                um = 0.5 * (ac.FirstParameter() + ac.LastParameter())
                p, d1 = gp_Pnt(), gp_Vec()
                try:
                    ac.D1(um, p, d1)
                except Exception:
                    continue
                mids.append((p.X(), p.Y(), p.Z()))
            nbs = set()
            for l in aag.links_of(f["idx"]):
                if any(math.dist(l["mid"], m) < 1e-6 for m in mids):
                    nbs.add(aag.other(l, f["idx"]))
            if not nbs:
                continue
            side = 0.0
            for k in nbs:
                sp = aag.faces[k]["sample_point"]
                if sp is None:
                    continue
                side += _dot(_sub(sp, f["sample_point"]), n)
            if all(k in hole_faces or k in blend_faces for k in nbs):
                continue        # a hole or blend in the host face is not a boss (duality guard)
            if side > 1e-6:
                feats.append({"type": "boss", "host_face": f["idx"], "faces": sorted(nbs),
                              "n_side_faces": len(nbs), "normal": n})

    # ---- 7. body symmetries of the BASE BODY (fillets and chamfers are post-operations and must
    #         therefore not be able to destroy an extrude/revolve -- exactly what a recipe needs:
    #         base verb + post verb, never a face that blocks the base)
    skip = set()
    for x in feats:
        if x["type"] in ("fillet", "chamfer"):
            skip.update(x["faces"])
    for ap, ad in _axis_candidates(aag):
        if _is_revolve_about(aag, ap, ad, skip):
            feats.append({"type": "revolve_body", "axis_point": ap, "axis_dir": ad,
                          "n_base_faces": len(aag.faces) - len(skip)})
            break
    for ap, ad in _axis_candidates(aag):
        if _is_extrude_along(aag, ad, skip):
            feats.append({"type": "extrude_body", "direction": ad,
                          "n_base_faces": len(aag.faces) - len(skip)})
            break

    # ---- 8. monster
    feats.extend(_patterns(holes))

    # ---- 9. rib: a boss with a long, thin footprint
    for b in [x for x in feats if x["type"] == "boss"]:
        exts = []
        for k in b["faces"]:
            fs = aag.faces[k]
            if fs["type"] != "PLANE":
                exts = []
                break
            exts.append(fs)
        if len(exts) >= 4:
            dims = _wall_dims(aag, b["faces"])
            if dims and dims[0] >= RIB_ASPECT * dims[1] and dims[1] <= RIB_MAX_T:
                feats.append({"type": "rib", "faces": b["faces"],
                              "run_length": dims[0], "thickness": dims[1]})

    return {"summary": aag.summary(), "features": feats,
            "counts": _counts(feats), "total_area": total_area}


def _wall_dims(aag: AAG, faces: list[int]):
    pts = []
    for k in faces:
        sp = aag.faces[k]["sample_point"]
        if sp:
            pts.append(sp)
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    d = sorted([max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)], reverse=True)
    return (d[0], d[2]) if d[2] > 1e-9 else (d[0], d[1])


def _patterns(holes: list[dict]) -> list[dict]:
    out = []
    groups: dict[tuple, list[dict]] = {}
    for h in holes:
        key = (h["type"], round(h["radius"], 6),
               round(h["depth"], 4) if h["depth"] is not None else None,
               tuple(round(c, 6) for c in _norm(h["axis_dir"])))
        groups.setdefault(key, []).append(h)
    for key, g in groups.items():
        if len(g) < 3:
            continue
        cs = [h["center"] for h in g]
        ad = _norm(g[0]["axis_dir"])
        cen = tuple(sum(c[i] for c in cs) / len(cs) for i in range(3))
        rs = [_pt_axis_dist(c, cen, ad) for c in cs]
        if max(rs) - min(rs) < 1e-4 and max(rs) > 1e-6:
            e1 = _norm(_sub(cs[0], _add(cen, _scale(ad, _dot(_sub(cs[0], cen), ad)))))
            e2 = _cross(ad, e1)
            angs = sorted(math.degrees(math.atan2(_dot(_sub(c, cen), e2),
                                                  _dot(_sub(c, cen), e1))) % 360.0 for c in cs)
            steps = [(angs[i + 1] - angs[i]) for i in range(len(angs) - 1)]
            steps.append(360.0 - sum(steps))
            if max(steps) - min(steps) < 1e-3:
                out.append({"type": "pattern_circular", "count": len(g),
                            "axis_point": cen, "axis_dir": ad,
                            "bolt_circle_radius": sum(rs) / len(rs),
                            "member_signature": {"hole": key[0], "radius": key[1], "depth": key[2]}})
                continue
        # linear: collinear centres with equal pitch
        d0 = _norm(_sub(cs[1], cs[0]))
        proj = sorted(_dot(_sub(c, cs[0]), d0) for c in cs)
        offaxis = max(math.dist(c, _add(cs[0], _scale(d0, _dot(_sub(c, cs[0]), d0)))) for c in cs)
        steps = [proj[i + 1] - proj[i] for i in range(len(proj) - 1)]
        if offaxis < 1e-4 and max(steps) - min(steps) < 1e-4:
            out.append({"type": "pattern_linear", "count": len(g), "direction": d0,
                        "spacing": steps[0], "start": cs[0],
                        "member_signature": {"hole": key[0], "radius": key[1], "depth": key[2]}})
    return out


def _counts(feats) -> dict:
    from collections import Counter
    return dict(Counter(f["type"] for f in feats))


def recognize_step(path: str) -> dict:
    shape = bd.import_step(path)
    solids = shape.solids()
    if len(solids) > 1:
        shape = solids[0]
    r = recognize(shape)
    r["path"] = path
    return r


# --------------------------------------------------------------------- fallbevis
def _selftest() -> dict:
    """Positive-control fixtures with a KNOWN reference (built from a KNOWN operation sequence)."""
    res = {}
    cases = {}

    # F1: plate with 4 through holes in a rectangular pattern + 1 blind hole
    p = bd.Box(60, 60, 10)
    for (x, y) in [(-20, -20), (20, -20), (20, 20), (-20, 20)]:
        p -= bd.Pos(x, y, 0) * bd.Cylinder(radius=3, height=30)
    p -= bd.Pos(0, 0, 5) * bd.Cylinder(radius=6, height=10)   # blindhal djup 5
    cases["plate_4holes_1blind"] = (p, {"hole_through": 4, "hole_blind": 1, "pocket": 0})

    # F2: flange with a bolt circle (6 holes) -- circular pattern
    fl = bd.Cylinder(radius=50, height=8)
    for k in range(6):
        a = math.radians(60 * k)
        fl -= bd.Pos(35 * math.cos(a), 35 * math.sin(a), 0) * bd.Cylinder(radius=4, height=30)
    cases["flange_6bolt"] = (fl, {"hole_through": 6, "pattern_circular": 1, "revolve_body": 0})

    # F3: ficka
    pk = bd.Box(60, 60, 20) - bd.Pos(0, 0, 12) * bd.Box(30, 30, 20)
    cases["pocket"] = (pk, {"pocket": 1})

    # F4: boss
    bo = bd.Box(60, 60, 10) + bd.Pos(0, 0, 10) * bd.Cylinder(radius=8, height=10)
    cases["boss"] = (bo, {"boss": 1})

    # F5: chamfered shaft (chamfer on both cylinder ends) + filleted box
    ax = bd.Cylinder(radius=10, height=40)
    ax = bd.chamfer(ax.edges().filter_by(bd.GeomType.CIRCLE), 2.0)
    cases["chamfered_shaft"] = (ax, {"chamfer": 2, "revolve_body": 1})

    fb = bd.fillet(bd.Box(40, 40, 20).edges().filter_by(bd.Axis.Z), 5.0)
    cases["filleted_box"] = (fb, {"fillet": 4, "extrude_body": 1})

    ok = True
    for name, (shp, facit) in cases.items():
        r = recognize(shp)
        c = r["counts"]
        got = {k: c.get(k, 0) for k in facit}
        res[name] = {"facit": facit, "got": got, "all_counts": c}
        res[name]["pass"] = all(got[k] == v for k, v in facit.items())
        ok = ok and res[name]["pass"]
    res["ALL_PASS"] = ok
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = _selftest()
        print(json.dumps(r, indent=2, default=str))
        sys.exit(0 if r["ALL_PASS"] else 1)
    for p in sys.argv[1:]:
        r = recognize_step(p)
        print(os.path.basename(p), json.dumps(r["counts"]), r["summary"])
