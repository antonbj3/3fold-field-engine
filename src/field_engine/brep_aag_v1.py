#!/usr/bin/env python3
"""Attributed adjacency graph (AAG) from a "dumb" B-rep solid.

The missing link from an imported B-rep solid without operation history to a recipe. This is the
substrate the learned B-rep networks build on, done classically: no training, no weights.

Per FACE: surface type (PLANE/CYLINDER/CONE/SPHERE/TORUS/BSPLINE/...), area, and the surface's own
parameters (plane: origin + normal; cylinder: axis + radius; cone: axis + half angle + reference
radius; torus: axis + R + r).
Per EDGE PAIR: the shared edge, a dihedral class {CONVEX, CONCAVE, TANGENT} and the angle in degrees.

The dihedral convention is derived, not guessed:
  For a face F with OUTWARD normal n1, its boundary wire is traversed so that the face interior lies
  to the LEFT when walking along the edge tangent t with the head along n1. Therefore
        w1 = n1 x t
  points INTO F's interior. The neighbour face F2 has outward normal n2, and
        w1 . n2 < 0  =>  CONVEX edge   (the neighbour folds AWAY from F's interior)
        w1 . n2 > 0  =>  CONCAVE edge
        w1 . n2 ~ 0  =>  TANGENT (smooth transition -- the fillet signature)
  t is taken with the edge orientation AS IT APPEARS IN F (TopAbs_REVERSED flips t), and n1/n2 are
  flipped when the face is TopAbs_REVERSED. Calibration and planted-fault fixtures live in
  _selftest(): a box has ONLY convex edges; a through hole gives ONLY convex rim edges; a fillet
  gives TANGENT.

I/O: a build123d/OCC solid in, an AAG object (faces, links, summary dict) out; --selftest builds six
fixtures and checks the convention on each.
"""
from __future__ import annotations

import math
import sys

import build123d as bd
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Pnt, gp_Vec
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopTools import (TopTools_IndexedDataMapOfShapeListOfShape,
                          TopTools_IndexedMapOfShape)

TOL_TANGENT_DEG = 1.5      # |dihedralvinkel| under detta => TANGENT (slat)
_ST = GeomAbs_SurfaceType


def _stype_name(t) -> str:
    return {
        _ST.GeomAbs_Plane: "PLANE", _ST.GeomAbs_Cylinder: "CYLINDER",
        _ST.GeomAbs_Cone: "CONE", _ST.GeomAbs_Sphere: "SPHERE",
        _ST.GeomAbs_Torus: "TORUS", _ST.GeomAbs_BezierSurface: "BEZIER",
        _ST.GeomAbs_BSplineSurface: "BSPLINE",
        _ST.GeomAbs_SurfaceOfRevolution: "REVOLUTION",
        _ST.GeomAbs_SurfaceOfExtrusion: "EXTRUSION",
        _ST.GeomAbs_OffsetSurface: "OFFSET", _ST.GeomAbs_OtherSurface: "OTHER",
    }.get(t, "OTHER")


def _v(g) -> tuple:
    return (g.X(), g.Y(), g.Z())


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-12 else (0.0, 0.0, 0.0)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _canon_axis(d):
    """Canonical direction (sign invariant) so that axes +Z and -Z hash alike."""
    d = _norm(d)
    for c in d:
        if abs(c) > 1e-9:
            return d if c > 0 else _scale(d, -1.0)
    return d


# ---------------------------------------------------------------- yt-attribut
def _surface_params(face) -> dict:
    ad = BRepAdaptor_Surface(face)
    t = ad.GetType()
    name = _stype_name(t)
    out = {"type": name}
    try:
        if t == _ST.GeomAbs_Plane:
            pl = ad.Plane()
            out["origin"] = _v(pl.Location())
            out["normal"] = _norm(_v(pl.Axis().Direction()))
        elif t == _ST.GeomAbs_Cylinder:
            cy = ad.Cylinder()
            out["axis_point"] = _v(cy.Location())
            out["axis_dir"] = _norm(_v(cy.Axis().Direction()))
            out["radius"] = cy.Radius()
        elif t == _ST.GeomAbs_Cone:
            co = ad.Cone()
            out["axis_point"] = _v(co.Location())
            out["axis_dir"] = _norm(_v(co.Axis().Direction()))
            out["ref_radius"] = co.RefRadius()
            out["half_angle_deg"] = math.degrees(co.SemiAngle())
            ap = co.Apex()
            out["apex"] = _v(ap)
        elif t == _ST.GeomAbs_Sphere:
            sp = ad.Sphere()
            out["sphere_center"] = _v(sp.Location())
            out["radius"] = sp.Radius()
        elif t == _ST.GeomAbs_Torus:
            to = ad.Torus()
            out["axis_point"] = _v(to.Location())
            out["axis_dir"] = _norm(_v(to.Axis().Direction()))
            out["major_radius"] = to.MajorRadius()
            out["minor_radius"] = to.MinorRadius()
    except Exception as exc:  # pragma: no cover
        out["param_error"] = repr(exc)
    return out


def _face_normal_at(face, p3: tuple):
    """OUTWARD normal of `face` at the point p3 (projected onto the surface)."""
    surf = BRep_Tool.Surface_s(face)
    proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(*p3), surf)
    if proj.NbPoints() < 1:
        return None
    u, v = proj.LowerDistanceParameters()
    p = gp_Pnt()
    d1u, d1v = gp_Vec(), gp_Vec()
    surf.D1(u, v, p, d1u, d1v)
    n = _cross(_v(d1u), _v(d1v))
    n = _norm(n)
    if n == (0.0, 0.0, 0.0):
        return None
    if face.Orientation() == TopAbs_REVERSED:
        n = _scale(n, -1.0)
    return n


def _face_uv_sample(face):
    """Point + OUTWARD normal in the middle of the face's OWN UV range. Needed so that a full
    cylinder centroid lies ON the axis (degenerate projection, measured: normal=None)."""
    umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
    surf = BRep_Tool.Surface_s(face)
    u = 0.5 * (umin + umax)
    v = 0.5 * (vmin + vmax)
    for uu, vv in ((u, v), (0.25 * umin + 0.75 * umax, v), (u, 0.25 * vmin + 0.75 * vmax)):
        try:
            pnt = gp_Pnt()
            d1u, d1v = gp_Vec(), gp_Vec()
            surf.D1(uu, vv, pnt, d1u, d1v)
            n = _norm(_cross(_v(d1u), _v(d1v)))
        except Exception:
            continue
        if n == (0.0, 0.0, 0.0):
            continue
        if face.Orientation() == TopAbs_REVERSED:
            n = _scale(n, -1.0)
        return _v(pnt), n
    return None, None


def _face_area(face) -> float:
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, g)
    return g.Mass()


def _face_center(face) -> tuple:
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, g)
    return _v(g.CentreOfMass())


def _surface_concavity(rec: dict) -> str | None:
    """INWARD = the face's outward normal points TOWARDS the axis/centre (the material lies OUTSIDE
    the face) -- bore / hole / concave blend. OUTWARD = away from the axis -- shaft / boss / convex.
    This IS the hole-vs-boss discriminant; the dihedral edge class is NOT (both rim edges of a
    through hole are 90-degree CONVEX, measured in _selftest())."""
    t = rec["type"]
    n = rec.get("sample_normal")
    c = rec.get("sample_point")
    if n is None or c is None:
        return None
    if t in ("CYLINDER", "CONE", "TORUS"):
        ap, ad = rec.get("axis_point"), rec.get("axis_dir")
        if ap is None:
            return None
        d = _sub(c, ap)
        radial = _norm(_sub(d, _scale(ad, _dot(d, ad))))
        if radial == (0.0, 0.0, 0.0):
            return None
        s = _dot(n, radial)
        if abs(s) < 1e-6:
            return None
        return "OUTWARD" if s > 0 else "INWARD"
    if t == "SPHERE":
        cen = rec.get("sphere_center")
        if cen is None:
            return None
        radial = _norm(_sub(c, cen))
        if radial == (0.0, 0.0, 0.0):
            return None
        s = _dot(n, radial)
        if abs(s) < 1e-6:
            return None
        return "OUTWARD" if s > 0 else "INWARD"
    return None


# ---------------------------------------------------------------- AAG
class AAG:
    """Attributed adjacency graph. faces: list of dicts. links: list of dicts."""

    def __init__(self, shape):
        self.shape = shape
        topods = shape.wrapped if hasattr(shape, "wrapped") else shape
        self.topods = topods
        self.faces: list[dict] = []
        self._face_shapes: list = []
        self.links: list[dict] = []
        self.adj: dict[int, list[int]] = {}
        self._idx_by_key: dict[int, int] = {}
        self._build()

    # -- konstruktion
    def _build(self):
        fmap = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(self.topods, TopAbs_FACE, fmap)
        self._fmap = fmap
        exp = TopExp_Explorer(self.topods, TopAbs_FACE)
        seen = set()
        while exp.More():
            f = TopoDS.Face_s(exp.Current())
            key = fmap.FindIndex(f)
            if key in seen:
                exp.Next()
                continue
            seen.add(key)
            i = len(self._face_shapes)
            self._face_shapes.append(f)
            self._idx_by_key[key] = i
            rec = _surface_params(f)
            rec["idx"] = i
            rec["area"] = _face_area(f)
            rec["center"] = _face_center(f)
            rec["reversed"] = bool(f.Orientation() == TopAbs_REVERSED)
            # outward normal at the face centroid (for type-independent direction questions)
            sp, sn = _face_uv_sample(f)
            rec["sample_point"] = sp
            rec["sample_normal"] = sn
            rec["concavity"] = _surface_concavity(rec)
            try:
                u0, u1, v0, v1 = BRepTools.UVBounds_s(f)
                rec["u_span"] = u1 - u0
                rec["v_span"] = v1 - v0
            except Exception:
                rec["u_span"] = rec["v_span"] = None
            self.faces.append(rec)
            self.adj[i] = []
            exp.Next()

        # kant -> ytor
        m = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(self.topods, TopAbs_EDGE, TopAbs_FACE, m)
        for k in range(1, m.Extent() + 1):
            edge = TopoDS.Edge_s(m.FindKey(k))
            fl = m.FindFromIndex(k)
            fs = _list_faces(fl)
            if len(fs) != 2:
                continue
            i1 = self._index_of(fs[0])
            i2 = self._index_of(fs[1])
            if i1 is None or i2 is None or i1 == i2:
                continue
            link = self._classify_edge(edge, fs[0], i1, fs[1], i2)
            if link is None:
                continue
            self.links.append(link)
            self.adj[i1].append(i2)
            self.adj[i2].append(i1)

    def _index_of(self, f):
        return self._idx_by_key.get(self._fmap.FindIndex(f))

    def _classify_edge(self, edge, f1, i1, f2, i2) -> dict | None:
        ac = BRepAdaptor_Curve(edge)
        u0, u1 = ac.FirstParameter(), ac.LastParameter()
        if not (math.isfinite(u0) and math.isfinite(u1)):
            return None
        um = 0.5 * (u0 + u1)
        p = gp_Pnt()
        d1 = gp_Vec()
        try:
            ac.D1(um, p, d1)
        except Exception:
            return None
        P = _v(p)
        t = _norm(_v(d1))
        if t == (0.0, 0.0, 0.0):
            return None
        # kantens orientering SOM DEN UPPTRADER I f1
        ori = _edge_orientation_in_face(edge, f1)
        if ori == TopAbs_REVERSED:
            t = _scale(t, -1.0)
        n1 = _face_normal_at(f1, P)
        n2 = _face_normal_at(f2, P)
        if n1 is None or n2 is None:
            return None
        w1 = _cross(n1, t)          # pekar in i f1:s inre (harledning i moduldocstring)
        s = _dot(w1, n2)
        # dihedral angle: the angle between the normals, signed with s
        c = max(-1.0, min(1.0, _dot(n1, n2)))
        ang = math.degrees(math.acos(c))
        if abs(ang) < TOL_TANGENT_DEG or abs(180.0 - ang) < TOL_TANGENT_DEG:
            kind = "TANGENT"
        elif s < 0:
            kind = "CONVEX"
        else:
            kind = "CONCAVE"
        ct = ac.GetType()
        etype = {0: "LINE", 1: "CIRCLE", 2: "ELLIPSE", 3: "HYPERBOLA",
                 4: "PARABOLA", 5: "BEZIER", 6: "BSPLINE"}.get(int(ct), "OTHER")
        d = {"f1": i1, "f2": i2, "kind": kind, "angle_deg": ang,
             "edge_type": etype, "mid": P, "tangent": t,
             "length": _edge_length(edge)}
        if etype == "CIRCLE":
            circ = ac.Circle()
            d["circle_radius"] = circ.Radius()
            d["circle_center"] = _v(circ.Location())
            d["circle_axis"] = _norm(_v(circ.Axis().Direction()))
        return d

    # -- fragor
    def links_of(self, i) -> list[dict]:
        return [l for l in self.links if l["f1"] == i or l["f2"] == i]

    def other(self, link, i) -> int:
        return link["f2"] if link["f1"] == i else link["f1"]

    def face_shape(self, i):
        return self._face_shapes[i]

    def outer_wire_edges(self, i):
        return _wire_edges(BRepTools.OuterWire_s(self._face_shapes[i]))

    def inner_wires(self, i) -> list:
        f = self._face_shapes[i]
        ow = BRepTools.OuterWire_s(f)
        out = []
        from OCP.TopAbs import TopAbs_WIRE
        exp = TopExp_Explorer(f, TopAbs_WIRE)
        while exp.More():
            from OCP.TopoDS import TopoDS as _T
            w = _T.Wire_s(exp.Current())
            if not w.IsSame(ow):
                out.append(w)
            exp.Next()
        return out

    def summary(self) -> dict:
        from collections import Counter
        return {
            "n_faces": len(self.faces),
            "n_links": len(self.links),
            "face_types": dict(Counter(f["type"] for f in self.faces)),
            "edge_kinds": dict(Counter(l["kind"] for l in self.links)),
        }


def _round_loc(f):
    tr = f.Location().Transformation().TranslationPart()
    return (round(tr.X(), 9), round(tr.Y(), 9), round(tr.Z(), 9))


def _list_faces(fl):
    return [TopoDS.Face_s(sh) for sh in fl]


def _edge_orientation_in_face(edge, face):
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge_s(exp.Current())
        if e.IsSame(edge):
            return e.Orientation()
        exp.Next()
    return edge.Orientation()


def _edge_length(edge) -> float:
    g = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, g)
    return g.Mass()


def _wire_edges(wire) -> list:
    out = []
    exp = TopExp_Explorer(wire, TopAbs_EDGE)
    while exp.More():
        out.append(TopoDS.Edge_s(exp.Current()))
        exp.Next()
    return out


def build_aag(shape) -> AAG:
    return AAG(shape)


# ---------------------------------------------------------------- fallbevis
def _selftest() -> dict:
    """Calibration and planted-fault fixtures for the dihedral convention (6 independent shapes).

    Pre-registered, and one falsified expectation reported as such: both rim edges of a THROUGH
    hole are 90-degree CONVEX -- the material angle there is 90, not 270. Concave edges appear only
    when a BOTTOM exists (blind hole, pocket) or when material RISES out of a face (boss foot).
    Hole vs boss is therefore decided by the FACE CONCAVITY (normal towards / away from the axis),
    not by the edge class."""
    res = {}

    a = build_aag(bd.Box(20, 20, 20))
    res["box"] = a.summary()

    plate = bd.Box(40, 40, 10) - bd.Cylinder(radius=5, height=40)
    a2 = build_aag(plate)
    res["plate_through_hole"] = a2.summary()
    cyl2 = [f for f in a2.faces if f["type"] == "CYLINDER"]

    blind = bd.Box(40, 40, 20) - bd.Pos(0, 0, 5) * bd.Cylinder(radius=5, height=20)
    a3 = build_aag(blind)
    res["plate_blind_hole"] = a3.summary()

    pocket = bd.Box(40, 40, 20) - bd.Pos(0, 0, 5) * bd.Box(20, 20, 20)
    a4 = build_aag(pocket)
    res["box_pocket"] = a4.summary()

    filleted = bd.fillet(bd.Box(20, 20, 20).edges().group_by(bd.Axis.Z)[-1], 2.0)
    a5 = build_aag(filleted)
    res["box_top_fillet"] = a5.summary()

    boss = bd.Box(40, 40, 10) + bd.Pos(0, 0, 10) * bd.Cylinder(radius=5, height=10)
    a6 = build_aag(boss)
    res["plate_boss"] = a6.summary()
    cyl6 = [f for f in a6.faces if f["type"] == "CYLINDER"]

    # pocket: the bottom face is the only planar face whose edges are ALL concave
    pocket_bottoms = []
    for f in a4.faces:
        if f["type"] != "PLANE":
            continue
        ls = a4.links_of(f["idx"])
        if ls and all(l["kind"] == "CONCAVE" for l in ls):
            pocket_bottoms.append(f["idx"])
    blind_bottoms = []
    for f in a3.faces:
        if f["type"] != "PLANE":
            continue
        ls = a3.links_of(f["idx"])
        if ls and all(l["kind"] == "CONCAVE" for l in ls):
            blind_bottoms.append(f["idx"])

    checks = [
        ("box_all_convex", res["box"]["edge_kinds"] == {"CONVEX": 12}),
        ("through_hole_no_concave", res["plate_through_hole"]["edge_kinds"].get("CONCAVE", 0) == 0),
        ("through_hole_cyl_INWARD", len(cyl2) == 1 and cyl2[0]["concavity"] == "INWARD"),
        ("blind_hole_one_concave", res["plate_blind_hole"]["edge_kinds"].get("CONCAVE", 0) == 1),
        ("blind_bottom_found", blind_bottoms == [i for i in blind_bottoms][:1] and len(blind_bottoms) == 1),
        ("pocket_eight_concave", res["box_pocket"]["edge_kinds"].get("CONCAVE", 0) == 8),
        ("pocket_bottom_found", len(pocket_bottoms) == 1),
        ("fillet_has_tangent", res["box_top_fillet"]["edge_kinds"].get("TANGENT", 0) == 8),
        ("boss_one_concave", res["plate_boss"]["edge_kinds"].get("CONCAVE", 0) == 1),
        ("boss_cyl_OUTWARD", len(cyl6) == 1 and cyl6[0]["concavity"] == "OUTWARD"),
    ]
    res["checks"] = {k: bool(v) for k, v in checks}
    res["ALL_PASS"] = all(v for _, v in checks)
    return res


if __name__ == "__main__":
    import json
    if "--selftest" in sys.argv:
        r = _selftest()
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ALL_PASS"] else 1)
    for p in sys.argv[1:]:
        s = bd.import_step(p)
        print(p, json.dumps(build_aag(s).summary()))
