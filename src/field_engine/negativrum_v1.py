#!/usr/bin/env python3
"""Negative space as a first-class object: declare the void before the material.

The usual chain builds MATERIAL first and repairs collisions afterwards: a clearance sweep is
subtracted from finished machines, and joints get un-published because material was already in the
way. The inversion here is void-first. Motion sweeps, service access, media corridors and cooling-air
paths are DECLARED as named negative spaces (negativrum) before any material exists; the material is
the complement. A joint cannot collide with what was carved out by its own sweep at birth.

Two representations, one source, and no SDF->B-rep stitching: every room is declared as a union of
ANALYTIC primitives (lada = box / cylinder / kon = truncated cone). From one and the same parameter
list both representations fall out directly --

  the FIELD   an IKARUS expression tree (ikarus_v1/expr.py: box/cylinder/cone/union), evaluable
              analytically: cheap pre-filter, differentiable.
  the SOLID   an exact build123d body (Box/Cylinder/Cone + fuse): the verdict, exact B-rep.

Published SDF->B-rep pipelines report non-watertight artefacts when a field is re-stitched into a
B-rep on complex intersections. Nothing is stitched here: the field is never marching-cubed and the
complement of a field union is never taken to B-rep, so that failure class does not arise instead of
being gated afterwards. What is still gated is (a) G0, that every produced solid is valid and closed,
and (b) G1, field parity: the sign of the field and the containment test of the solid must agree in N
random points, fail-closed. Two independent evaluators of the same declaration.

Four gates, all fail-closed:
  G0 WATERTIGHTNESS  every field->solid transition yields a valid body: closed shells and an even
                     Euler characteristic chi = V - E + F - H (H = inner wires; a face with a hole is
                     not a disk, and the first version of the gate failed a correct body by
                     forgetting that)
  G1 FIELD PARITY    sign(sdf) == solid.contains for N points (outside a narrow band |sdf|<tol)
  G2 SWEEP COVERAGE  the DECLARED room must CONTAIN the joint's real swept volume:
                     volume(sweep - room) == 0, exact B-rep. A room smaller than the motion is a
                     fail-open calm: the joint would "fit" and still jam.
  G3 KEEP-OUT        no part outside the room's OWNERS may have an intersection volume > tol against
                     the room, exact B-rep per solid pair. The field is only the pre-filter that
                     selects the pairs.

Ownership (the directed exception): the owners of a motion sweep are exactly the parts whose motion
generated it. They may -- must -- lie inside the room; it is their path. All other parts are excluded.
Without that direction every sweep would fail itself.

The solver: losa_placering() searches for the SMALLEST displacement of a part group along DECLARED
freedoms that takes it out of all negative spaces AND keeps it inside its own boundary conditions. If
none is found the outcome is a MEASURED CONGESTION FINDING -- which rooms compete, over what volume --
not a failure.

I/O: primitive declarations and part solids in; room objects, gate reports and a negativrum_v1 JSON
out. --selftest runs seven gate cases (field parity on all three primitive types, the watertightness
gate on a closed body and an open shell, sweep coverage on a too-small and a covering room, keep-out
with and without ownership, the solver and a congestion finding) and writes its report next to this
module.

Run: python negativrum_v1.py --selftest
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np

REPO = os.environ.get("FIELD_ENGINE_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

KLASSER = ("rorelsesvep", "service", "media", "kyl")
KLASS_DOC = {
    "rorelsesvep": "the volume a joint and its drive chain sweep through its WHOLE declared range",
    "service": "access cone/corridor to a consumable part that must be replaceable with tools",
    "media": "cable/hose corridor between a consumer and the cabinet",
    "kyl": "air path: intake -> heat source -> outlet, it must not be walled in",
}
PRIMTYPER = ("lada", "cylinder", "kon")

VOL_TOL_MM3 = 1.0        # below this a B-rep intersection is tessellation/tangency noise
PARITET_BAND_MM = 0.75   # points closer than this to the boundary are not judged (discretisation band)
BBOX_TOL_MM = 0.05       # manifest bboxes are rounded to 2 decimals -- containment margin


# ==================================================================================================
# PRIMITIVES -- one declaration, two representations
# ==================================================================================================
def lada(center, half, namn=None):
    """Axis-aligned box. center=(x,y,z) mm, half=(hx,hy,hz) mm (HALF extents)."""
    if min(half) <= 0:
        raise ValueError(f"lada {namn}: half extents must be > 0, got {half}")
    return {"typ": "lada", "center": [float(c) for c in center],
            "half": [float(h) for h in half], "namn": namn}


def cylinder(center, axel, radie, hojd, namn=None):
    """Cylinder along coordinate axis `axel` ('x'|'y'|'z'). hojd = FULL extent."""
    if axel not in ("x", "y", "z"):
        raise ValueError(f"cylinder {namn}: axel must be x/y/z, got {axel!r}")
    if radie <= 0 or hojd <= 0:
        raise ValueError(f"cylinder {namn}: radie/hojd must be > 0")
    return {"typ": "cylinder", "center": [float(c) for c in center], "axel": axel,
            "radie": float(radie), "hojd": float(hojd), "namn": namn}


def kon(center, axel, r0, r1, hojd, namn=None):
    """Truncated cone along `axel`; r0 at -hojd/2, r1 at +hojd/2. The access-cone shape."""
    if axel not in ("x", "y", "z"):
        raise ValueError(f"kon {namn}: axel must be x/y/z, got {axel!r}")
    if min(r0, r1) <= 0 or hojd <= 0:
        raise ValueError(f"kon {namn}: radii/hojd must be > 0")
    return {"typ": "kon", "center": [float(c) for c in center], "axel": axel,
            "r0": float(r0), "r1": float(r1), "hojd": float(hojd), "namn": namn}


def prim_flytta(p, d):
    """The same primitive, translated by d=(dx,dy,dz). The solver moves CANDIDATES, never rooms."""
    q = dict(p)
    q["center"] = [p["center"][i] + float(d[i]) for i in range(3)]
    return q


def prim_bbox(p):
    c = p["center"]
    if p["typ"] == "lada":
        h = p["half"]
    else:
        ai = "xyz".index(p["axel"])
        r = max(p["radie"], 0.0) if p["typ"] == "cylinder" else max(p["r0"], p["r1"])
        h = [r, r, r]
        h[ai] = p["hojd"] / 2.0
    return [c[0] - h[0], c[1] - h[1], c[2] - h[2], c[0] + h[0], c[1] + h[1], c[2] + h[2]]


def prim_volym(p):
    if p["typ"] == "lada":
        return 8.0 * p["half"][0] * p["half"][1] * p["half"][2]
    if p["typ"] == "cylinder":
        return math.pi * p["radie"] ** 2 * p["hojd"]
    r0, r1 = p["r0"], p["r1"]
    return math.pi * p["hojd"] / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1)


# -------------------------------------------------------------------------------- FIELD (numpy)
def _sdf_prim(p, P):
    """Exact analytic SDF (mm, NEGATIVE INSIDE -- the same sign convention as ikarus/eval_warp).

    The canonical evaluator. The cone field is the conservative min-of-half-spaces form; it never
    understates the outside distance and can therefore never be fail-open as a pre-filter."""
    Q = P - np.asarray(p["center"], dtype=np.float64)
    if p["typ"] == "lada":
        d = np.abs(Q) - np.asarray(p["half"], dtype=np.float64)
        out = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        return out + np.minimum(np.max(d, axis=1), 0.0)
    ai = "xyz".index(p["axel"])
    rad = np.array([i for i in range(3) if i != ai])
    r_xy = np.linalg.norm(Q[:, rad], axis=1)
    z = Q[:, ai]
    hh = p["hojd"] / 2.0
    if p["typ"] == "cylinder":
        dr, dz = r_xy - p["radie"], np.abs(z) - hh
        return (np.minimum(np.maximum(dr, dz), 0.0)
                + np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2))
    # kon: the radius at height z is linearly interpolated; the distance to the lateral surface is
    # taken perpendicular to the slant line (exact inside the slant normal band, conservative outside)
    r0, r1 = p["r0"], p["r1"]
    t = np.clip((z + hh) / p["hojd"], 0.0, 1.0)
    r_at = r0 + (r1 - r0) * t
    slope = (r1 - r0) / p["hojd"]
    dr = (r_xy - r_at) / math.sqrt(1.0 + slope * slope)
    dz = np.abs(z) - hh
    return (np.minimum(np.maximum(dr, dz), 0.0)
            + np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2))


def sdf(prims, P):
    """Field of the UNION: min over the primitives. P = (N,3) mm. Returns (N,) mm, negative = inside."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if not prims:
        return np.full(len(P), 1e9)
    out = _sdf_prim(prims[0], P)
    for p in prims[1:]:
        out = np.minimum(out, _sdf_prim(p, P))
    return out


def ikarus_trad(prims):
    """The same union as an IKARUS expression tree (ikarus_v1/expr.py schema).

    Written into the report so the field can be evaluated by the GPU engine (eval_warp) WITHOUT
    running this file -- two independent evaluators of ONE declaration."""
    def nod(p):
        c = p["center"]
        if p["typ"] == "lada":
            return {"op": "box", "half_extents": list(p["half"]), "center": list(c)}
        if p["typ"] == "cylinder":
            return {"op": "cylinder", "radius": p["radie"], "height": p["hojd"],
                    "axis": p["axel"], "center": list(c)}
        return {"op": "cone", "radius1": p["r0"], "radius2": p["r1"], "height": p["hojd"],
                "axis": p["axel"], "center": list(c)}
    if not prims:
        raise ValueError("ikarus_trad: empty room -- a negative space without primitives is not a declaration")
    t = nod(prims[0])
    for p in prims[1:]:
        t = {"op": "union", "a": t, "b": nod(p), "k": 0.0}
    return t


# -------------------------------------------------------------------------------- SOLID (B-rep)
def _bd():
    import build123d as bd
    return bd


def solid_av_prim(p):
    """Exact B-rep twin of the primitive. No marching cubes, no SDF->B-rep stitching."""
    bd = _bd()
    c = p["center"]
    if p["typ"] == "lada":
        h = p["half"]
        s = bd.Box(2 * h[0], 2 * h[1], 2 * h[2])
    elif p["typ"] == "cylinder":
        s = bd.Cylinder(p["radie"], p["hojd"])
    else:
        s = bd.Cone(p["r0"], p["r1"], p["hojd"])
    if p["typ"] != "lada":
        if p["axel"] == "x":
            s = s.rotate(bd.Axis.Y, 90.0)
        elif p["axel"] == "y":
            s = s.rotate(bd.Axis.X, -90.0)
    return bd.Pos(c[0], c[1], c[2]) * s


def solid_av(prims, namn="?"):
    """The union as ONE exact body. Fails the validity gate loudly (never a silent half body)."""
    if not prims:
        raise ValueError(f"solid_av {namn}: tomt rum")
    s = solid_av_prim(prims[0])
    for p in prims[1:]:
        s = s + solid_av_prim(p)
    v = float(s.volume)
    if not (v > 0.0) or not math.isfinite(v):
        raise ValueError(f"solid_av {namn}: ogiltig volym {v}")
    iv = getattr(s, "is_valid", None)
    ok = bool(iv() if callable(iv) else iv)
    if not ok:
        raise ValueError(f"solid_av {namn}: the B-rep body is NOT valid")
    return s


# ==================================================================================================
# ROOMS
# ==================================================================================================
class Rum:
    """ONE named negative space.

    namn    unique name
    klass   one of KLASSER
    kalla   dict NAMING the origin: {"typ":"led","joint":...} / {"typ":"krav","krav_id":...}
            -- a room without an origin is a guess with coordinates
    prims   [primitive]
    agare   part names ALLOWED inside the room (the moving parts that generate the sweep). All others are excluded.
    motiv   why the room has the shape it has (measurement/requirement), in plain text
    """

    def __init__(self, namn, klass, kalla, prims, agare=(), motiv=""):
        if klass not in KLASSER:
            raise ValueError(f"{namn}: okand klass {klass!r} (valj bland {KLASSER})")
        if not isinstance(kalla, dict) or not kalla.get("typ"):
            raise ValueError(f"{namn}: kalla missing -- which joint or requirement generates the room?")
        if not prims:
            raise ValueError(f"{namn}: inga primitiver")
        if not str(motiv).strip():
            raise ValueError(f"{namn}: motiv missing -- an unmotivated room is just a drawn rectangle")
        for p in prims:
            if p["typ"] not in PRIMTYPER:
                raise ValueError(f"{namn}: okand primitivtyp {p['typ']!r}")
        self.namn, self.klass, self.kalla = namn, klass, kalla
        self.prims, self.agare, self.motiv = list(prims), list(agare), motiv
        self._solid = None

    def bbox(self):
        bs = [prim_bbox(p) for p in self.prims]
        return [min(b[i] for b in bs) for i in range(3)] + [max(b[i + 3] for b in bs) for i in range(3)]

    def sdf(self, P):
        return sdf(self.prims, P)

    def solid(self):
        if self._solid is None:
            self._solid = solid_av(self.prims, self.namn)
        return self._solid

    def volym_mm3(self):
        return float(self.solid().volume)

    def som_dict(self):
        return {"namn": self.namn, "klass": self.klass, "klass_doc": KLASS_DOC[self.klass],
                "kalla": self.kalla, "agare": self.agare, "motiv": self.motiv,
                "primitiver": self.prims, "bbox_mm": [round(v, 3) for v in self.bbox()],
                "volym_mm3": round(self.volym_mm3(), 3),
                "falt_ikarus": ikarus_trad(self.prims)}


# ==================================================================================================
# MOTION SWEEP FROM A JOINT DECLARATION -- the room is born from the joint table, not from a drawn rectangle
# ==================================================================================================
def _rot_bbox(b, origin, axis, deg):
    """AABB of bbox `b` rotated `deg` degrees about the coordinate axis `axis` through `origin`.

    Conservative: the AABB of the eight rotated corners always encloses the rotated body."""
    ai = "xyz".index(axis)
    u, v = [i for i in range(3) if i != ai]
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    lo = [1e18] * 3
    hi = [-1e18] * 3
    for cu in (b[u], b[u + 3]):
        for cv in (b[v], b[v + 3]):
            du, dv = cu - origin[u], cv - origin[v]
            ru = origin[u] + ca * du - sa * dv
            rv = origin[v] + sa * du + ca * dv
            for cw in (b[ai], b[ai + 3]):
                pt = [0.0, 0.0, 0.0]
                pt[u], pt[v], pt[ai] = ru, rv, cw
                for k in range(3):
                    lo[k] = min(lo[k], pt[k])
                    hi[k] = max(hi[k], pt[k])
    return lo + hi


def svepprims_ur_led(bboxar, typ, limit, origin, axel, spalt_mm, n_steg=None):
    """DECLARED sweep primitives from the joint's own numbers + the parts' AABBs.

    The result is a union of axis-aligned boxes -- a CONSERVATIVE over-solid of the true swept
    volume (each box AABB encloses the part in that pose). Being conservative is the right sign
    for a negative space: a room that is TOO LARGE reserves too much, a room that is TOO SMALL is
    fail-open. G2 (sweep coverage) measures that it really encloses the exact sweep.

    bboxar: [(part_name, bbox6)] in the build pose (q=0). The joint axis must be a coordinate axis.
    """
    ax = [abs(float(a)) for a in axel]
    if sum(1 for a in ax if a > 1e-9) != 1:
        raise ValueError(f"svepprims_ur_led: axis {axel} is not a coordinate axis -- extend the kit "
                         f"instead of approximating")
    ai = int(np.argmax(ax))
    achar = "xyz"[ai]
    lo, hi = float(limit[0]), float(limit[1])
    span = hi - lo
    prims = []
    if typ == "prismatic":
        # EXACT, NOT SAMPLED: an AABB translated along a coordinate axis sweeps exactly the AABB
        # extended by the stroke along that axis. One step is enough, and it is the tightest
        # possible box -- no discretisation remainder to account for.
        for namn, b in bboxar:
            bb = list(b)
            bb[ai] += min(lo, hi)
            bb[ai + 3] += max(lo, hi)
            c = [(bb[i] + bb[i + 3]) / 2.0 for i in range(3)]
            h = [max((bb[i + 3] - bb[i]) / 2.0 + spalt_mm, 1e-3) for i in range(3)]
            prims.append(lada(c, h, namn=f"{namn}@{round(lo, 2)}..{round(hi, 2)}"))
        return prims
    # THE STEP IS DERIVED, NOT CHOSEN -- and it is FINER than the sweep's own step.
    # The swept-volume helper picks dq so that the sagitta at the outermost radius R stays below
    # clearance/4. Here the room is tested AGAINST that sweep, so the step must be at least as
    # fine; the safety factor 4 is there because one AABB per pose does not automatically cover
    # the REAL body between two poses. A first version took 10 degrees flat and G2 then measured
    # 186 895 mm3 of a service hatch's exact sweep OUTSIDE the room -- a fail-open room that
    # looked green. The gate caught its own generator.
    for namn, b in bboxar:
        R = 0.0
        for cu in (b[0], b[3]):
            for cv in (b[1], b[4]):
                for cw in (b[2], b[5]):
                    d = [cu - origin[0], cv - origin[1], cw - origin[2]]
                    d[ai] = 0.0
                    R = max(R, math.hypot(d[(ai + 1) % 3], d[(ai + 2) % 3]))
        if R > 1e-6 and spalt_mm > 0:
            dq = math.degrees(2.0 * math.acos(max(min(1.0 - spalt_mm / (4.0 * R), 1.0), -1.0)))
        else:
            dq = 5.0
        dq = min(max(dq / 4.0, abs(span) / 1024.0, 0.1), 5.0)
        n = n_steg or max(int(math.ceil(abs(span) / dq)) + 1, 5)
        qs = [lo + span * k / (n - 1) for k in range(n)]
        for q in qs:
            bb = _rot_bbox(b, origin, achar, q)
            c = [(bb[i] + bb[i + 3]) / 2.0 for i in range(3)]
            h = [max((bb[i + 3] - bb[i]) / 2.0 + spalt_mm, 1e-3) for i in range(3)]
            prims.append(lada(c, h, namn=f"{namn}@q={round(q, 3)}"))
    return prims


# ==================================================================================================
# G0 WATERTIGHTNESS -- every produced B-rep body, at every field->solid transition
# ==================================================================================================
def vattentathet(rum):
    """Is the room's B-rep twin a CLOSED, valid body?

    Why the gate exists even though nothing is stitched: published SDF->B-rep pipelines report
    non-watertight artefacts when a field is marching-cubed and re-stitched into a B-rep. That
    transition is never made here -- neither for the rooms nor for their complement -- so the
    failure class cannot arise. But "cannot arise" is a CLAIM, and an unmeasured claim is a
    fail-open. So it is measured:

      * every solid is valid (`is_valid`)
      * every solid has finite volume > 0
      * every SHELL in the body is CLOSED (OCC's own `Closed` flag) -- the exact B-rep
        counterpart of "watertight"
      * the body's EULER CHARACTERISTIC is even (a closed orientable surface has chi = 2-2g) --
        a second, independent witness

    The gate first asked the wrong question, and that was measured: a first version computed
    chi = V - E + F flat and failed a correct body with chi = 3 (64 - 96 + 35). V-E+F=2 only holds
    when EVERY face is a disk. In a union of boxes faces get INNER WIRES (a small box in the middle
    of a larger face punches a hole), and a face with h holes contributes 1 - h, not 1. The right
    quantity is chi = V - E + F - H with H the number of inner wires. A gate that fails a correct
    body teaches the reader to ignore it -- the fix belongs in the QUESTION, not in the threshold.

    The gate runs on EVERY room, i.e. at every transition from field declaration to exact body."""
    S = rum.solid()
    v = float(S.volume)
    iv = getattr(S, "is_valid", None)
    giltig = bool(iv() if callable(iv) else iv)
    shells, slutna = 0, 0
    try:
        for sh in S.shells():
            shells += 1
            w = getattr(sh.wrapped, "Closed", None)
            slutna += 1 if bool(w() if callable(w) else w) else 0
    except Exception:
        pass
    try:
        ytor = S.faces()
        nv, ne, nf = len(S.vertices()), len(S.edges()), len(ytor)
        nh = 0                      # INNER wires: a face with h holes contributes 1 - h, not 1
        for f in ytor:
            try:
                nh += max(len(f.wires()) - 1, 0)
            except Exception:
                pass
        chi = nv - ne + nf - nh
    except Exception:
        nv = ne = nf = nh = chi = None
    return {"rum": rum.namn, "volym_mm3": round(v, 3), "is_valid": giltig,
            "n_shells": shells, "n_slutna_shells": slutna,
            "V": nv, "E": ne, "F": nf, "inre_wires": nh, "euler_chi": chi,
            "euler_jamn": (chi % 2 == 0) if chi is not None else None,
            "GRON": bool(giltig and v > 0.0 and math.isfinite(v)
                         and shells > 0 and slutna == shells
                         and chi is not None and chi % 2 == 0),
            "regel": ("every field->solid transition yields a VALID body with volume > 0, ONLY "
                      "closed shells and an even Euler characteristic; fail-closed")}


# ==================================================================================================
# G1 FIELD PARITY -- two evaluators of ONE declaration must agree
# ==================================================================================================
def falt_paritet(rum, n=20000, seed=7, marginal_mm=60.0):
    """Sample the room bbox (+margin); compare sign(SDF) against the exact B-rep containment test.

    Points within PARITET_BAND_MM of the boundary are excluded (there both are right within their
    own tolerance) and COUNTED -- a band that eats half the sample is not a test."""
    bd = _bd()
    b = rum.bbox()
    lo = np.array(b[:3]) - marginal_mm
    hi = np.array(b[3:]) + marginal_mm
    rng = np.random.default_rng(seed)
    P = lo + (hi - lo) * rng.random((int(n), 3))
    d = rum.sdf(P)
    band = np.abs(d) < PARITET_BAND_MM
    S = rum.solid()
    fel = 0
    provade = 0
    for i in range(len(P)):
        if band[i]:
            continue
        provade += 1
        inne_brep = bool(S.is_inside(bd.Vector(*P[i]))) if hasattr(S, "is_inside") \
            else bool(len(bd.Vertex(*P[i]).intersect(S).vertices()) > 0)
        if inne_brep != bool(d[i] < 0.0):
            fel += 1
    return {"n": int(n), "n_i_band": int(band.sum()), "n_provade": provade, "n_oense": fel,
            "band_mm": PARITET_BAND_MM,
            "GRON": bool(fel == 0 and provade >= 0.5 * n),
            "regel": "sign(field SDF) == exact B-rep containment test outside the band; fail-closed"}


# ==================================================================================================
# G2 SWEEP COVERAGE -- the declared room must CONTAIN the real motion
# ==================================================================================================
def sveptackning(rum, svep_kroppar):
    """volume(sweep - room) == 0, exact B-rep, per swept body. Fail-closed.

    svep_kroppar: [(name, solid)] -- e.g. the poses of a swept-volume generator. A room SMALLER
    than the motion is exactly the fail-open that lets a joint look free and still jam."""
    S = rum.solid()
    rest = []
    olosta = []
    tot = 0.0
    n_bbox_bevisade = 0
    lador = [prim_bbox(p) for p in rum.prims if p["typ"] == "lada"]
    for namn, kropp in svep_kroppar:
        # PROOF BEFORE BOOLEAN. If the swept body's own bbox fits inside ONE room box the rest is
        # proven empty: body <= bbox(body) <= box <= room. No OCC boolean is needed, and this is
        # not a shortcut but a TIGHTER verdict -- a first version let OCC's cut decide instead
        # and reported 153 599 mm3 "outside the room" for a sliding door handle in two of 147
        # poses while the other 145 gave exactly 0. A body that fits in the box cannot stick out
        # of it; the number was a boolean artefact, not geometry.
        try:
            bb = kropp.bounding_box()
            kb = [bb.min.X, bb.min.Y, bb.min.Z, bb.max.X, bb.max.Y, bb.max.Z]
        except Exception:
            kb = None
        if kb is not None and any(all(kb[i] >= L[i] - BBOX_TOL_MM and kb[i + 3] <= L[i + 3] + BBOX_TOL_MM
                                      for i in range(3)) for L in lador):
            n_bbox_bevisade += 1
            continue
        try:
            r = kropp - S
            v = float(r.volume) if r is not None else 0.0
        except Exception as e:
            # OCC's fuse/cut fails on some compound bodies ("Null TopoDS_Shape"). Fall back to
            # subtracting the room primitives ONE BY ONE -- geometrically identical (A minus the
            # union is the same as A minus the terms in turn), only slower. If that fails too the
            # pair is UNRESOLVED and FAILS the gate; it is never silenced to 0.
            try:
                r = kropp
                for p in rum.prims:
                    r = r - solid_av_prim(p)
                    if r is None or float(r.volume) <= VOL_TOL_MM3:
                        break
                v = float(r.volume) if r is not None else 0.0
            except Exception:
                olosta.append({"svepkropp": namn, "fel": str(e)[:160],
                               "aterfall": "primitivvis subtraktion foll ocksa"})
                continue
        if v > VOL_TOL_MM3:
            rest.append({"svepkropp": namn, "utanfor_rummet_mm3": round(v, 3)})
        tot += max(v, 0.0)
    return {"GRON": bool(not rest and not olosta), "n_svepkroppar": len(svep_kroppar),
            "n_bbox_bevisade": n_bbox_bevisade, "n_booleanskt_provade": len(svep_kroppar) - n_bbox_bevisade,
            "utanfor_rummet_mm3": round(tot, 3), "vartill": rest[:60], "n_vartill": len(rest),
            "olosta": olosta[:20], "n_olosta": len(olosta), "tol_mm3": VOL_TOL_MM3,
            "regel": ("volume(sweep - room) == 0: the room must COVER the motion it claims to be. "
                      "An UNRESOLVED pair is never silenced to 0 -- it fails the gate.")}


# ==================================================================================================
# G3 KEEP-OUT -- the field selects the pairs, exact B-rep judges
# ==================================================================================================
def keepout_dom(rum_lista, delar, forfilter_marginal_mm=2.0):
    """delar: {part_name: (solid, point cloud (M,3))}. The point cloud is the FIELD pre-filter.

    The field judges nothing. It selects which (part, room) pairs are computed exactly -- the
    pre-filter is deliberately GENEROUS (margin) so it can never discard a pair that the exact
    B-rep would have failed. The verdict is always an exact intersection volume."""
    utfall = []
    n_par_falt = 0
    n_par_exakt = 0
    for R in rum_lista:
        S = R.solid()
        agare = set(R.agare)
        for namn, (solid, moln) in sorted(delar.items()):
            if namn in agare:
                continue
            if moln is not None and len(moln):
                if float(np.min(R.sdf(moln))) > forfilter_marginal_mm:
                    continue                      # field: the whole part is proven outside
            n_par_falt += 1
            try:
                r = solid & S
                v = float(r.volume) if r is not None else 0.0
            except Exception as e:
                utfall.append({"rum": R.namn, "del": namn, "OLOST": str(e)[:160]})
                continue
            n_par_exakt += 1
            if v > VOL_TOL_MM3:
                utfall.append({"rum": R.namn, "del": namn, "snitt_mm3": round(v, 3)})
    brott = [u for u in utfall if u.get("snitt_mm3")]
    olosta = [u for u in utfall if u.get("OLOST")]
    return {"GRON": bool(not brott and not olosta), "n_brott": len(brott),
            "n_olosta": len(olosta), "brott": brott, "olosta": olosta,
            "n_par_efter_faltforfilter": n_par_falt, "n_par_exakt_raknade": n_par_exakt,
            "tol_mm3": VOL_TOL_MM3,
            "regel": "material outside a room's OWNERS may never intersect the room (exact B-rep)"}


# ==================================================================================================
# THE SOLVER -- the design redistributes itself, and where it cannot that is a physical no
# ==================================================================================================
def kandidatoffset(friheter, steg_mm, tak_mm):
    """DETERMINISTIC candidate list, sorted on |offset| (smallest displacement first).

    friheter: [(axis index 0/1/2, sign +1/-1/0)] -- 0 = both directions allowed.
    A freedom that is not declared is NEVER used: the solver may not invent a motion the
    construction does not allow."""
    ax = {}
    for i, sg in friheter:
        ax.setdefault(int(i), set()).add(int(sg))
    steg_per_axel = {}
    for i, sgs in ax.items():
        vals = [0.0]
        k = 1
        while k * steg_mm <= tak_mm + 1e-9:
            if 0 in sgs or 1 in sgs:
                vals.append(+k * steg_mm)
            if 0 in sgs or -1 in sgs:
                vals.append(-k * steg_mm)
            k += 1
        steg_per_axel[i] = vals
    axlar = sorted(steg_per_axel)
    ut = []

    def rek(j, cur):
        if j == len(axlar):
            ut.append(tuple(cur))
            return
        for v in steg_per_axel[axlar[j]]:
            c = list(cur)
            c[axlar[j]] = v
            rek(j + 1, c)
    rek(0, [0.0, 0.0, 0.0])
    ut = sorted(set(ut), key=lambda d: (round(math.sqrt(sum(x * x for x in d)), 6), d))
    return ut


def losa_placering(grupp, rum_lista, friheter, steg_mm=5.0, tak_mm=300.0,
                   randvillkor=None, marginal_mm=0.0, log=print):
    """Smallest common displacement of `grupp` that takes it OUT of all negative spaces.

    grupp:      {part_name: primitive} -- the part group described in the SAME primitive family,
                so both the field test and the exact test are exact on the same declaration.
    randvillkor: callable(offset) -> None|"reason" -- the construction's OWN limits (enclosure
                inner dimensions, belt reach). A returned reason counts as CONGESTION.
    Returns a dict; `offset_mm` = None means a MEASURED CONGESTION FINDING, not an error."""
    kand = kandidatoffset(friheter, steg_mm, tak_mm)
    prov = []
    for d in kand:
        rand = randvillkor(d) if randvillkor else None
        if rand:
            prov.append({"offset_mm": list(d), "avvisad_av": "randvillkor", "skal": rand})
            continue
        konflikt = None
        for namn, p in sorted(grupp.items()):
            q = prim_flytta(p, d)
            for R in rum_lista:
                if namn in R.agare:
                    continue
                sep = _prim_separation(q, R)
                if sep < marginal_mm:
                    konflikt = {"del": namn, "rum": R.namn, "separation_mm": round(sep, 3)}
                    break
            if konflikt:
                break
        if konflikt is None:
            log(f"  negativrum: losning funnen vid offset {d} mm")
            return {"LOST": True, "offset_mm": list(d), "n_provade": len(prov) + 1,
                    "n_kandidater": len(kand), "avvisade": prov[:40],
                    "metod": "deterministisk rutsokning sorterad pa |offset|; forsta giltiga vinner"}
        prov.append({"offset_mm": list(d), "avvisad_av": "negativrum", **konflikt})
    # CONGESTION FINDING: name which rooms compete, and over what volume
    per_rum = {}
    for p in prov:
        if p.get("rum"):
            per_rum[p["rum"]] = per_rum.get(p["rum"], 0) + 1
    return {"LOST": False, "offset_mm": None, "n_kandidater": len(kand),
            "TRANGSEL": True,
            "konkurrerande_rum": sorted(per_rum.items(), key=lambda kv: -kv[1]),
            "gruppvolym_mm3": round(sum(prim_volym(p) for p in grupp.values()), 3),
            "rumsvolym_mm3": {R.namn: round(sum(prim_volym(p) for p in R.prims), 3)
                              for R in rum_lista},
            "avvisade": prov[:120],
            "dom": ("MEASURED CONGESTION FINDING -- no declared freedom takes the group out of the "
                    "rooms within the boundary conditions. A real physical no, not a failure."),
            "metod": "deterministic grid search sorted on |offset|"}


def _lada_lada_sep(a, b):
    """Exact separation between two axis-aligned boxes (mm; < 0 = they intersect).

    Positive: euclidean distance between the boxes. Negative: smallest overlap along some axis."""
    gaps = [max(a[i] - b[i + 3], b[i] - a[i + 3]) for i in range(3)]
    pos = [g for g in gaps if g > 0.0]
    return math.sqrt(sum(g * g for g in pos)) if pos else max(gaps)


def _prim_separation(q, R, n=4096, seed=11):
    """Smallest SDF value over primitive q's own points against room R (mm; <0 = intersects).

    BOX AGAINST BOX IS EXACT, NOT SAMPLED. A sampled test can MISS a thin intersection and let
    through a placement that cuts the room -- exactly the fail-open this module exists to close.
    Only when a non-box is involved do we fall back on the point cloud, and then with q's corners
    AND a deterministic cloud INSIDE q."""
    b = prim_bbox(q)
    if q["typ"] == "lada" and all(p["typ"] == "lada" for p in R.prims):
        return min(_lada_lada_sep(b, prim_bbox(p)) for p in R.prims)
    horn = np.array([[b[i], b[j], b[k]] for i in (0, 3) for j in (1, 4) for k in (2, 5)],
                    dtype=np.float64)
    rng = np.random.default_rng(seed)
    lo, hi = np.array(b[:3]), np.array(b[3:])
    P = np.vstack([horn, lo + (hi - lo) * rng.random((n, 3))])
    if q["typ"] != "lada":                       # keep only points lying INSIDE the primitive
        inne = _sdf_prim(q, P) <= 0.0
        P = np.vstack([P[inne], np.array([q["center"]])])
    return float(np.min(R.sdf(P)))


# ==================================================================================================
# WRITING
# ==================================================================================================
def skriv_negativrum(sokvag, maskin, rum_lista, grindar=None, extra=None):
    d = {
        "_doc": ("Negative spaces: named voids declared BEFORE the material. The material is "
                 "the complement. Every room carries its source (the joint or requirement that "
                 "generates it), its class and its owners (the parts allowed inside it). The "
                 "field (falt_ikarus) is the pre-filter; the verdict is exact B-rep against the same declaration."),
        "schema": "negativrum_v1",
        "generator": "negativrum_v1.py",
        "maskin": maskin,
        "genererad": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "klasser": KLASS_DOC,
        "regel_keepout": "material outside a room's owners may never intersect the room (fail-closed)",
        "n_rum": len(rum_lista),
        "rum": [R.som_dict() for R in rum_lista],
    }
    if grindar:
        d["grindar"] = grindar
    if extra:
        d.update(extra)
    os.makedirs(os.path.dirname(sokvag), exist_ok=True)
    with open(sokvag + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    os.replace(sokvag + ".tmp", sokvag)
    return sokvag


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ==================================================================================================
# SELFTEST -- a reference pair that can fail
# ==================================================================================================
def _selftest():
    ut = {"prov": []}

    def p(namn, ok, **kv):
        ut["prov"].append({"namn": namn, "GRON": bool(ok), **kv})
        print(f"  {'GRON' if ok else 'ROD '}  {namn}  {kv}")

    # --- P1 field parity on all three primitive types
    R1 = Rum("prov_lada", "kyl", {"typ": "krav", "krav_id": "prov"},
             [lada((0, 0, 0), (50, 30, 20))], motiv="test room")
    R2 = Rum("prov_cyl", "media", {"typ": "krav", "krav_id": "prov"},
             [cylinder((10, 0, 0), "x", 25.0, 120.0)], motiv="test room")
    R3 = Rum("prov_kon", "service", {"typ": "krav", "krav_id": "prov"},
             [kon((0, 0, 0), "z", 20.0, 60.0, 100.0)], motiv="test room")
    for R in (R1, R2, R3):
        g = falt_paritet(R, n=4000, seed=3)
        p(f"P1 field parity {R.namn}", g["GRON"], **{k: g[k] for k in ("n_oense", "n_provade")})

    # --- P1b WATERTIGHTNESS GATE: a closed body PASSES, an OPEN SHELL FAILS
    bd0 = _bd()
    g = vattentathet(R1)
    p("P1b watertightness: closed box passes", g["GRON"], chi=g["euler_chi"],
      shells=f"{g['n_slutna_shells']}/{g['n_shells']}")

    class _Trasig:                      # same interface, deliberately an OPEN surface
        namn = "trasigt_skal"

        def solid(self):
            lada_ = bd0.Box(60, 40, 20)
            return bd0.Shell(lada_.faces()[:-1])      # one cap removed => not watertight

    g = vattentathet(_Trasig())
    p("P1b2 watertightness: OPEN shell FAILS", not g["GRON"], chi=g.get("euler_chi"),
      slutna=g.get("n_slutna_shells"), volym=g.get("volym_mm3"))

    # --- P2 the union field = min of the parts, and the solid is valid
    RU = Rum("prov_union", "rorelsesvep", {"typ": "led", "joint": "prov"},
             [lada((0, 0, 0), (50, 30, 20)), lada((90, 0, 0), (50, 30, 20))],
             motiv="two boxes overlapping at x=40..40")
    g = falt_paritet(RU, n=6000, seed=5)
    p("P2 field parity union", g["GRON"], **{k: g[k] for k in ("n_oense", "n_provade")})

    # --- P3 SWEEP COVERAGE: a room that is TOO SMALL must fail, a covering room must pass
    bd = _bd()
    svep = [("kropp", bd.Box(180.0, 40.0, 30.0))]          # x -90..90
    R_liten = Rum("for_litet", "rorelsesvep", {"typ": "led", "joint": "prov"},
                  [lada((0, 0, 0), (60, 30, 20))], motiv="avsiktligt for litet")
    R_tack = Rum("tackande", "rorelsesvep", {"typ": "led", "joint": "prov"},
                 [lada((0, 0, 0), (95, 30, 20))], motiv="tacker svepet med marginal")
    g_l = sveptackning(R_liten, svep)
    g_t = sveptackning(R_tack, svep)
    p("P3a room too small FAILS", not g_l["GRON"], utanfor_mm3=g_l["utanfor_rummet_mm3"])
    p("P3b covering room PASSES", g_t["GRON"], utanfor_mm3=g_t["utanfor_rummet_mm3"])

    # --- P4 KEEP-OUT: a part inside the room is failed, the owner is not
    rum = [Rum("bana", "rorelsesvep", {"typ": "led", "joint": "prov"},
               [lada((0, 0, 0), (100, 30, 30))], agare=["dorrblad"], motiv="banan")]
    inne = bd.Pos(20, 0, 0) * bd.Box(40, 40, 40)
    agarkropp = bd.Pos(0, 0, 0) * bd.Box(40, 40, 40)
    ute = bd.Pos(300, 0, 0) * bd.Box(40, 40, 40)
    moln = {"vaxel": np.array([[20.0, 0, 0]]), "dorrblad": np.array([[0.0, 0, 0]]),
            "fjarran": np.array([[300.0, 0, 0]])}
    g = keepout_dom(rum, {"vaxel": (inne, moln["vaxel"]),
                          "dorrblad": (agarkropp, moln["dorrblad"]),
                          "fjarran": (ute, moln["fjarran"])})
    p("P4 keep-out fails the intruder, clears the owner and the distant part",
      (not g["GRON"]) and g["n_brott"] == 1 and g["brott"][0]["del"] == "vaxel",
      n_brott=g["n_brott"], brott=[b["del"] for b in g["brott"]],
      n_exakt=g["n_par_exakt_raknade"])

    # --- P5 SOLVER: finds the smallest displacement out of the room
    grupp = {"vaxel": lada((20, 0, 0), (20, 20, 20))}
    r = losa_placering(grupp, rum, friheter=[(0, +1)], steg_mm=5.0, tak_mm=300.0, log=lambda *a: None)
    p("P5a solver finds the smallest offset out of the path", r["LOST"] and r["offset_mm"][0] == 100.0,
      offset=r.get("offset_mm"))

    # --- P5b CONGESTION FINDING: same group, boundary conditions that forbid the solution
    r2 = losa_placering(grupp, rum, friheter=[(0, +1)], steg_mm=5.0, tak_mm=300.0,
                        randvillkor=lambda d: "kapans innermatt" if d[0] > 50.0 else None,
                        log=lambda *a: None)
    p("P5b congestion finding instead of a silent error", (not r2["LOST"]) and r2.get("TRANGSEL") is True,
      konkurrerande=r2.get("konkurrerande_rum"))

    ut["GRON"] = all(x["GRON"] for x in ut["prov"])
    ut["n_prov"] = len(ut["prov"])
    print(json.dumps(ut, ensure_ascii=False))
    return 0 if ut["GRON"] else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
