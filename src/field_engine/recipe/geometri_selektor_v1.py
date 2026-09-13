#!/usr/bin/env python3
"""geometri_selektor_v1.py -- address faces and edges by geometry, never by index.

A kernel index names a position in the topology enumeration, so any parameter change that alters
topology renumbers every subshape after it and a stored edges=[3, 7] keeps validating while silently
addressing different edges. The selectors here instead describe what the geometry IS (type, radius,
normal, direction, area bounds, wall membership, proximity to a point) and refuse when the
description no longer picks out the expected number of entities.

API:
    selector_match(shape, selector: dict, context: dict | None = None) -> list[Face] | list[Edge]
        ValueError    -- unknown key, missing required key, entity/key mismatch
        RefusalError  -- the match count differs from the selector's expected_count, in either
                         direction; `context` (op id, recipe, preceding steps) is mirrored into
                         RefusalError.diagnosis["history"]
    wrap_topods(topods_shape) -> build123d.Solid
    record_expected_population(shape, entity, geometry_type=None) -> dict
        Authoring-time helper: measures the current population so it can be stored in an op's
        "expected_population" field.
    check_all_population(shape, entity, expected_population=None, *, geometry_type=None, op_id=None,
                         context=None) -> dict
        Execution-time guard for a chain position that acts on ALL edges or faces. A declared
        population that has drifted raises RefusalError(POPULATION_DRIFT). An undeclared one warns on
        stderr and records a machine-readable flag instead of refusing, so older recipes keep running
        while being counted.
    drain_population_warnings() -> list[dict]
        Empties and returns the process-local flag log, one entry per undeclared ALL call.
    geometric_edge_population(shape, selector) -> list[Edge]
    local_wall_thickness_mm(shape, point) -> float

Every RefusalError carries `.reason` in {POPULATION_DRIFT, REFERENCE_LOST, AMBIGUOUS_NEAREST,
SCALE_DEGENERATE} and a `.diagnosis` dict (reason, expected/actual, candidate sample, entity,
selector keys, history, distances where relevant).

Run the selftest with `python geometri_selektor_v1.py`; it prints a JSON report and exits non-zero if
a check fails.
"""
from __future__ import annotations

import sys
from typing import Any

try:
    from build123d import Axis, CenterOf, Solid, Vector
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "geometri_selektor_v1 kraver build123d -- kor under .venv-cad/bin/python3"
    ) from _e

# --------------------------------------------------------------------------------- slutet nyckelset
REQUIRED_KEYS = {"entity", "expected_count"}
FACE_ONLY_KEYS = {"normal", "normal_tol_deg", "min_area", "max_area", "wall"}
EDGE_ONLY_KEYS = {"direction", "direction_tol_deg", "min_length", "max_length",
                   "vertex_safe_k", "blend_radius_mm"}
COMMON_KEYS = {"geometry_type", "radius", "radius_tol_mm", "near_point", "near_point_max_dist_mm"}
ALLOWED_KEYS = REQUIRED_KEYS | FACE_ONLY_KEYS | EDGE_ONLY_KEYS | COMMON_KEYS

DEFAULT_NORMAL_TOL_DEG = 1.0
DEFAULT_DIRECTION_TOL_DEG = 1.0
DEFAULT_RADIUS_TOL_MM = 1e-6

# 04S2: near_point RANKS candidates but used to throw the ranking away -- len(candidates) !=
# expected_count (the pre-fix line below) counted the WHOLE ranked list, so any selector of the form
# "cylinder nearest (x,y,z)" refused every time the part had more than one candidate of that type,
# regardless of how obviously nearest the top hit was. MEASURED (facegrammatik_04): truncating to
# the expected_count nearest lifts reference survival 15.0%->78.6% over the frozen 220-case corpus,
# but truncating BLINDLY (no margin check) is exactly the rival resolver's silent-wrong source (arm
# E, 2/220) -- so truncation is gated on a DECISIVE margin between the last kept candidate and the
# first excluded one; an indecisive margin refuses (KRAV R2: silent-wrong <=0.1%) instead of
# guessing on canonical tie order.
NEAR_POINT_TIE_BAND = 0.15  # (d_next-d_last)/(d_next+eps) below this => refuse, never truncate
_NEAR_POINT_TIE_EPS_MM = 1e-9

# --------------------------------------------------------------------------------- refusal-taxonomi
REASON_POPULATION_DRIFT = "POPULATION_DRIFT"
REASON_REFERENCE_LOST = "REFERENCE_LOST"
REASON_AMBIGUOUS_NEAREST = "AMBIGUOUS_NEAREST"
REASON_SCALE_DEGENERATE = "SCALE_DEGENERATE"
REFUSAL_REASONS = (
    REASON_POPULATION_DRIFT, REASON_REFERENCE_LOST, REASON_AMBIGUOUS_NEAREST, REASON_SCALE_DEGENERATE,
)


def _classify_refusal_reason(selector: dict, actual_count: int) -> str:
    """Deterministic, always-classifying (no None fallback -- an unclassified refusal is as bad as an
    unrefused drift). Order, most specific first:
      1. 0 hits                        -> REFERENCE_LOST (the named entity is gone)
      2. near_point in the selector    -> AMBIGUOUS_NEAREST (the ranking could not separate)
      3. radius in the selector        -> SCALE_DEGENERATE (dimension/scale selection ambiguous)
      4. otherwise                     -> POPULATION_DRIFT (the candidate set's SIZE drifted)
    """
    if actual_count == 0:
        return REASON_REFERENCE_LOST
    if "near_point" in selector:
        return REASON_AMBIGUOUS_NEAREST
    if "radius" in selector or "vertex_safe_k" in selector:
        return REASON_SCALE_DEGENERATE
    return REASON_POPULATION_DRIFT


class RefusalError(Exception):
    """Fail-closed: expected_count/expected_population did not match. Carries {expected_count,
    actual_count, available} plus `.reason` (one of REFUSAL_REASONS) and `.diagnosis` (a
    machine-readable dict: candidates, distances where relevant, history). Never a best guess, never an
    index into `available`.
    """

    def __init__(self, expected_count: int, actual_count: int, available: list[dict], *,
                 reason: str | None = None, diagnosis: dict | None = None):
        self.expected_count = expected_count
        self.actual_count = actual_count
        self.available = available
        self.reason = reason if reason in REFUSAL_REASONS else REASON_POPULATION_DRIFT
        self.diagnosis = diagnosis if diagnosis is not None else {
            "reason": self.reason, "expected_count": expected_count, "actual_count": actual_count,
            "candidates_sample": available[:25] if available else [],
        }
        super().__init__(
            f"selector_match VAGRAN [{self.reason}]: expected_count={expected_count} "
            f"actual_count={actual_count} available={available[:5]}{'...' if len(available) > 5 else ''}"
        )


# --------------------------------------------------------------------------------- validering
def _validate_selector(selector: dict) -> None:
    if not isinstance(selector, dict):
        raise ValueError(f"selector maste vara dict, fick {type(selector).__name__}")
    keys = set(selector.keys())
    unknown = keys - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"okand nyckel i selector (ADR-029): {sorted(unknown)}")
    missing = REQUIRED_KEYS - keys
    if missing:
        raise ValueError(f"required key missing: {sorted(missing)}")
    entity = selector["entity"]
    if entity not in ("face", "edge"):
        raise ValueError(f"entity must be 'face' or 'edge', got {entity!r}")
    ec = selector["expected_count"]
    if not isinstance(ec, int) or isinstance(ec, bool) or ec < 0:
        raise ValueError(f"expected_count maste vara icke-negativ int, fick {ec!r}")
    if entity == "edge":
        bad = keys & FACE_ONLY_KEYS
        if bad:
            raise ValueError(f"nyckel {sorted(bad)} galler endast entity='face' (fick entity='edge')")
    if entity == "face":
        bad = keys & EDGE_ONLY_KEYS
        if bad:
            raise ValueError(f"nyckel {sorted(bad)} galler endast entity='edge' (fick entity='face')")
    has_k = "vertex_safe_k" in keys
    has_r = "blend_radius_mm" in keys
    if has_k != has_r:
        raise ValueError("vertex_safe_k and blend_radius_mm must be given TOGETHER (both or neither)")
    if "wall" in keys and selector["wall"] not in ("outer", "inner"):
        raise ValueError(f"wall must be 'outer' or 'inner', got {selector['wall']!r}")


# ------------------------------------------------------------------------------ safe extractors
# (never crash on the wrong face/edge type -- exclude the candidate, never guess)
def _safe_radius(c) -> float | None:
    try:
        r = c.radius
        return float(r) if r is not None else None
    except Exception:
        return None


def _ref_point(c) -> Vector:
    """build123d Face/Edge.center() defaults to CenterOf.GEOMETRY, which for a non-planar face
    (CYLINDER/CONE/SPHERE/TORUS) or a non-LINE edge (CIRCLE/ELLIPSE) evaluates the surface or curve at
    its mid-UV or mid-parameter: a point ON THE BOUNDARY, offset from the geometric axis or midpoint by
    up to the radius, in a direction that depends on the surface's internal u=0 seam. Reproduced: a
    Cylinder(r=5) centred on the world origin gives default.center() = (-5.0, ~0, 0), i.e. five
    millimetres off, on the boundary, not on the axis. CenterOf.MASS gives the true centroid (same
    cylinder -> (~0,~0,0); a full CIRCLE edge -> the circle's true midpoint). Used ONLY for near_point
    ranking and _describe() diagnostics, never for _safe_normal, which needs a point ON the surface for
    normal_at(), where the boundary point of the GEOMETRY default is the right semantics.
    """
    try:
        return c.center(CenterOf.MASS)
    except Exception:
        return c.center()


def _safe_normal(f) -> Vector | None:
    try:
        return f.normal_at(f.center())
    except Exception:
        return None


def _safe_tangent(e) -> Vector | None:
    try:
        if e.geom_type.name != "LINE":
            return None  # direction ar ENDAST ratt-definierad for raka kanter (se docstring)
        return e.tangent_at(0.5)
    except Exception:
        return None


_WALL_RAY_EPS_MM = 1e-3
_WALL_RAY_TOL_MM = 1e-6
_OWN_SOLID_BBOX_EPS_MM = 1e-6


def _own_solid_ray_cap_mm(shape, face, n: Vector) -> float | None:
    """wall_pattern_fix_v1. ROOT CAUSE: _wall_side's ray-cast runs against the WHOLE (possibly
    patterned) shape with NO length limit -- when a pattern's translation axis is COLLINEAR with the
    probed face's normal, a ray leaving one copy's true outer face travels straight down the inter-
    copy gap and registers a 'forward hit' against a NEIGHBOURING copy's near wall, silently
    misclassifying an outer face as 'inner' (1/5 instead of 3/3 on the repro).

    FIX: cap the ray at the reach of the candidate's OWN solid -- a hit belonging to a different solid
    (a different pattern copy) can never be nearer than that solid's own bounding-box extent along
    the normal, so capping the ray there removes cross-copy contamination WITHOUT needing to know
    anything about "which copy" a hit face belongs to. Returns the cap in mm (>=0), or None if no
    owning solid could be identified (fail-safe -- _wall_side then falls back to the historical
    unbounded ray, identical to pre-fix behaviour for that one candidate rather than risk excluding
    a legitimate hit).

    WHY THIS DOES NOT DISTURB THE B1/B7 (single-solid / orthogonal-pattern) CASES (measured, not
    assumed -- see selftest AT8/AT9 and the kernel_repro_suite_v1 unchanged claim): when the pattern
    axis is ORTHOGONAL to the probed normal, every copy's bounding box has the IDENTICAL extent
    along the normal axis (only the orthogonal coordinate differs between copies), so the cap equals
    the single-shape extent regardless of which copy owns the candidate -- the same number the
    unbounded ray would have produced. When there is only ONE solid (B1), the cap trivially equals
    the whole shape's own extent -- again identical to the unbounded ray. The cap only CHANGES the
    outcome when the pattern axis is collinear with the normal, which is exactly the failure this
    fixes.
    """
    try:
        solids = list(shape.solids())
    except Exception:
        return None
    if not solids:
        return None
    pt = face.center()
    owner = solids[0]
    if len(solids) > 1:
        owner = None
        eps = _OWN_SOLID_BBOX_EPS_MM
        for s in solids:
            try:
                bb = s.bounding_box()
            except Exception:
                continue
            if (bb.min.X - eps <= pt.X <= bb.max.X + eps
                    and bb.min.Y - eps <= pt.Y <= bb.max.Y + eps
                    and bb.min.Z - eps <= pt.Z <= bb.max.Z + eps):
                owner = s
                break
        if owner is None:
            return None  # fail-safe: no bbox contains the reference point -- do not guess an owner
    try:
        bb = owner.bounding_box()
    except Exception:
        return None
    corners = (
        (bb.min.X, bb.min.Y, bb.min.Z), (bb.max.X, bb.min.Y, bb.min.Z),
        (bb.min.X, bb.max.Y, bb.min.Z), (bb.min.X, bb.min.Y, bb.max.Z),
        (bb.max.X, bb.max.Y, bb.min.Z), (bb.max.X, bb.min.Y, bb.max.Z),
        (bb.min.X, bb.max.Y, bb.max.Z), (bb.max.X, bb.max.Y, bb.max.Z),
    )
    base_proj = pt.dot(n)
    max_proj = max(Vector(*c).dot(n) for c in corners)
    cap = max_proj - base_proj
    return cap if cap > 0.0 else 0.0


def _wall_side(shape, face) -> str | None:
    """Inner/outer discriminator. Casts a line (build123d Solid.faces_intersected_by_axis, an OCCT line
    intersection, not a guess) from the face's own normal point (f.center(), the same GEOMETRY point
    _safe_normal uses for normal_at()) along the outward normal, offset _WALL_RAY_EPS_MM forward so the
    starting face itself lands BEHIND the origin and never counts as a forward hit. Hits with t > eps
    are counted: 0 forward hits -> "outer" (the line leaves the solid and meets nothing more, a true
    outer face); >= 1 forward hit -> "inner" (the normal points INTO a cavity and meets the opposite
    shell wall).

    The ray cast is chosen over an area or bounding-box heuristic because it is shape-independent: a
    bounding-box comparison is only well defined for an axis-aligned planar face against an axis-aligned
    box, and does not generalise to a curved shell wall or a part that is not axis-aligned. The ray cast
    only needs normal_at() to be well defined, which the normal filter already requires, so it costs no
    new precondition.

    Measured on a shelled box: the true outer wall (center=[50,0,20], normal=+X) gives 0 hits, while the
    inner wall's corresponding face (center=[-48,0,21], normal=+X, the SAME normal direction -- which is
    exactly why the selector is ambiguous without this key) gives 2 hits, because the ray crosses the
    cavity and meets BOTH faces of the opposite wall on its way out of the material.

    Fail-safe: any geometric operation that fails (a degenerate face, say) returns None, so the candidate
    takes no part in the wall filtering or the diagnosis -- it is neither included nor excluded on a
    guess, the same pattern as the module's other _safe_*() helpers.

    Own-solid limit: a hit only counts as forward if it also lies within the measured bounding-box
    extent of the candidate's OWN solid along the normal (_own_solid_ray_cap_mm). Without that, a ray
    leaving a patterned copy's true outer face can continue straight through the gap and hit the NEXT
    copy's wall, silently misclassifying an outer face as inner (measured: outer=1/inner=5 instead of
    outer=3/inner=3 when the pattern axis is collinear with the probed normal). The cap is identical to
    the previous unbounded behaviour when the shape is a single solid, or when the pattern axis is
    orthogonal to the normal, since then every copy's extent along the normal axis is the same. See
    selftest AT10/AT11.
    """
    hits = _wall_forward_hits_mm(shape, face)
    if hits is None:
        return None
    return "outer" if len(hits) == 0 else "inner"


def _wall_forward_hits_mm(shape, face) -> list[float] | None:
    """Shared ray-cast core, split out of _wall_side so the SIGSEGV wall-thickness guard
    (local_wall_thickness_mm below, consumed by cad_op_exec_v1's pre-fillet/chamfer degeneracy
    check) can reuse the exact same MEASURED mechanism -- same ray origin (face.center() nudged
    _WALL_RAY_EPS_MM along the outward normal), same _own_solid_ray_cap_mm cross-copy cap, same
    forward-hit filter (t>eps, t<=cap) -- rather than a second, unproven ray-cast implementation.
    Returns the SORTED list of forward hit distances (mm) -- [] means 'outer' (no opposing wall
    found, _wall_side's 0-forward-hits case), None means the ray itself could not be computed (fail-
    safe, identical contract to _wall_side's pre-refactor None).
    """
    n = _safe_normal(face)
    if n is None:
        return None
    try:
        n = n.normalized()
        origin = face.center() + n * _WALL_RAY_EPS_MM
        axis = Axis(origin, tuple(n))
        hits = shape.faces_intersected_by_axis(axis, tol=_WALL_RAY_TOL_MM)
    except Exception:
        return None
    ray_cap_mm = _own_solid_ray_cap_mm(shape, face, n)
    out = []
    for h in hits:
        try:
            t = (h.center() - origin).dot(n)
        except Exception:
            continue
        if t > _WALL_RAY_EPS_MM and (ray_cap_mm is None or t <= ray_cap_mm):
            out.append(t)
    out.sort()
    return out


def _edge_direction(e) -> tuple | None:
    """Unit chord direction (start->end) of edge `e`, or None if degenerate/unavailable. Exact for a
    LINE edge; an approximation (chord, not tangent) for a curved edge -- adequate here because this
    is only used as a cheap PRE-FILTER for 'does this other edge run roughly alongside the candidate
    edge', never as the distance measurement itself (that is always the exact BRepExtrema value)."""
    try:
        p0 = tuple(e.start_point())
        p1 = tuple(e.end_point())
    except Exception:
        return None
    dx = tuple(p1[i] - p0[i] for i in range(3))
    n = sum(v * v for v in dx) ** 0.5
    if n <= 1e-9:
        return None
    return tuple(v / n for v in dx)


_FLANGE_PARALLEL_DOT_TOL = 0.999  # cos(~2.6 deg) -- 'runs alongside', not 'crosses at a corner'


def local_wall_thickness_mm(shape, edges: list, near_hint_mm: float | None = None) -> float | None:
    """LOCAL flange/wall thickness AT `edges`. Used by cad_op_exec_v1's pre-fillet/ chamfer SIGSEGV
    guard.

    REFRAME DECLARED: the ORIGINAL implementation measured 'the exact minimum BRepExtrema distance
    between each of `edges`' adjacent faces and EVERY OTHER face of the WHOLE `shape`' -- a GLOBAL
    scan. MEASURED FALSE POSITIVE: the RAW OCC bd.fillet() call on this exact geometry SUCCEEDS
    cleanly (no crash, volume unchanged to the mm3, confirmed via a direct isolated repro with the
    guard disabled) -- yet the global scan flags it, because BRepExtrema over 'every other face of
    shape' happens to find SOME unrelated face (here: the tube's OWN opposite-side inner wall, an
    accident of the tube being 90mm across with a 6mm wall on BOTH sides) sitting exactly `radius`
    away, even though THAT face has nothing to do with the fillet edges actually being blended.
    In other words, BRepExtrema can find a narrow gap between faces that are not the wall being
    filleted at all.

    MEASURED TRUE mechanism: the shape that CRASHES is a shell(thickness=5) box with its top face
    removed -- OCC's shell builder closes that opening with a thin RIBBON face (width == shell
    thickness) connecting the outer top rim to the inner top rim. Filleting the OUTER rim edge
    (shared between the tube's outer side wall and that 5mm-wide ribbon face) at radius==5.0 crashes
    (confirmed exit 139); filleting the INNER rim edge or the (unrelated, still-capped) BOTTOM rim
    edge at the same radius does NOT crash. The distinguishing, measurable fact: the crashing edge
    has, among the OTHER edges of its OWN two directly-adjacent faces, a PARALLEL edge running
    alongside it at a distance exactly equal to the fillet radius (the ribbon face's own far
    boundary) -- i.e. the degeneracy is a property of the edge's own LOCAL neighbourhood (its own
    faces' own other edges), never of some unrelated face elsewhere in the body. Re-run against BOTH
    founding cases: the F2 tube's corner edges have no such parallel-alongside partner within
    radius-distance of themselves (their own adjacent faces are 78-90mm wide, not 6mm) -- guard
    correctly stays silent; the shell recipe's top-rim edges do (5mm partner, exact radius match) --
    guard correctly still fires.

    METHOD: for each edge in `edges`, find its OWN directly-adjacent faces (faces of `shape` whose
    own edge list contains this edge, topological IsSame -- typically 1-2 faces, never a global face
    list), then for each OTHER edge of THOSE faces that runs roughly parallel to the candidate edge
    (_FLANGE_PARALLEL_DOT_TOL), measure the exact BRepExtrema distance between the two edges
    (skipping <=1e-6mm shared-vertex touches). Returns the minimum such distance across all `edges`,
    or None when no parallel-alongside partner exists anywhere in the candidate edges' own local
    neighbourhood (the common case -- no SIGSEGV risk from this mechanism, caller must not fabricate
    a thickness and must not guard). `near_hint_mm` is accepted for call-site compatibility but no
    longer used for pruning -- the search space is now O(n_edges * faces_per_edge * edges_per_face),
    a handful of comparisons per candidate edge (not the whole body's face count), so the AABB pre-
    filter that used to be load-bearing for wall-clock time is no longer needed.
    """
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    def _faces_of_edge(e) -> list:
        out = []
        for f in shape.faces():
            try:
                f_edges = f.edges()
            except Exception:
                continue
            if any(fe.wrapped.IsSame(e.wrapped) for fe in f_edges):
                out.append(f)
        return out

    best = None
    for e in edges:
        e_dir = _edge_direction(e)
        if e_dir is None:
            continue
        for f in _faces_of_edge(e):
            try:
                f_edges = f.edges()
            except Exception:
                continue
            for oe in f_edges:
                if oe.wrapped.IsSame(e.wrapped):
                    continue
                oe_dir = _edge_direction(oe)
                if oe_dir is None:
                    continue
                dot = abs(sum(e_dir[i] * oe_dir[i] for i in range(3)))
                if dot < _FLANGE_PARALLEL_DOT_TOL:
                    continue  # crosses/meets the candidate edge -- not a 'runs alongside' flange wall
                try:
                    dss = BRepExtrema_DistShapeShape(e.wrapped, oe.wrapped)
                    if not dss.IsDone():
                        continue
                    d = float(dss.Value())
                except Exception:
                    continue
                if d <= 1e-6:
                    continue  # shared-vertex touch, not a wall-thickness signal
                if best is None or d < best:
                    best = d
    return best


def _wall_ambiguity_diagnosis(shape, candidates: list, limit: int = 25) -> dict | None:
    """Builds the AMBIGUOUS_NEAREST diagnosis for a face-normal selector without a "wall" key. Returns
    None when the ray-cast sides do not actually separate the candidates (for example a purely
    pattern-driven over-hit where ALL copies sit on the same side -- that is POPULATION_DRIFT, see
    _detect_pattern_multiple, not an inner/outer question). Always NAMES every candidate with its
    measured wall_side; never a silent first hit.
    """
    sided = [(c, _wall_side(shape, c)) for c in candidates[:limit]]
    classes = sorted({s for _, s in sided if s is not None})
    if len(classes) < 2:
        return None
    named = []
    for c, side in sided:
        d = _describe(c, "face")
        d["wall_side"] = side
        named.append(d)
    return {
        "candidates": named,
        "note": (
            f"{len(classes)} skilda wall-sidor bland kandidaterna {classes} -- ange "
            "selector['wall']='outer'|'inner' for att disambiguera (ray-cast fran ansiktets "
            "centroid langs normalen: 'outer'=0 framatriktade traffar/lamnar soliden, "
            "'inner'>=1 traff mot motstaende skal)."
        ),
    }


def _detect_pattern_multiple(candidates: list, entity: str, expected_count: int) -> dict | None:
    """Pattern-aware count diagnosis. When a population has been multiplied upstream (pattern_linear,
    mirror) the surplus candidates are usually spread REGULARLY along ONE axis. Measured, not guessed:
    the candidates' mass centres (_ref_point) are sorted along each axis in turn, and if ALL consecutive
    gaps on an axis are equal (within 2 % relative tolerance) AND the candidate count divides evenly by
    expected_count, a hypothesis dict is returned (observed_multiple = N/expected). Never an
    auto-correction of expected_count -- only a proposed cause in the diagnosis, which a text-to-CAD
    loop may act on. Returns None when no axis shows regular spacing (a circular pattern, say, or a
    genuine irregular drift): no hypothesis is better than a wrong one here.
    """
    n = len(candidates)
    if expected_count <= 0 or n <= expected_count or n % expected_count != 0:
        return None
    try:
        centers = [_ref_point(c) for c in candidates]
    except Exception:
        return None
    for axis_name in ("X", "Y", "Z"):
        vals = sorted(round(getattr(v, axis_name), 6) for v in centers)
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if not diffs or any(abs(d) <= 1e-6 for d in diffs):
            continue  # duplicate positions on this axis -- no clean 1-D spacing here
        mean_d = sum(diffs) / len(diffs)
        if mean_d <= 0:
            continue
        if all(abs(d - mean_d) <= 0.02 * mean_d for d in diffs):
            multiple = n // expected_count
            return {
                "observed_count": n, "expected_count": expected_count, "observed_multiple": multiple,
                "axis": axis_name, "spacing_mm": round(mean_d, 6),
                "hypothesis": (
                    f"count {n} observed, an upstream pattern x{multiple} is the likely cause "
                    f"(regelbunden spacing {round(mean_d, 4)}mm langs {axis_name})"
                ),
            }
    return None


def _describe(c, entity: str) -> dict:
    """GEOMETRISK deskriptor for RefusalError.available -- ALDRIG ett index."""
    d: dict[str, Any] = {"geometry_type": c.geom_type.name, "center": [round(v, 4) for v in tuple(_ref_point(c))]}
    r = _safe_radius(c)
    if r is not None:
        d["radius_mm"] = round(r, 6)
    if entity == "face":
        d["area_mm2"] = round(c.area, 6)
    else:
        d["length_mm"] = round(c.length, 6)
    return d


# --------------------------------------------------------------------------------- varm-scen-brygga
def wrap_topods(topods_shape):
    """Downcasts and wraps a raw OCP TopoDS_Shape (TopAbs_SOLID, transform already applied) into a
    build123d.Solid, so the SAME selector_match() serves an in-memory shape and a standalone STEP
    import -- one implementation, no drift between two code paths.
    """
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopoDS import TopoDS

    if topods_shape.ShapeType() != TopAbs_ShapeEnum.TopAbs_SOLID:
        raise ValueError(f"wrap_topods VAGRAN: forvantade TopAbs_SOLID, fick {topods_shape.ShapeType()}")
    return Solid(TopoDS.Solid_s(topods_shape))


# --------------------------------------------------------------------------------- huvud-API
def selector_match(shape, selector: dict, context: dict | None = None) -> list:
    """shape: build123d Solid/Compound/Shape (a standalone STEP via import_step(), or an in-memory shape
    via wrap_topods()). selector: the closed key set in the module docstring. context: optional op-chain
    history (for example {"op_id":..., "recipe":..., "prev_op":...}) mirrored straight into any
    RefusalError.diagnosis["history"], which makes the error message self-explanatory without
    correlating separate logs.
    -> list[Face] or list[Edge], of LENGTH == expected_count, otherwise a taxonomy-classified
    RefusalError.
    """
    _validate_selector(selector)
    entity = selector["entity"]
    expected_count = selector["expected_count"]

    candidates = list(shape.faces()) if entity == "face" else list(shape.edges())

    if "geometry_type" in selector:
        gt = str(selector["geometry_type"]).upper()
        candidates = [c for c in candidates if c.geom_type.name == gt]

    if "radius" in selector:
        target_r = float(selector["radius"])
        tol = float(selector.get("radius_tol_mm", DEFAULT_RADIUS_TOL_MM))
        candidates = [c for c in candidates if (r := _safe_radius(c)) is not None and abs(r - target_r) <= tol]

    if "normal" in selector:
        target_n = Vector(*selector["normal"]).normalized()
        tol_deg = float(selector.get("normal_tol_deg", DEFAULT_NORMAL_TOL_DEG))
        out = []
        for c in candidates:
            n = _safe_normal(c)
            if n is not None and n.normalized().get_angle(target_n) <= tol_deg:
                out.append(c)
        candidates = out

    if "direction" in selector:
        target_d = Vector(*selector["direction"]).normalized()
        tol_deg = float(selector.get("direction_tol_deg", DEFAULT_DIRECTION_TOL_DEG))
        out = []
        for c in candidates:
            t = _safe_tangent(c)
            if t is None:
                continue
            ang = t.normalized().get_angle(target_d)
            ang = min(ang, 180.0 - ang)  # a line has no canonical direction
            if ang <= tol_deg:
                out.append(c)
        candidates = out

    if "min_area" in selector:
        candidates = [c for c in candidates if c.area >= float(selector["min_area"])]
    if "max_area" in selector:
        candidates = [c for c in candidates if c.area <= float(selector["max_area"])]
    if "min_length" in selector:
        candidates = [c for c in candidates if c.length >= float(selector["min_length"])]
    if "max_length" in selector:
        candidates = [c for c in candidates if c.length <= float(selector["max_length"])]

    if "wall" in selector:
        want = selector["wall"]
        candidates = [c for c in candidates if _wall_side(shape, c) == want]

    _vertex_safe_diag = None
    if "vertex_safe_k" in selector:
        k = float(selector["vertex_safe_k"])
        blend_radius_mm = float(selector["blend_radius_mm"])
        vinfo = vertex_aware_safe_edges(shape, "edge", blend_radius_mm, k, candidate_pool=candidates,
                                         include_edges=True)
        _vertex_safe_diag = {
            "k": vinfo["k"], "blend_radius_mm": vinfo["blend_radius_mm"], "threshold_mm": vinfo["threshold_mm"],
            "n_excluded_own_short": vinfo["n_excluded_own_short"],
            "n_excluded_vertex_adjacent": vinfo["n_excluded_vertex_adjacent"],
        }
        candidates = vinfo["safe_edges"]

    _near_diag = None
    if "near_point" in selector:
        pt = Vector(*selector["near_point"])
        scored = [(c, _ref_point(c).sub(pt).length) for c in candidates]
        if "near_point_max_dist_mm" in selector:
            cap = float(selector["near_point_max_dist_mm"])
            scored = [(c, d) for c, d in scored if d <= cap]
        scored.sort(key=lambda t: t[1])
        # 04S2: truncate to the expected_count nearest ONLY when the cut margin is decisive; an
        # indecisive margin leaves the over-count in place so the expected_count check below refuses
        # (fail-closed) instead of silently keeping a tie-broken guess (see module-level comment).
        if 0 < expected_count < len(scored):
            d_last = scored[expected_count - 1][1]
            d_next = scored[expected_count][1]
            margin = (d_next - d_last) / (d_next + _NEAR_POINT_TIE_EPS_MM)
            if margin >= NEAR_POINT_TIE_BAND:
                scored = scored[:expected_count]
        # AMBIGUOUS_NEAREST-diagnos: avstanden sjalva (ej bara descriptorn) -- text->CAD-loopen kan
        # show HOW close the two were, not only THAT it refused.
        _near_diag = [{"descriptor": _describe(c, entity), "dist_mm": round(d, 6)} for c, d in scored[:25]]
        candidates = [c for c, _ in scored]

    if len(candidates) != expected_count:
        reason = _classify_refusal_reason(selector, len(candidates))
        avail = [_describe(c, entity) for c in candidates[:25]]

        # INRE/YTTRE-TVETYDIGHET (selektor_wall_v1, mission-krav 1): en face-normal-selektor UTAN
        # "wall" som overtraffar OCH vars overtaliga kandidater faktiskt skiljer sig at pa ray-cast-
        # sida forskjuts till AMBIGUOUS_NEAREST (INTE POPULATION_DRIFT) -- disambiguerbar via "wall",
        # never a bare count mismatch whose cause a caller has to guess.
        wall_amb = None
        if (entity == "face" and "wall" not in selector and "normal" in selector
                and len(candidates) > expected_count):
            wall_amb = _wall_ambiguity_diagnosis(shape, candidates)
            if wall_amb is not None:
                reason = REASON_AMBIGUOUS_NEAREST

        diagnosis = {
            "reason": reason, "expected_count": expected_count, "actual_count": len(candidates),
            "entity": entity, "selector_keys": sorted(selector.keys()),
            "candidates_sample": avail, "history": context or {},
        }
        if wall_amb is not None:
            diagnosis["wall_candidates"] = wall_amb["candidates"]
            diagnosis["wall_ambiguity_note"] = wall_amb["note"]
        elif reason == REASON_POPULATION_DRIFT:
            # PATTERN-MEDVETEN COUNT (selektor_wall_v1, mission-krav 2): forslag, aldrig tyst
            # auto-korrigering -- se _detect_pattern_multiple.
            pattern = _detect_pattern_multiple(candidates, entity, expected_count)
            if pattern is not None:
                diagnosis["pattern_hypothesis"] = pattern
        if _near_diag is not None:
            diagnosis["distances"] = _near_diag
        if _vertex_safe_diag is not None:
            diagnosis["vertex_safe_filter"] = _vertex_safe_diag
        raise RefusalError(expected_count, len(candidates), avail, reason=reason, diagnosis=diagnosis)
    return candidates


_SAFETY_FILTER_KEYS = {"min_length", "max_length", "vertex_safe_k", "blend_radius_mm"}

def geometric_edge_population(shape, selector: dict | None) -> list:
    """. RADIUS_FRAC SEMANTICS DECISION: "measure radius_frac's min-edge-length against the population
    a destructive op will TOUCH" (selector_match's own, fully filtered output) is WRONG, because a
    SAFETY filter (min_length/vertex_safe_k -- keys that exist to PROTECT short/vertex-adjacent
    edges FROM a fillet/chamfer, not to DEFINE what the recipe author geometrically means) removes
    exactly the short edges that carry the shape's TRUE local minimum length. A
    min_length/vertex_safe_k-filtered population's own min edge length is therefore ALWAYS >= the
    shape's true local minimum BY CONSTRUCTION of the filter itself -- scaling radius_frac off that
    inflated floor pushes the resolved radius/distance value up until it exceeds what the surviving
    (safety-filtered) edges can individually tolerate, which is EXACTLY the ceiling an earlier
    stress-test corpus measured (legA: v3's own capped scaling reached depths [5,9,19,5,3,10]; the
    uncapped-but-population-inflated form died at [0,1,0,0, 1,2] -- shallower, not deeper).

    DECISION: radius_frac measures the selector's GEOMETRIC population -- everything the selector's
    geometry-defining filters
    (geometry_type/radius/normal/direction/min_area/max_area/wall/near_point) select -- BEFORE the
    safety filters (min_length/max_length/vertex_safe_k, _SAFETY_FILTER_KEYS) ever run. The radius
    is dimensioned by the GEOMETRY (the true local minimum edge length in the region the selector
    points at); the safety filters then protect the EXECUTION (which of those edges are actually
    safe to touch) -- two separate concerns that selector_match's single filtered output had
    conflated into one population. selector=None -> ALL edges of shape, IDENTICAL to
    selector_match's own ALL contract (no filter exists to skip either way, so this is not a
    behaviour change for that case -- see cad_op_exec_v1._resolve_target_edges).

    ALTERNATIVE REJECTED (a cap/clamp on the resolved value, e.g. adaptiv_radie_rib_v1's own removed
    v3-style min(DEFAULT, min_len*frac)): rejected there already, on measured grounds
    (adaptiv_radie_ rib_v1.json's own docstring: a cap/floor silently falsifies the analytical cube
    reference value, radius== L*radius_frac EXACTLY) -- reopening that decision here would re-break
    the ALREADY-MEASURED reference value contract for no offsetting depth gain.

    Duplicates (rather than reuses) selector_match's own geometric-filter code, deliberately: this
    function must NEVER apply min_length/max_length/vertex_safe_k, so sharing selector_match's
    single linear filter chain (where those keys are interleaved with the geometric ones, not
    contiguous --min_length/max_length precede "wall" in selector_match's own key order) would
    require either a return-value hook into selector_match or threading a skip-set through it; a
    self-contained duplicate keeps selector_match completely untouched (0 regression risk to its own
    selftest) at the cost of two filter chains to keep in sync
    -- acceptable because the two chains share the SAME 4 module-level `_safe_*` primitives, so a
    correctness fix to normal/tangent/radius extraction lands in both automatically.
    """
    if selector is None:
        return list(shape.edges())
    if selector.get("entity") != "edge":
        raise ValueError(
            f"geometric_edge_population: selector entity must be 'edge' (radius_frac only applies to "
            f"fillet/chamfer), got {selector.get('entity')!r}"
        )
    candidates = list(shape.edges())
    if "geometry_type" in selector:
        gt = str(selector["geometry_type"]).upper()
        candidates = [c for c in candidates if c.geom_type.name == gt]
    if "radius" in selector:
        target_r = float(selector["radius"])
        tol = float(selector.get("radius_tol_mm", DEFAULT_RADIUS_TOL_MM))
        candidates = [c for c in candidates if (r := _safe_radius(c)) is not None and abs(r - target_r) <= tol]
    if "normal" in selector:
        target_n = Vector(*selector["normal"]).normalized()
        tol_deg = float(selector.get("normal_tol_deg", DEFAULT_NORMAL_TOL_DEG))
        out = []
        for c in candidates:
            n = _safe_normal(c)
            if n is not None and n.normalized().get_angle(target_n) <= tol_deg:
                out.append(c)
        candidates = out
    if "direction" in selector:
        target_d = Vector(*selector["direction"]).normalized()
        tol_deg = float(selector.get("direction_tol_deg", DEFAULT_DIRECTION_TOL_DEG))
        out = []
        for c in candidates:
            t = _safe_tangent(c)
            if t is None:
                continue
            ang = t.normalized().get_angle(target_d)
            ang = min(ang, 180.0 - ang)
            if ang <= tol_deg:
                out.append(c)
        candidates = out
    if "min_area" in selector:
        candidates = [c for c in candidates if c.area >= float(selector["min_area"])]
    if "max_area" in selector:
        candidates = [c for c in candidates if c.area <= float(selector["max_area"])]
    if "wall" in selector:
        want = selector["wall"]
        candidates = [c for c in candidates if _wall_side(shape, c) == want]
    if "near_point" in selector:
        pt = Vector(*selector["near_point"])
        scored = [(c, _ref_point(c).sub(pt).length) for c in candidates]
        if "near_point_max_dist_mm" in selector:
            cap = float(selector["near_point_max_dist_mm"])
            scored = [(c, d) for c, d in scored if d <= cap]
        scored.sort(key=lambda t: t[1])
        expected_count = selector.get("expected_count", 0)
        if isinstance(expected_count, int) and 0 < expected_count < len(scored):
            d_last = scored[expected_count - 1][1]
            d_next = scored[expected_count][1]
            margin = (d_next - d_last) / (d_next + _NEAR_POINT_TIE_EPS_MM)
            if margin >= NEAR_POINT_TIE_BAND:
                scored = scored[:expected_count]
        candidates = [c for c, _ in scored]
    # deliberately SKIPPED: min_length, max_length, vertex_safe_k/blend_radius_mm -- the safety filters
    # this function exists to measure AROUND, not through (see docstring DECISION).
    return candidates


# --------------------------------------------------------------------------------- ALL-ALL-
# population guard
UNDECLARED_POPULATION_LOG: list[dict] = []


def _entity_list(shape, entity: str, geometry_type: str | None = None) -> list:
    if entity not in ("face", "edge"):
        raise ValueError(f"entity must be 'face' or 'edge', got {entity!r}")
    candidates = list(shape.faces()) if entity == "face" else list(shape.edges())
    if geometry_type:
        gt = str(geometry_type).upper()
        candidates = [c for c in candidates if c.geom_type.name == gt]
    return candidates


def record_expected_population(shape, entity: str, geometry_type: str | None = None) -> dict:
    """Authoring-time helper: MEASURES the current population of a prospective ALL selector (it reads
    shape.faces()/edges() now, it never guesses) and returns a declaration that can be stored verbatim
    in an op's "expected_population" field, so the next run has something to compare against (see
    check_all_population).
    """
    n = len(_entity_list(shape, entity, geometry_type))
    return {"entity": entity, "geometry_type": geometry_type, "expected_population": n}


def check_all_population(shape, entity: str, expected_population: int | None = None, *,
                          geometry_type: str | None = None, op_id: str | None = None,
                          context: dict | None = None) -> dict:
    """Execution-time guard for a selector=None/'ALL' chain position.

    expected_population GIVEN (declared, for example by record_expected_population at authoring time)
    and differing from the actual population -> RefusalError(reason=POPULATION_DRIFT) carrying both
    counts and the delta. Never silent.

    expected_population=None (undeclared, the backward-compatible mode for recipes that do not declare
    one yet) -> does NOT refuse, so existing recipes keep working, but writes a warning to stderr and
    logs a fail-closed flag in UNDECLARED_POPULATION_LOG (also returned as flag=True): a
    machine-readable "this step ran without a population check", never a silent pass.
    """
    actual = len(_entity_list(shape, entity, geometry_type))
    if expected_population is None:
        rec = {
            "op_id": op_id, "entity": entity, "geometry_type": geometry_type,
            "actual_population": actual, "fail_closed_flag": True,
            "warning": ("ALL-selector utan expected_population-deklaration (bakat-kompat-lage, "
                        "SYSTEMSYNEN_V1 §5 SPRAKREGELN) -- populationen ar OKONTROLLERAD."),
        }
        UNDECLARED_POPULATION_LOG.append(rec)
        sys.stderr.write(
            f"[geometri_selektor_v1] VARNING population_guard odeklarerad: op_id={op_id} "
            f"entity={entity} actual_population={actual}\n"
        )
        return {"population": actual, "declared": False, "flag": True}

    if actual != expected_population:
        full = _entity_list(shape, entity, geometry_type)
        avail = [_describe(c, entity) for c in full[:25]]
        diagnosis = {
            "reason": REASON_POPULATION_DRIFT,
            "expected_population": expected_population, "actual_population": actual,
            "delta": actual - expected_population,
            "entity": entity, "geometry_type": geometry_type,
            "candidates_sample": avail, "history": context or {}, "op_id": op_id,
        }
        # PATTERN-MEDVETEN COUNT (selektor_wall_v1, mission-krav 2) -- se _detect_pattern_multiple;
        # a proposal in the diagnosis, never a silent auto-correction of expected_population.
        pattern = _detect_pattern_multiple(full, entity, expected_population)
        if pattern is not None:
            diagnosis["pattern_hypothesis"] = pattern
        raise RefusalError(expected_population, actual, avail, reason=REASON_POPULATION_DRIFT,
                            diagnosis=diagnosis)
    return {"population": actual, "declared": True, "flag": False}


def drain_population_warnings() -> list[dict]:
    """Empties and returns the process-local fail-closed flag log, one entry per undeclared ALL call
    since the last drain, so a recipe report can state "N of M ALL steps carried no population
    declaration" as a number instead of passing silently.
    """
    out = list(UNDECLARED_POPULATION_LOG)
    UNDECLARED_POPULATION_LOG.clear()
    return out


# --------------------------------------------------------------------------------- vertex-medveten
# filtrering
def _edge_geom_key(e) -> tuple:
    """Geometric identity key for an edge, robust across several independent .edges() calls on the same
    shape (build123d gives no stable Python object identity between two separate .edges() calls, so an
    id() comparison between selector_match's candidate list and a standalone _entity_list would never
    match even for the same geometric edge). Endpoints, length and geometry type are enough to tell
    edges apart in this module's use.
    """
    try:
        length = round(float(e.length), 6)
    except Exception:
        length = None
    try:
        gt = e.geom_type.name
    except Exception:
        gt = None
    return (_edge_endpoints(e), length, gt)


def _edge_endpoints(e) -> tuple | None:
    """Rounded endpoint coordinates ((x,y,z),(x,y,z)) for an edge, used to identify a SHARED VERTEX
    between two edges (compared on rounded coordinates; no topological vertex identity is needed).
    Fail-safe: position_at() can occasionally fail on a degenerate edge, in which case None is returned
    and the edge takes no part in the vertex-neighbourhood analysis.
    """
    try:
        p0 = tuple(round(v, 6) for v in e.position_at(0.0))
        p1 = tuple(round(v, 6) for v in e.position_at(1.0))
        return (p0, p1)
    except Exception:
        return None


def vertex_aware_safe_edges(shape, entity: str, blend_radius_mm: float, k: float = 2.0, *,
                             geometry_type: str | None = None, include_edges: bool = True,
                             candidate_pool: list | None = None) -> dict:
    """Measures (never raises) the vertex-aware safe edge set for a prospective fillet or chamfer of
    blend_radius_mm (a radius or a distance). An edge is EXCLUDED if (a) its own length < k *
    blend_radius_mm, or (b) it shares an endpoint with an edge that is (a). That is the measured
    mechanism: OCC's blend-offset conflict propagates through shared corners, not only through the
    chosen edge's own length. "Short" in (a) is measured over the WHOLE shape's edge population (all
    geometry types -- a short CIRCLE edge can poison an adjacent LINE edge just as well), but the safe
    subset returned is limited to candidate_pool when one is given (the same Edge OBJECTS a calling
    selector_match has already filtered, so identity is preserved and the link to that call's candidate
    list is not lost). candidate_pool=None falls back to _entity_list(shape, entity, geometry_type). k is
    measured by the calling harness. include_edges=False leaves out the heavy Edge lists and keeps the
    return JSON-friendly for a report.
    """
    all_edges = _entity_list(shape, entity, None)  # ALLA kanter (obeaktat geometry_type) -- kortedetektionen
    pool = candidate_pool if candidate_pool is not None else _entity_list(shape, entity, geometry_type)
    threshold = float(k) * float(blend_radius_mm)

    own_short: list = []
    for e in all_edges:
        try:
            length = float(e.length)
        except Exception:
            length = 0.0
        if length < threshold:
            own_short.append(e)

    short_vertex_points: set = set()
    for e in own_short:
        ep = _edge_endpoints(e)
        if ep is not None:
            short_vertex_points.add(ep[0])
            short_vertex_points.add(ep[1])
    own_short_keys = {_edge_geom_key(e) for e in own_short}

    vertex_adjacent_excluded: list = []
    own_short_in_pool: list = []
    safe: list = []
    for e in pool:
        if _edge_geom_key(e) in own_short_keys:
            own_short_in_pool.append(e)
            continue
        ep = _edge_endpoints(e)
        if ep is not None and (ep[0] in short_vertex_points or ep[1] in short_vertex_points):
            vertex_adjacent_excluded.append(e)
        else:
            safe.append(e)

    out = {
        "entity": entity, "geometry_type": geometry_type, "k": float(k),
        "blend_radius_mm": float(blend_radius_mm), "threshold_mm": threshold,
        "n_total": len(pool),
        "n_excluded_own_short": len(own_short_in_pool),
        "n_excluded_vertex_adjacent": len(vertex_adjacent_excluded),
        "n_safe": len(safe),
        "excluded_own_short_descr": [_describe(c, entity) for c in own_short_in_pool[:25]],
        "excluded_vertex_adjacent_descr": [_describe(c, entity) for c in vertex_adjacent_excluded[:25]],
    }
    if include_edges:
        out["safe_edges"] = safe
    return out


def check_vertex_safe_population(shape, entity: str, blend_radius_mm: float, k: float = 2.0, *,
                                  geometry_type: str | None = None, op_id: str | None = None,
                                  context: dict | None = None) -> dict:
    """Execution-time gate for a fillet/chamfer candidate set, vertex-aware (see
    vertex_aware_safe_edges). n_safe == 0 (no edge in the whole candidate set is safe after the
    vertex-neighbourhood exclusion) raises a structured RefusalError(reason=SCALE_DEGENERATE) naming the
    excluded edges, never a bare OCC ValueError from inside bd.fillet/bd.chamfer. n_safe > 0 returns the
    measurement, including 'safe_edges', the safe subset a caller can build a targeted
    selector={'entity':..., 'expected_count': n_safe, ...} on.
    """
    meas = vertex_aware_safe_edges(shape, entity, blend_radius_mm, k, geometry_type=geometry_type,
                                    include_edges=True)
    if meas["n_safe"] == 0:
        avail = meas["excluded_own_short_descr"][:15] + meas["excluded_vertex_adjacent_descr"][:10]
        diagnosis = {
            "reason": REASON_SCALE_DEGENERATE,
            "k": meas["k"], "blend_radius_mm": meas["blend_radius_mm"], "threshold_mm": meas["threshold_mm"],
            "n_total": meas["n_total"], "n_excluded_own_short": meas["n_excluded_own_short"],
            "n_excluded_vertex_adjacent": meas["n_excluded_vertex_adjacent"], "n_safe": 0,
            "entity": entity, "geometry_type": geometry_type,
            "excluded_own_short_descr": meas["excluded_own_short_descr"],
            "excluded_vertex_adjacent_descr": meas["excluded_vertex_adjacent_descr"],
            "history": context or {}, "op_id": op_id,
        }
        raise RefusalError(1, 0, avail, reason=REASON_SCALE_DEGENERATE, diagnosis=diagnosis)
    return meas


def sweep_vertex_safe_k(shape, entity: str, blend_radius_mm: float, k_values: list, *,
                         geometry_type: str | None = None) -> list[dict]:
    """Survival-curve helper: measures n_safe (and the two exclusion classes) for EVERY k in k_values on
    the same shape, so a caller can find the smallest k that still leaves a non-empty safe edge set
    without rebuilding the geometry per k. JSON-friendly (include_edges=False).
    """
    rows = []
    for k in k_values:
        m = vertex_aware_safe_edges(shape, entity, blend_radius_mm, k, geometry_type=geometry_type,
                                     include_edges=False)
        rows.append({"k": m["k"], "n_total": m["n_total"], "n_excluded_own_short": m["n_excluded_own_short"],
                     "n_excluded_vertex_adjacent": m["n_excluded_vertex_adjacent"], "n_safe": m["n_safe"]})
    return rows


# =================================================================================== SELFTEST
def _make_box_two_holes(extra_hole_first: bool):
    """AT3c-fixturen: Box(100,60,30) med 2 genomgaende cylinderhal (r=5 @ x=-35, r=8 @ x=22).
    extra_hole_first=True infogar ETT TREDJE hal (r=3, x=0) FORE de tva -- omnumrerar faces()."""
    from build123d import Align, BuildPart, Box, Cylinder, Locations, Mode

    with BuildPart() as bp:
        Box(100, 60, 30)
        if extra_hole_first:
            with Locations((0, 0, 0)):
                Cylinder(3, 40, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        with Locations((-30, -10, 0)):
            Cylinder(5, 40, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        with Locations((30, 10, 0)):
            Cylinder(8, 40, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.CENTER))
    return bp.part


def _at3a() -> dict:
    """okand nyckel => ValueError."""
    ok = False
    try:
        _validate_selector({"entity": "face", "expected_count": 1, "bogus_key": 123})
    except ValueError:
        ok = True
    return {"id": "AT3a", "desc": "okand nyckel => ValueError", "pass": ok}


def _at3b() -> dict:
    """expected_count-refusal pa BADA hallen: 0-traff OCH over-traff."""
    box = _make_box_two_holes(extra_hole_first=False)
    zero_hit_refused = False
    zero_actual = None
    try:
        selector_match(box, {"entity": "face", "geometry_type": "CYLINDER", "radius": 99.0, "expected_count": 1})
    except RefusalError as e:
        zero_hit_refused = e.actual_count == 0
        zero_actual = e.actual_count

    over_hit_refused = False
    over_actual = None
    try:
        # the box still has 4 planar faces untouched by the holes (top/bottom/2 ends), so >1 is guaranteed
        selector_match(box, {"entity": "face", "geometry_type": "PLANE", "expected_count": 1})
    except RefusalError as e:
        over_hit_refused = e.actual_count > 1
        over_actual = e.actual_count

    return {
        "id": "AT3b",
        "desc": "expected_count-refusal 0-traff OCH over-traff (fail-closed bada hallen)",
        "pass": zero_hit_refused and over_hit_refused,
        "zero_hit_actual_count": zero_actual,
        "over_hit_actual_count": over_actual,
    }


def _at3c() -> dict:
    """DECISIVE: topology-change replay (the K33 index class). An extra hole inserted IN THE MIDDLE of
    the sequence -> faces() indices shift (MACHINE-verified below); index-based selection would have
    silently addressed the WRONG cylindrical face (same geom_type -- no type mismatch warns).
    selector_match(radius=...) must hit the RIGHT face in BOTH versions."""
    v1 = _make_box_two_holes(extra_hole_first=False)
    v2 = _make_box_two_holes(extra_hole_first=True)

    def cyl_faces(s):
        return [(i, round(f.radius, 3), round(f.center().X, 3), round(f.center().Y, 3))
                for i, f in enumerate(s.faces()) if f.geom_type.name == "CYLINDER"]

    v1_cyl = cyl_faces(v1)
    v2_cyl = cyl_faces(v2)

    # MACHINE-verify that the topology ACTUALLY shifted (otherwise the test is not sharp)
    n_shift = len(v2.faces()) != len(v1.faces())
    v1_r8_idx = next((i for i, r, x, y in v1_cyl if abs(r - 8.0) < 1e-6), None)
    v2_r8_idx = next((i for i, r, x, y in v2_cyl if abs(r - 8.0) < 1e-6), None)
    index_shifted = (v1_r8_idx is not None and v2_r8_idx is not None and v1_r8_idx != v2_r8_idx)
    # index-baserat val (naiv gammal-stil): faces()[v1_r8_idx] i BADA versionerna
    v1_naive = v1.faces()[v1_r8_idx]
    v2_naive = v2.faces()[v1_r8_idx]  # the index hard-coded from v1 -- the old failure class
    naive_silently_wrong = (
        v1_naive.geom_type.name == "CYLINDER" and v2_naive.geom_type.name == "CYLINDER"
        and abs(v1_naive.radius - 8.0) < 1e-6 and abs(v2_naive.radius - 8.0) > 1e-6
    )

    # selector_match: geometric (radius), independent of index -- must hit the RIGHT face in BOTH
    sel = {"entity": "face", "geometry_type": "CYLINDER", "radius": 8.0, "expected_count": 1}
    m1 = selector_match(v1, sel)[0]
    m2 = selector_match(v2, sel)[0]
    # the reference is the independently measured (x,y) of the v1 cylinder from cyl_faces() above,
    # compared against both the v1 and the v2 hit: the same real face must be found in both, whatever
    # the faces() index shift
    _, _, facit_x, facit_y = next(t for t in v1_cyl if abs(t[1] - 8.0) < 1e-6)
    selector_correct_both = (
        abs(m1.center().X - facit_x) < 1e-3 and abs(m1.center().Y - facit_y) < 1e-3
        and abs(m2.center().X - facit_x) < 1e-3 and abs(m2.center().Y - facit_y) < 1e-3
    )

    return {
        "id": "AT3c",
        "desc": "topology-change replay: the index silently picks the WRONG face (same geom_type), the selector the RIGHT one in both",
        "pass": bool(n_shift and index_shifted and naive_silently_wrong and selector_correct_both),
        "v1_n_faces": len(v1.faces()), "v2_n_faces": len(v2.faces()),
        "v1_cylinders_idx_r_x_y": v1_cyl, "v2_cylinders_idx_r_x_y": v2_cyl,
        "naive_index_used": v1_r8_idx,
        "naive_v2_result_radius": round(v2_naive.radius, 3) if _safe_radius(v2_naive) is not None else None,
        "naive_silently_wrong_face": naive_silently_wrong,
        "selector_v1_hit_center": [round(v, 3) for v in tuple(m1.center())],
        "selector_v2_hit_center": [round(v, 3) for v in tuple(m2.center())],
        "selector_correct_both": selector_correct_both,
        "anchor_extern": (
            "the same mechanism has been measured independently at assembly scale: removing one "
            "solid from an assembly shifts every later solid index by -2, so an index-bound choice "
            "silently re-addresses a different part after a topology change."
        ),
    }


def _integration_test() -> dict:
    """The same selector must hit the same face through a standalone STEP file and through a raw OCC
    shape wrapped with wrap_topods(), which is how a warm in-memory scene hands geometry over."""
    import os
    import tempfile

    from build123d import export_step, import_step

    part = _make_box_two_holes(extra_hole_first=False)
    with tempfile.TemporaryDirectory() as tmp:
        step_path = os.path.join(tmp, "selector_integration_v1.step")
        export_step(part, step_path)

        # A) standalone STEP
        standalone = import_step(step_path)
        standalone_solid = standalone.solids()[0]
        sel = {"entity": "face", "geometry_type": "CYLINDER", "radius": 8.0, "radius_tol_mm": 0.01,
               "expected_count": 1}
        a_hit = selector_match(standalone_solid, sel)[0]

        # B) raw OCC shape -> wrap_topods() -> the same selector_match call
        topods = standalone_solid.wrapped
        wrapped = wrap_topods(topods)
        b_hit = selector_match(wrapped, sel)[0]

    round_trip_identical = (
        abs(a_hit.center().X - b_hit.center().X) < 1e-6
        and abs(a_hit.center().Y - b_hit.center().Y) < 1e-6
        and abs(a_hit.center().Z - b_hit.center().Z) < 1e-6
        and abs(a_hit.radius - b_hit.radius) < 1e-6
    )
    return {
        "id": "INTEGRATION", "desc": "standalone STEP + raw-shape bridge (wrap_topods) same face",
        "pass": bool(round_trip_identical),
        "standalone_center": [round(v, 6) for v in tuple(a_hit.center())],
        "warm_scene_center": [round(v, 6) for v in tuple(b_hit.center())],
    }


def _at4_taxonomy() -> dict:
    """Planted-fault test per taxonomy class: inject a geometry that MUST get the right class. A
    classifier without a red planted-fault case per class is unpublished.
    """
    box = _make_box_two_holes(extra_hole_first=False)

    # REFERENCE_LOST: 0 hits (no radius-99 face exists).
    ref_lost = None
    try:
        selector_match(box, {"entity": "face", "geometry_type": "CYLINDER", "radius": 99.0, "expected_count": 1})
    except RefusalError as e:
        ref_lost = e.reason

    # AMBIGUOUS_NEAREST: near_point exakt mitt mellan tva likvarda CYLINDER-ytor -> tie-band vagrar.
    ambiguous = None
    try:
        selector_match(box, {"entity": "face", "geometry_type": "CYLINDER", "near_point": [0.0, 0.0, 0.0],
                              "expected_count": 1})
    except RefusalError as e:
        ambiguous = e.reason

    # SCALE_DEGENERATE: a radius selector with a tolerance so coarse that both hole faces (r=5, r=8)
    # match -- the scale cannot separate the candidates.
    scale_degen = None
    try:
        selector_match(box, {"entity": "face", "geometry_type": "CYLINDER", "radius": 6.5,
                              "radius_tol_mm": 2.0, "expected_count": 1})
    except RefusalError as e:
        scale_degen = e.reason

    # POPULATION_DRIFT: bar geometry_type-only-selektor (ingen near_point/radius) over-traffar --
    # boxen har 4 PLANE-ytor kvar (topp/botten/2 andar), expected_count=1 -> kandidatmangdens
    # SIZE (4) is what drove the refusal; no single reference was lost.
    pop_drift = None
    try:
        selector_match(box, {"entity": "face", "geometry_type": "PLANE", "expected_count": 1})
    except RefusalError as e:
        pop_drift = e.reason

    ok = (ref_lost == REASON_REFERENCE_LOST and ambiguous == REASON_AMBIGUOUS_NEAREST
          and scale_degen == REASON_SCALE_DEGENERATE and pop_drift == REASON_POPULATION_DRIFT)
    return {
        "id": "AT4", "desc": "taxonomy case evidence: 4/4 classes injected and classified CORRECTLY",
        "pass": bool(ok),
        "REFERENCE_LOST_measured": ref_lost, "AMBIGUOUS_NEAREST_measured": ambiguous,
        "SCALE_DEGENERATE_measured": scale_degen, "POPULATION_DRIFT_measured": pop_drift,
    }


def _at5_all_population_guard() -> dict:
    """The ALL-population guard. (a) declared and drifted -> RefusalError POPULATION_DRIFT with the
    before/after counts and the delta, never silent. (b) undeclared -> no raise (backward compatible)
    but a fail-closed flag is logged machine-readably (drain_population_warnings). (c) declared and not
    drifted -> passes without a flag, so the guard is not over-sensitive.
    """
    drain_population_warnings()  # clear before measuring so this check's count is clean
    box_before = _make_box_two_holes(extra_hole_first=False)
    declared = record_expected_population(box_before, "edge")  # MATT, ej gissat
    box_after = _make_box_two_holes(extra_hole_first=True)  # tredje halet driftar kantpopulationen

    drift_refused = False
    drift_payload = None
    try:
        check_all_population(box_after, "edge", declared["expected_population"], op_id="AT5_drift_case")
    except RefusalError as e:
        drift_refused = (
            e.reason == REASON_POPULATION_DRIFT
            and e.diagnosis["expected_population"] == declared["expected_population"]
            and e.diagnosis["actual_population"] == len(list(box_after.edges()))
            and e.diagnosis["delta"] == len(list(box_after.edges())) - declared["expected_population"]
        )
        drift_payload = {"expected": e.diagnosis["expected_population"],
                          "actual": e.diagnosis["actual_population"], "delta": e.diagnosis["delta"]}

    undeclared_no_raise = True
    undeclared_flagged = False
    try:
        res = check_all_population(box_after, "edge", None, op_id="AT5_undeclared_case")
        undeclared_no_raise = (res["declared"] is False and res["flag"] is True)
    except RefusalError:
        undeclared_no_raise = False
    warnings = drain_population_warnings()
    undeclared_flagged = any(w["op_id"] == "AT5_undeclared_case" for w in warnings)

    no_drift_res = check_all_population(box_before, "edge", declared["expected_population"],
                                         op_id="AT5_stable_case")
    no_drift_ok = no_drift_res["declared"] is True and no_drift_res["flag"] is False

    ok = drift_refused and undeclared_no_raise and undeclared_flagged and no_drift_ok
    return {
        "id": "AT5", "desc": "ALL-populationsvakten: driftad->RefusalError, odeklarerad->flagga ej vagran, stabil->tyst pass",
        "pass": bool(ok),
        "drift_refused": drift_refused, "drift_payload": drift_payload,
        "undeclared_no_raise": undeclared_no_raise, "undeclared_flagged": undeclared_flagged,
        "no_drift_ok": no_drift_ok,
    }


def _make_vertex_adjacency_fixture():
    """AT6/AT7 fixture: Box(50,50,50) with ONE edge chamfered by a small distance (0.4 mm). OCC replaces
    that edge with a NEW, measurably short triangle edge (length about 0.4*sqrt(2) = 0.566 mm) and
    leaves the two adjacent 50 mm edges alive but with a new endpoint at the chamfered edge. A
    length-only selector would keep those two 50 mm neighbours, since they are themselves long; that is
    exactly the measured mechanism (the fraction of selected edges adjacent to a short edge is 1.0),
    because they share a NEW endpoint with the short chamfer edge. vertex_aware_safe_edges() must
    exclude them, while the roughly nine edges that do not touch the chamfered corner stay safe.
    """
    from build123d import Box, BuildPart

    with BuildPart() as bp:
        Box(50, 50, 50)
    solid = bp.part
    edges = sorted(solid.edges(), key=lambda e: (round(e.center().X, 3), round(e.center().Y, 3), round(e.center().Z, 3)))
    target_edge = edges[0]
    import build123d as bd
    chamfered = bd.chamfer([target_edge], length=0.4)
    return chamfered


def _at6_vertex_aware_filter() -> dict:
    """MEASURED case evidence: vertex_aware_safe_edges excludes BOTH the own-short edge AND its long
    vertex neighbours, with correct counting (the leg2 mechanism reproduced on a minimal, DETERMINISTIC
    fixture instead of the stochastic deep chain -- same phenomenon, cheaper to verify)."""
    shape = _make_vertex_adjacency_fixture()
    total_edges = len(list(shape.edges()))
    # blend_radius_mm=0.25 (the same value that broke in djupkedja_diagnos_v1), k=2.0 -- threshold=0.5mm,
    # just above the new chamfer edge's ~0.566 mm length, to force the own-short class to catch it,
    # and also to test a k at which the neighbours are pushed out as well.
    m_k2 = vertex_aware_safe_edges(shape, "edge", blend_radius_mm=0.25, k=2.0, include_edges=False)
    # k tillrackligt stort for att sjalva chamferkanten (langd ~0.566mm) klassas kort (0.566 < k*0.25
    # kraver k>2.26) -- valj k=3.0 for att TVINGA "egen-kort"-klassen att aven fanga chamferkanten,
    # sa att grann-uteslutningen (vertex-adjacency) far nagot att utesluta.
    m_k3 = vertex_aware_safe_edges(shape, "edge", blend_radius_mm=0.25, k=3.0, include_edges=False)
    own_short_found = m_k3["n_excluded_own_short"] >= 1
    vertex_adjacent_found = m_k3["n_excluded_vertex_adjacent"] >= 2  # de tva 50mm-grannarna vid det chamfrade hornet
    accounting_ok = (m_k3["n_excluded_own_short"] + m_k3["n_excluded_vertex_adjacent"] + m_k3["n_safe"]
                      == m_k3["n_total"] == total_edges)
    safe_shrinks_monotonically = m_k3["n_safe"] <= m_k2["n_safe"]  # storre k => stramare filter, aldrig fler sakra
    return {
        "id": "AT6", "desc": "vertex-medveten filtrering utesluter bade egen-kort OCH vertex-granne (leg2-mekanismen, deterministisk fixtur)",
        "pass": bool(own_short_found and vertex_adjacent_found and accounting_ok and safe_shrinks_monotonically),
        "n_total": total_edges, "k2_result": m_k2, "k3_result": m_k3,
    }


def _at7_check_vertex_safe_population() -> dict:
    """Measured planted-fault test for BOTH paths through check_vertex_safe_population: (a) a case where
    SOME edge is safe returns the measurement without raising, and (b) a case where NO edge is safe (a
    very large k against a small box) raises a structured RefusalError(SCALE_DEGENERATE) that names the
    excluded edges, never a bare OCC ValueError.
    """
    shape = _make_vertex_adjacency_fixture()

    some_safe_ok = False
    n_safe_a = None
    try:
        res = check_vertex_safe_population(shape, "edge", blend_radius_mm=0.25, k=2.0, op_id="AT7_ok_case")
        some_safe_ok = res["n_safe"] > 0 and "safe_edges" in res
        n_safe_a = res["n_safe"]
    except RefusalError:
        some_safe_ok = False

    refuses_when_none_safe = False
    reason_ok = False
    names_edges = False
    try:
        # k=1e6: threshold blir absurt stort -> ALLA kanter klassas egen-korta -> n_safe=0 garanterat.
        check_vertex_safe_population(shape, "edge", blend_radius_mm=0.25, k=1e6, op_id="AT7_refuse_case")
    except RefusalError as e:
        refuses_when_none_safe = True
        reason_ok = e.reason == REASON_SCALE_DEGENERATE
        names_edges = len(e.diagnosis.get("excluded_own_short_descr", [])) > 0

    ok = some_safe_ok and refuses_when_none_safe and reason_ok and names_edges
    return {
        "id": "AT7", "desc": "check_vertex_safe_population: some edge safe -> measurement without raising; none safe -> structured SCALE_DEGENERATE refusal naming the edges",
        "pass": bool(ok),
        "n_safe_ok_case": n_safe_a, "refuses_when_none_safe": refuses_when_none_safe,
        "reason_ok": reason_ok, "names_excluded_edges": names_edges,
    }


def _make_shelled_wall_fixture():
    """AT8 fixture: Box(100,60,40) with the top removed and a 2 mm shell (bd.hollow, the same API the
    executor's shell handler uses). Built directly with build123d so the selftest does not depend on the
    executor. normal=[1,0,0] hits BOTH the true outer wall (center=[50,0,0], normal +X) and the opposite
    wall's INNER face (center=[-48,0,1], normal also +X because it points into the cavity) -- exactly
    the ambiguity the "wall" key disambiguates.
    """
    import build123d as bd
    from build123d import Box, BuildPart

    with BuildPart() as bp:
        Box(100, 60, 40)
    box = bp.part
    top = [f for f in box.faces() if abs(f.normal_at(f.center()).Z - 1.0) < 1e-6][0]
    return box.hollow([top], -2.0, kind=bd.Kind.INTERSECTION)


def _at8_wall_discriminator() -> dict:
    """Planted-fault test on the AT8 fixture: (a) wall='outer' must hit exactly the true outer wall, (b)
    wall='inner' must hit exactly the inner wall's corresponding face, and (c) no wall key at all must
    raise RefusalError(AMBIGUOUS_NEAREST) whose diagnosis["wall_candidates"] names BOTH candidates with
    their measured wall_side -- never a silent first hit.
    """
    shape = _make_shelled_wall_fixture()
    sel_base = {"entity": "face", "normal": [1.0, 0.0, 0.0], "expected_count": 1}

    outer_hit = None
    outer_ok = False
    try:
        outer_hit = selector_match(shape, {**sel_base, "wall": "outer"})[0]
        outer_ok = abs(outer_hit.center().X - 50.0) < 1e-3 and abs(outer_hit.area - 2400.0) < 1e-3
    except RefusalError:
        outer_ok = False

    inner_hit = None
    inner_ok = False
    try:
        inner_hit = selector_match(shape, {**sel_base, "wall": "inner"})[0]
        inner_ok = abs(inner_hit.center().X - (-48.0)) < 1e-3 and abs(inner_hit.area - 2128.0) < 1e-3
    except RefusalError:
        inner_ok = False

    refused_without_key = False
    reason_ok = False
    both_named = False
    sides_measured = None
    try:
        selector_match(shape, sel_base)
        refused_without_key = False
    except RefusalError as e:
        refused_without_key = True
        reason_ok = e.reason == REASON_AMBIGUOUS_NEAREST
        wc = e.diagnosis.get("wall_candidates", [])
        both_named = len(wc) == 2
        sides_measured = sorted(d.get("wall_side") for d in wc)

    ok = outer_ok and inner_ok and refused_without_key and reason_ok and both_named and sides_measured == ["inner", "outer"]
    return {
        "id": "AT8", "desc": "wall discriminator: wall='outer'/'inner' hits the right face; without the key -> AMBIGUOUS_NEAREST naming both",
        "pass": bool(ok),
        "outer_ok": outer_ok, "inner_ok": inner_ok,
        "outer_hit_center": [round(v, 3) for v in tuple(outer_hit.center())] if outer_hit is not None else None,
        "inner_hit_center": [round(v, 3) for v in tuple(inner_hit.center())] if inner_hit is not None else None,
        "refused_without_key": refused_without_key, "reason_ok": reason_ok, "both_candidates_named": both_named,
        "wall_sides_measured": sides_measured,
    }


def _make_pattern_wall_fixture():
    """AT9 fixture: 3 copies of the same Box(100,60,40), moved along Y in constant 150 mm steps (a
    Compound of three moved copies). normal=[1,0,0] hits 3 IDENTICAL +X faces (centres [50,0,0],
    [50,150,0], [50,300,0]), all on the same wall side (this fixture has no cavity), so "wall" cannot
    disambiguate -- which is exactly why this case is POPULATION_DRIFT plus a pattern hypothesis, not
    AMBIGUOUS_NEAREST.
    """
    from build123d import Box, BuildPart, Compound, Location

    with BuildPart() as bp:
        Box(100, 60, 40)
    base = bp.part
    return Compound([base.moved(Location((0, i * 150.0, 0))) for i in range(3)])


def _at9_pattern_aware_count() -> dict:
    """Planted-fault test on the AT9 fixture: an over-hitting normal selector (expected_count=1,
    actual=3) gets no wall disambiguation (all three are on the same side, unlike AT8), but
    diagnosis["pattern_hypothesis"] must propose the observed multiple (3) and the measured spacing
    (150 mm along Y) -- a diagnosis, never a silent auto-correction of expected_count, so the reason
    stays POPULATION_DRIFT rather than AMBIGUOUS_NEAREST.
    """
    shape = _make_pattern_wall_fixture()
    sel = {"entity": "face", "normal": [1.0, 0.0, 0.0], "expected_count": 1}

    reason = None
    hyp = None
    refused = False
    try:
        selector_match(shape, sel)
    except RefusalError as e:
        refused = True
        reason = e.reason
        hyp = e.diagnosis.get("pattern_hypothesis")

    reason_ok = reason == REASON_POPULATION_DRIFT
    hyp_ok = bool(
        hyp is not None and hyp.get("observed_count") == 3 and hyp.get("expected_count") == 1
        and hyp.get("observed_multiple") == 3 and abs(hyp.get("spacing_mm", 0.0) - 150.0) < 1e-3
        and hyp.get("axis") == "Y"
    )
    ok = refused and reason_ok and hyp_ok
    return {
        "id": "AT9", "desc": "pattern-medveten count-diagnos: 1->3 uppstroms-multipel foreslas (aldrig auto-korrigerad), reason stannar POPULATION_DRIFT",
        "pass": bool(ok),
        "refused": refused, "reason": reason, "reason_ok": reason_ok, "pattern_hypothesis": hyp, "hypothesis_ok": hyp_ok,
    }


def _make_pattern_wall_fixture_axis(direction: tuple, spacing: float = 300.0, n_copies: int = 3):
    """AT10/AT11 fixture: Box(200,120,80), shelled 5 mm with the top face removed, patterned
    `n_copies` times along `direction` with `spacing`. Built directly with build123d, so the fixture
    stands on its own without going through the op-list executor, the same pattern as _at8/_at9."""
    import build123d as bd
    from build123d import Box, BuildPart

    with BuildPart() as bp:
        Box(200, 120, 80)
    box = bp.part
    top = [f for f in box.faces() if abs(f.normal_at(f.center()).Z - 1.0) < 1e-6][0]
    shelled = box.hollow([top], -5.0, kind=bd.Kind.INTERSECTION)
    dx, dy, dz = direction
    return bd.Compound([
        shelled.moved(bd.Location((dx * spacing * i, dy * spacing * i, dz * spacing * i)))
        for i in range(n_copies)
    ])


def _at10_wall_pattern_same_axis_fix() -> dict:
    """Planted-fault test for the wall miscount under a pattern: 3 copies patterned ALONG THE SAME
    AXIS as the probed normal (X), spacing=300 against a width of 200, so there is a 100 mm gap.
    Before the fix this measured outer=1/inner=5, one outer face misclassified as inner because the
    ray crossed into the next copy's wall. After the fix both wall='outer' and wall='inner' must give
    EXACTLY 3 hits each (one per copy), and the hits' centre X must correspond to each copy's OWN
    faces, never a neighbouring copy's.
    """
    shape = _make_pattern_wall_fixture_axis((1.0, 0.0, 0.0))
    sel_base = {"entity": "face", "normal": [1.0, 0.0, 0.0]}
    expected_outer_x = [100.0, 400.0, 700.0]
    expected_inner_x = [-95.0, 205.0, 505.0]

    outer_n = None
    outer_x_ok = False
    try:
        outer = selector_match(shape, {**sel_base, "wall": "outer", "expected_count": 3})
        outer_n = len(outer)
        outer_x = sorted(round(f.center().X, 3) for f in outer)
        outer_x_ok = all(abs(a - b) < 1e-3 for a, b in zip(outer_x, expected_outer_x))
    except RefusalError as e:
        outer_n = e.actual_count
        outer_x = None

    inner_n = None
    inner_x_ok = False
    try:
        inner = selector_match(shape, {**sel_base, "wall": "inner", "expected_count": 3})
        inner_n = len(inner)
        inner_x = sorted(round(f.center().X, 3) for f in inner)
        inner_x_ok = all(abs(a - b) < 1e-3 for a, b in zip(inner_x, expected_inner_x))
    except RefusalError as e:
        inner_n = e.actual_count
        inner_x = None

    ok = outer_n == 3 and inner_n == 3 and outer_x_ok and inner_x_ok
    return {
        "id": "AT10", "desc": "wall-diskriminator OROBUST mot samlinjart pattern FORE fixet (1/5) -- MASTE ge 3/3 EFTER (egen-solid ray-cap)",
        "pass": bool(ok),
        "outer_n": outer_n, "inner_n": inner_n,
        "outer_x_ok": outer_x_ok, "inner_x_ok": inner_x_ok,
        "outer_centers_x": outer_x if outer_x_ok is not None else None,
        "inner_centers_x": inner_x if inner_x_ok is not None else None,
    }


def _at11_wall_pattern_orthogonal_regression() -> dict:
    """Regression gate: 3 copies patterned ORTHOGONALLY to the probed normal (along Y while the normal
    is X). This case was never broken (outer=3/inner=3 before the fix), and the own-solid cap must not
    change it, since every copy's extent along X is identical whatever its Y position.
    """
    shape = _make_pattern_wall_fixture_axis((0.0, 1.0, 0.0))
    sel_base = {"entity": "face", "normal": [1.0, 0.0, 0.0]}

    outer_n = inner_n = None
    try:
        selector_match(shape, {**sel_base, "wall": "outer", "expected_count": 3})
        outer_n = 3
    except RefusalError as e:
        outer_n = e.actual_count
    try:
        selector_match(shape, {**sel_base, "wall": "inner", "expected_count": 3})
        inner_n = 3
    except RefusalError as e:
        inner_n = e.actual_count

    ok = outer_n == 3 and inner_n == 3
    return {
        "id": "AT11", "desc": "regression: ortogonalt pattern (aldrig trasigt) forblir 3/3 efter egen-solid-capen",
        "pass": bool(ok), "outer_n": outer_n, "inner_n": inner_n,
    }


def selftest() -> dict:
    results = [_at3a(), _at3b(), _at3c(), _integration_test(), _at4_taxonomy(), _at5_all_population_guard(),
               _at6_vertex_aware_filter(), _at7_check_vertex_safe_population(), _at8_wall_discriminator(),
               _at9_pattern_aware_count(), _at10_wall_pattern_same_axis_fix(),
               _at11_wall_pattern_orthogonal_regression()]
    return {"module": "geometri_selektor_v1", "results": results, "all_pass": all(r["pass"] for r in results)}


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        out = selftest()
        print(json.dumps(out, indent=1))
        sys.exit(0 if out["all_pass"] else 1)
    print(__doc__)
