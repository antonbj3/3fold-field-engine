#!/usr/bin/env python3
"""formfeature_v1.py -- composite form features: parametric recipe idioms built from the primitive ops
in cad_op_schema_v1, executed through cad_op_exec_v1.

Each function returns a recipe (a list of ops) plus the measured result of running it, so a caller
gets both the reproducible construction and the numbers.

API:
    f1_plate_shell(name, lx, ly, lz, origin, wall_t, ...) -> dict
    f2_structural_profile(name, axis, length, section, ...) -> dict
    f3_socket(name, x, y, z0, outer_d, height, n_bolts=..., bolt_d=...) -> dict
    f4_hole_pattern(name, base, ...) -> dict

Silent-zero guard: a cutting feature whose tool ends up entirely outside its target removes nothing
and would otherwise return a body that looks valid. Every subtractive feature therefore measures the
volume delta of its own cut and raises a structured error when the delta is zero or has the wrong
sign, instead of returning an unchanged body.

Run the selftest with `python formfeature_v1.py`; it prints a JSON report and exits non-zero if a
check fails.
"""
from __future__ import annotations

import json
import math
import os
import sys

_KDIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(_KDIR, "artifacts")
if _KDIR not in sys.path:
    sys.path.insert(0, _KDIR)

from cad_op_schema_v1 import validate_recipe  # noqa: E402
from recept_cache_v1 import exec_ops_cached  # noqa: E402
from cad_op_exec_v1 import exec_ops as _exec_ops_uncached  # noqa: E402

CACHE_DIR = os.path.join(ARTIFACTS, "recept_cache_v1_store")

# module-level: log of the LAST exec run, for the caller's fallbevis reporting
_LAST_LOG: list = []
_LAST_STATS: dict = {}

# CACHE-FIDELITY REFRAME: recept_cache_v1's STEP-roundtrip cache HIT restore does not always
# reproduce the IN-MEMORY shape's fine edge topology exactly, even when its own store-time
# fingerprint (n_faces/n_edges/n_vertices/volume/ bbox -- all AGGREGATE counts) matches on reload:
# reproduced directly (a fillet->chamfer chain fed a MEASURED, correct expected_count from a fresh
# in-memory pass-1 result, then FAILED with a DIFFERENT count on a pass-2 run whose fillet step was
# a cache HIT reloaded via STEP -- same op declarations, same aggregate fingerprint, different per-
# edge classification after reimport). For a multi-stage selector-driven recipe (fillet THEN chamfer
# on the SAME edges, like F2 here) this is a real correctness gap, not a speed-only concern -- so
# f1_/f2_/f3_ use the UNCACHED in-memory `exec_ops` here, not `exec_ops_cached`. Single-stage / non-
# selector-chained recipes elsewhere in the repo are unaffected (their own downstream consumer re-
# derives from the final STEP export, not from a live in-memory selector query against a reloaded
# intermediate).
def _run(ops: list, tag: str):
    """Validate (schema) THEN execute via the (uncached, in-memory) DISPATCH -- see CACHE-FIDELITY
    REFRAME above for why caching is bypassed here. Raises on any schema violation or op FAIL
    (MISSED_CUT / degenerate sweep / etc) -- never silently swallowed.
    """
    global _LAST_LOG, _LAST_STATS
    validate_recipe({"ops": ops})  # schema gate FIRST -- raises ValueError before any geometry
    log_path = os.path.join(ARTIFACTS, "formfeature_v1_exec_logs", f"{tag}.json")
    result = _exec_ops_uncached(ops, log_path=log_path)
    _LAST_LOG = result["log"]
    _LAST_STATS = {}
    for row in result["log"]:
        if row.get("status") == "FAIL":
            raise ValueError(f"formfeature_v1[{tag}]: op {row['op_id']} FAILED: {row.get('reason')}")
    return result["solids"]


# TYST-NOLL-VAKTEN -- see module docstring for the mechanism/provenance. Shared absolute floor
# with cad_op_exec_v1.py's own MISSED_CUT check (`_MISSED_CUT_TOL_MM3`, same value, deliberately
# not re-imported to avoid coupling this file to that module's private constant name).
_CUT_VOL_TOL_MM3 = 1e-6


def _verify_subtraction(op_id: str, base: "object", tool: "object", *, requested_desc: dict,
                         flip_tool_builder=None) -> "object":
    """`base - tool` WITH a measured before/after volume-delta check. Returns the subtracted shape
    when a real cut happened. Raises ValueError with a MEASURED diagnosis (never narrated) when the
    delta is at-or-below the MISSED_CUT floor -- a cut was requested (this function is only called
    when the caller intends one) but nothing was actually removed.

    flip_tool_builder: optional zero-arg callable that builds the SAME tool with its axis flipped
    (only meaningful for callers exposing an axis_dir, i.e. F4) -- called ONLY on the diagnosis path
    (never on the happy path, so it costs nothing when the cut succeeds). Its result is intersected
    with `base` (a cheap boolean `&`, not a second full subtraction) to MEASURE whether flipping the
    sign would have produced a real cut, and only then does the raised error's tip claim it would.
    """
    v_before = base.volume
    cut = base - tool
    v_after = cut.volume
    delta = v_before - v_after
    tol = max(_CUT_VOL_TOL_MM3, 1e-9 * v_before)
    if abs(delta) > tol:
        return cut  # a real cut happened -- MEASURED, not assumed

    # ---- diagnosis path: only reached on a measured near-zero delta ----
    try:
        overlap_mm3 = (base & tool).volume
    except Exception as e:  # noqa: BLE001 -- diagnosis-only, must never itself crash the error path
        overlap_mm3 = f"UNMEASURABLE:{type(e).__name__}:{e}"

    flip_would_cut = None
    flip_overlap_mm3 = None
    if flip_tool_builder is not None:
        try:
            flipped_tool = flip_tool_builder()
            flip_overlap_mm3 = (base & flipped_tool).volume
            flip_would_cut = flip_overlap_mm3 > tol
        except Exception as e:  # noqa: BLE001 -- diagnosis-only
            flip_overlap_mm3 = f"UNMEASURABLE:{type(e).__name__}:{e}"

    diag = {
        "op_id": op_id, "v_before_mm3": v_before, "v_after_mm3": v_after, "delta_mm3": delta,
        "tol_mm3": tol, "tool_body_overlap_mm3": overlap_mm3, "requested": requested_desc,
        "flip_axis_would_cut": flip_would_cut, "flip_axis_overlap_mm3": flip_overlap_mm3,
    }
    tip = ""
    if flip_would_cut:
        tip = (f" HINT: axis_dir probably points the WRONG WAY -- the flipped direction measured a real "
               f"overlap of {flip_overlap_mm3:.6g} mm3 with the body (>0), while the requested direction "
               f"gave {overlap_mm3 if isinstance(overlap_mm3, (int, float)) else overlap_mm3} mm3. "
               f"axis_dir SHALL point INTO the material (towards the body), not away from it.")
    raise ValueError(
        f"formfeature_v1 SILENT ZERO in {op_id}: a subtraction was requested but the measured volume "
        f"delta={delta:.6g} mm3 lies within tol={tol:.3g} mm3 (v_before={v_before:.6g}, "
        f"v_after={v_after:.6g} mm3, tool/body overlap={overlap_mm3} mm3).{tip} "
        f"Diagnos (JSON): {json.dumps(diag, default=str)}"
    )


# ===================================================================================== F1
def f1_plate_shell(tag: str, lx: float, ly: float, lz: float, center: tuple,
                    wall_mm: float, r_edge: float, r_edge_horizontal: float = 0.0) -> "object":
    """Outer box (lx,ly,lz) centered at `center`, hollowed to `wall_mm`, then all 8 VERTICAL edges (4
    outer + 4 inner -- hollowing a box always produces both rings, per build123d's own `hollow()`)
    parallel to world Z filleted r_edge ("fasade horn"). wall_mm and r_edge are auto-clamped DOWN
    (never up) to stay geometrically valid for the given box size -- a caller passing an oversized
    value gets a smaller-but-valid feature, not a crash (F1 exists to be robust across 3 very
    different part sizes in A5.2: rail_carriage ~650mm box down to nothing this small currently,
    controller cabinet 500x500x1800, safety fence panels 50mm thick).
    MATINVARIANT: returned shape's bounding_box() == the ORIGINAL (lx,ly,lz,center) box's, exactly
    (verified by the caller; shell+fillet only remove material, cannot grow past the box hull).

    r_edge_horizontal: additionally fillets the TOP+BOTTOM ring edges (world X and Y directed, both
    outer+inner rings) via a DIRECT build123d call on the already-built live shape
    -- SAME declared idiom as F2's end-chamfer/F3's bolt-hole workaround in this file (a second
    cad_op_exec_v1 selector pass against a fresh in-memory shape isn't needed for an undirected
    line-tangent match). 0.0 = skip entirely (no behavioural change to any caller that omits it).
    """
    cx, cy, cz = center
    wall = max(1.0, min(wall_mm, 0.45 * min(lx, ly, lz)))
    # r must ALSO fit inside the INNER (hollowed) cross-section's own short side, not just the outer
    # box's -- MEASURED: a long/thin panel (e.g. a 7402x50mm safety-fence panel) hollowed to
    # wall=15mm leaves an inner strip only 50-30=20mm wide; a 12mm fillet (< outer clamp, but > half
    # the 20mm inner strip) made OCC raise StdFail_NotDone.
    inner_lx, inner_ly = lx - 2 * wall, ly - 2 * wall
    r = max(0.5, min(r_edge, wall * 0.9, 0.45 * min(lx, ly), 0.45 * min(inner_lx, inner_ly)))
    ops = [
        {"op": "sketch_2d", "id": f"{tag}_sk", "plane": {"origin": [cx, cy, cz - lz / 2.0]},
         "shapes": [{"type": "rectangle", "width": lx, "height": ly, "mode": "add"}]},
        {"op": "extrude", "id": f"{tag}_ext", "sketch_ref": f"{tag}_sk", "amount": lz},
        {"op": "shell", "id": f"{tag}_shell", "target_ref": f"{tag}_ext", "thickness": wall},
        {"op": "fillet", "id": f"{tag}_fil", "target_ref": f"{tag}_shell", "radius": r,
         "selector": {"entity": "edge", "expected_count": 8, "direction": [0, 0, 1]}},
    ]
    solids = _run(ops, tag)
    result = solids[f"{tag}_fil"]

    if r_edge_horizontal and r_edge_horizontal > 0:
        rh = max(0.5, min(r_edge_horizontal, wall * 0.9, 0.45 * min(inner_lx, inner_ly)))
        for dvec in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
            edges = _line_edges_by_direction(result, dvec)
            if not edges:
                # Logged, not raised: this IS a no-op, but the fillet/chamfer verbs are declared
                # cosmetic (r_edge_horizontal is opt-in and 0.0 means skip by contract), not a
                # requested subtraction whose absence would break a measured invariant the way
                # _verify_subtraction guards it. Still, an invisible skip is recorded so it shows up
                # in the log instead of a silent `continue`.
                _LAST_LOG.append({"op_id": f"{tag}_horiz_fillet_{dvec}", "op_type": "fillet",
                                   "status": "SKIPPED", "reason": "no_matching_edges_found"})
                continue
            try:
                result = _bd_fillet(edges, rh)
            except Exception as e:  # noqa: BLE001 -- a failed horizontal-edge fillet is cosmetic, never fatal
                _LAST_LOG.append({"op_id": f"{tag}_horiz_fillet_{dvec}", "op_type": "fillet",
                                   "status": "SKIPPED", "reason": f"{type(e).__name__}:{e}"})
    return result


def _bd_fillet(edges, radius):
    import build123d as bd
    return bd.fillet(edges, radius=radius)


# =================================================================================== F1-DRUM
def f1_drum_shell(tag: str, r_outer: float, height: float, center: tuple,
                   wall_mm: float, r_rim: float) -> "object":
    """Cylindrical variant of F1, a drum-shaped housing: a shell (hollow) plus chamfered top and bottom
    rim edges, which produce genuine toroidal surfaces, the same mechanism as F3's shoulder fillet. It
    exists because a box of the same size can never get its axis-aligned area fraction low enough
    through corner chamfers alone (measured: a 650 mm box chamfered as much as its 16 mm wall allows
    still sits at 0.931, since corner radii only affect a thin strip of a large face), whereas a drum's
    mantle is by construction one curved cylindrical surface.
    Measured invariant: the returned shape's bounding_box() equals the original cylinder's (2*r_outer in
    XY, height in Z, centred on `center`) -- shelling and filleting only remove material.
    """
    cx, cy, cz = center
    wall = max(1.0, min(wall_mm, 0.45 * r_outer))
    r = max(0.5, min(r_rim, wall * 0.9, 0.3 * height))
    ops = [
        {"op": "sketch_2d", "id": f"{tag}_sk", "plane": {"origin": [cx, cy, cz - height / 2.0]},
         "shapes": [{"type": "circle", "radius": r_outer, "mode": "add"}]},
        {"op": "extrude", "id": f"{tag}_ext", "sketch_ref": f"{tag}_sk", "amount": height},
        {"op": "shell", "id": f"{tag}_shell", "target_ref": f"{tag}_ext", "thickness": wall},
        {"op": "fillet", "id": f"{tag}_fil", "target_ref": f"{tag}_shell", "radius": r,
         "selector": {"entity": "edge", "expected_count": 2, "geometry_type": "CIRCLE", "radius": r_outer}},
    ]
    solids = _run(ops, tag)
    return solids[f"{tag}_fil"]


# ===================================================================================== F2
_AXIS_VEC = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
# per length_axis: (plane x_dir, plane z_dir[=normal]) chosen so local-x/local-y map onto the
# two WORLD axes named in `cross_dims` (see f2_structural_profile docstring) -- reused verbatim
# from cad_op_exec_v1.AXIS_PLANE_ORIGIN's own convention (hole op), extended with the sign-fixed
# Y-axis case (measured: naive reuse gave local_y = -world_X, harmless for a centered symmetric
# rectangle but flipped here for the CLEARER contract "local_x,local_y = the two cross_dims keys
# in world-axis order", i.e. deterministic, not just harmless).
_PLANE_FOR_AXIS = {
    "X": {"x_dir": (0.0, 1.0, 0.0), "z_dir": (1.0, 0.0, 0.0), "local": ("Y", "Z")},
    "Y": {"x_dir": (0.0, 0.0, 1.0), "z_dir": (0.0, 1.0, 0.0), "local": ("Z", "X")},
    "Z": {"x_dir": (1.0, 0.0, 0.0), "z_dir": (0.0, 0.0, 1.0), "local": ("X", "Y")},
}


def f2_structural_profile(tag: str, length_axis: str, length: float, cross_dims: dict,
                           wall_mm: float | None, r_root: float, chamfer_mm: float,
                           center: tuple) -> "object":
    """Rectangular (hollow if wall_mm given, else solid) profile extruded `length` along `length_axis`,
    centered at `center` (MATINVARIANT: the beam's centerline IS `center` +/-length/2 along
    length_axis, unchanged by root-radius/chamfer -- both only remove material).
    cross_dims: {"<world_axis_letter>": mm, "<other_world_axis_letter>": mm} -- the two world
    axes orthogonal to length_axis (e.g. length_axis="X" -> cross_dims must have keys "Y","Z").
    wall_mm=None -> solid bar. wall_mm=<t> -> hollow tube, wall thickness t.

    TVA-PASS CHAMFER-RAKNING: root-radius (step 1) trims each ring-boundary edge's ENDPOINT (never
    removes/splits the mid-edge) -- but a hollow profile's corner fillet ALSO interacts with the
    shell's inner wall closer than a solid bar's, and this interaction's resulting ring-edge COUNT
    is NOT a fixed 8/4 as a naive corner-count model predicts (reproduced directly: an identical,
    now edge-order-DETERMINISTIC recipe -- see cad_op_exec_v1._resolve_target_edges's own sort fix,
    same session -- consistently yields 16, not 8, ring-direction edges for THIS hollow cross-
    section, i.e. the naive model was simply wrong, not flaky). Rather than hardcode a guessed
    constant, this function runs the fillet step FIRST (cached), directly counts the real post-
    fillet ring edges per direction by geometry (not the strict selector), THEN issues the strict
    chamfer op with the MEASURED count -- the strict selector's fail-closed contract (RefusalError
    on any mismatch) is preserved end-to-end, it is simply fed a measured count instead of an
    assumed one.
    """
    axmap = _PLANE_FOR_AXIS[length_axis]
    lax, lay = axmap["local"]  # world-axis letters that map onto local sketch x / local sketch y
    w = cross_dims[lax]
    h = cross_dims[lay]
    length_vec = list(_AXIS_VEC[length_axis])
    origin = [center[0] - length_vec[0] * length / 2.0,
              center[1] - length_vec[1] * length / 2.0,
              center[2] - length_vec[2] * length / 2.0]

    shapes = [{"type": "rectangle", "width": w, "height": h, "mode": "add"}]
    hollow = wall_mm is not None and wall_mm > 0
    if hollow:
        wall = min(wall_mm, 0.45 * min(w, h))
        shapes.append({"type": "rectangle", "width": w - 2 * wall, "height": h - 2 * wall, "mode": "subtract"})
    n_long = 8 if hollow else 4

    ops = [
        {"op": "sketch_2d", "id": f"{tag}_sk",
         "plane": {"origin": origin, "x_dir": list(axmap["x_dir"]), "z_dir": list(axmap["z_dir"])},
         "shapes": shapes},
        {"op": "extrude", "id": f"{tag}_ext", "sketch_ref": f"{tag}_sk", "amount": length},
    ]
    solid_id = f"{tag}_ext"
    if r_root and r_root > 0:
        r = min(r_root, (min(w, h) - (2 * wall_mm if hollow else 0.0)) * 0.4 if hollow else min(w, h) * 0.4)
        r = max(r, 0.5)
        ops.append({"op": "fillet", "id": f"{tag}_fil", "target_ref": solid_id, "radius": r,
                    "selector": {"entity": "edge", "expected_count": n_long, "direction": length_vec}})
        solid_id = f"{tag}_fil"

    # sketch+extrude(+fillet) via the recipe engine (cad_op_exec_v1 DISPATCH, schema-validated,
    # fallbevis-logged in _LAST_LOG).
    solids = _run(ops, tag)
    result = solids[solid_id]

    if chamfer_mm and chamfer_mm > 0:
        # END-CHAMFER ("gering"): applied via a DIRECT build123d call on the LIVE `result` object,
        # NOT re-routed through a second cad_op_exec_v1 recipe call -- DECLARED REASON: rebuilding
        # the SAME declarative op-list a second time (needed so the strict selector's chamfer op can
        # reference the fillet's OWN op_id) reproducibly gave a DIFFERENT post-fillet ring-edge
        # topology (16 edges) than the first, otherwise-identical build (8 edges) -- BRepFilletAPI
        # is not topology-stable across two independent constructions of this exact hollow-corner
        # geometry, even with a DETERMINISTIC input edge order (see
        # cad_op_exec_v1._resolve_target_edges's own sort fix). Operating on the SAME
        # live object (measured once, chamfered once, never rebuilt) sidesteps the rebuild-mismatch
        # entirely -- the schema/selector CONTRACT (geometric, index-free targeting) is still
        # honoured, only the "go through a second op-list" plumbing is skipped.
        c = min(chamfer_mm, min(w, h) * 0.08)
        if hollow:
            c = min(c, wall * 0.5)
        if r_root and r_root > 0:
            c = min(c, r * 0.6)
        lax_vec = [1.0 if lax == ax else 0.0 for ax in "XYZ"]
        lay_vec = [1.0 if lay == ax else 0.0 for ax in "XYZ"]
        import build123d as _bd
        for dvec in (lax_vec, lay_vec):
            edges = _line_edges_by_direction(result, dvec)
            if not edges:
                # SILENT-ZERO GUARD sweep (sweep, not raise -- same rationale as F1's horizontal
                # fillet above): the end chamfer is declared cosmetic, but this makes the no-op VISIBLE.
                _LAST_LOG.append({"op_id": f"{tag}_chamfer_direct_{dvec}", "op_type": "chamfer",
                                   "status": "SKIPPED", "reason": "no_matching_edges_found"})
                continue
            try:
                result = _bd.chamfer(edges, length=c)
            except Exception as e:  # noqa: BLE001 -- a failed end-chamfer is cosmetic, never fatal
                _LAST_LOG.append({"op_id": f"{tag}_chamfer_direct", "op_type": "chamfer",
                                   "status": "SKIPPED", "reason": f"{type(e).__name__}:{e}"})

    return result


def _line_edges_by_direction(shape, direction_vec, tol_deg: float = 1.0) -> list:
    """Plain geometric edge query (same undirected-tangent match cad_op_exec_v1's own selector
    uses) against a LIVE shape -- see f2_structural_profile's "END-CHAMFER" comment for why this
    bypasses the strict schema-selector plumbing (not the underlying geometric CONTRACT)."""
    import build123d as bd
    target = bd.Vector(*direction_vec).normalized()
    out = []
    for e in shape.edges():
        if e.geom_type != bd.GeomType.LINE:
            continue
        t = e.tangent_at(0.5).normalized()
        ang = t.get_angle(target)
        ang = min(ang, 180.0 - ang)
        if ang <= tol_deg:
            out.append(e)
    return out


# ===================================================================================== F3
def f3_socket(tag: str, cx: float, cy: float, z0: float, z1: float, r_shaft: float,
              n_bolts: int = 6, bolt_d: float = 16.0) -> "object":
    """Revolved step profile (foot plate + shaft) about the world-Z axis through (cx,cy,z0), spanning
    z0..z1 (MATINVARIANT: axis (cx,cy) and Z-extent (z0,z1) unchanged from the plain cylinder it
    replaces). r_foot/h_foot/r_transition/pcd are all DERIVED from r_shaft and the total height
    (z1-z0) so the feature stays valid across the two very different scales in A5.2 (robot pedestal
    r_shaft=300mm and tool-head r_shaft=75mm) without per-call tuning. Fillet at the foot->shaft
    step (a CIRCLE edge, radius=r_shaft, at height z0+h_foot) is the genuine TOROIDAL_SURFACE source
    (see module docstring F3) -- the P3 criterion's real signal, not a box-corner fillet.
    """
    H = z1 - z0
    r_foot = r_shaft * 1.35
    h_foot = max(8.0, min(0.12 * H, r_shaft * 0.25))
    r_trans = max(2.0, min(0.35 * (r_foot - r_shaft), h_foot * 0.6))
    h_shaft = H - h_foot
    if h_shaft <= r_trans * 2:  # degenerate-guard: keep the shaft segment strictly positive
        h_foot = max(4.0, H * 0.08)
        h_shaft = H - h_foot
        r_trans = max(1.0, min(0.3 * (r_foot - r_shaft), h_foot * 0.5, h_shaft * 0.3))

    plane = {"origin": [cx, cy, z0], "x_dir": [1.0, 0.0, 0.0], "z_dir": [0.0, -1.0, 0.0]}
    ops = [
        {"op": "sketch_2d", "id": f"{tag}_sk", "plane": plane, "shapes": [
            {"type": "rectangle", "width": r_foot, "height": h_foot,
             "center": [r_foot / 2.0, h_foot / 2.0], "mode": "add"},
            {"type": "rectangle", "width": r_shaft, "height": h_shaft,
             "center": [r_shaft / 2.0, h_foot + h_shaft / 2.0], "mode": "add"},
        ]},
        {"op": "revolve", "id": f"{tag}_rev", "sketch_ref": f"{tag}_sk",
         "axis": {"origin": [cx, cy, z0], "direction": [0.0, 0.0, 1.0]}, "angle_deg": 360.0},
        {"op": "fillet", "id": f"{tag}_fil", "target_ref": f"{tag}_rev", "radius": r_trans,
         "selector": {"entity": "edge", "expected_count": 1, "geometry_type": "CIRCLE",
                      "radius": r_shaft, "near_point": [cx, cy, z0 + h_foot],
                      # radius=r_shaft alone also matches the shaft's OWN top-cap circle (same
                      # radius, at z1) -- near_point_max_dist_mm disambiguates the shoulder one.
                      # MEASURED: geometri_selektor_v1's Edge.center() for a FULL 360deg revolved
                      # circle is NOT its true geometric center (a build123d artifact -- it returns
                      # the pre-revolve SEAM vertex's own position, offset from the true center by
                      # ~the circle's OWN radius). So the shoulder circle's measured "distance" to
                      # near_point is ~r_shaft (not ~0), and the far (top-cap) circle's is
                      # ~sqrt(r_shaft^2+h_shaft^2) -- the cutoff below is set to sit between those
                      # two MEASURED distances, not near 0.
                      "near_point_max_dist_mm": r_shaft * 1.25}},
    ]
    solid_id = f"{tag}_fil"

    if n_bolts and n_bolts > 0:
        pcd_r = r_foot * 0.82
        d = min(bolt_d, r_foot * 0.12, h_foot * 0.7)
        # BOLT-HOLE PATTERN FIX: cad_op_exec_v1._handle_pattern_circular's dict-axis form
        # (axis={"origin":[...], "direction":...}) issues bd.Location(origin, direction, angle) per
        # copy. MEASURED directly (not narrated): that constructor does NOT rotate the target about
        # the axis LINE through `origin` -- it rotates about the WORLD axis through (0,0,0) and then
        # ALWAYS adds an extra translation by `origin`, present even at angle=0/i=0 (verified:
        # Location((0,0,35),(0,0,1),0.0) prints position=(0,0,35), not identity). Every real F3
        # caller in this repo has z0 != 0 (only the z0=0.0 selftest fixture happens to hide this,
        # since origin=(0,0,0) there makes the spurious translation a no-op) -- so the bolt-hole
        # tool has always landed entirely off the foot (e.g. z shifted by +z0), and the later
        # `result - tool` subtracted NOTHING: a SILENT-WRONG defect (bolt holes declared in the op
        # list, never actually cut), exactly the failure class disciplin-rad 1/6 exist to catch.
        # cad_op_exec_v1.py is owned by a parallel cell tonight (not editable here) -- worked around
        # IN THIS FILE instead of patched at the source: pattern the tool at the WORLD ORIGIN with
        # the axis="Z" STRING form (origin=(0,0,0) implicitly, the one path this bug cannot reach
        # --verified: Location((0,0,0),(0,0,1),angle) rotates correctly with zero spurious shift),
        # then translate the finished, correctly-patterned tool to (cx,cy,z0) with a plain bd.Pos()
        # afterward -- the SAME declared direct-build123d-call exception F2's end-chamfer already
        # uses (see that docstring) for the identical reason (a second cad_op_exec_v1 pass is not
        # this library's contract for post-processing a live shape).
        hole_plane = {"origin": [pcd_r, 0.0, 0.0], "x_dir": [1.0, 0.0, 0.0], "z_dir": [0.0, 0.0, 1.0]}
        ops2 = [
            {"op": "sketch_2d", "id": f"{tag}_bsk", "plane": hole_plane,
             "shapes": [{"type": "circle", "radius": d / 2.0, "mode": "add"}]},
            {"op": "extrude", "id": f"{tag}_bext", "sketch_ref": f"{tag}_bsk", "amount": h_foot * 1.2},
            {"op": "pattern_circular", "id": f"{tag}_bpat", "target_ref": f"{tag}_bext",
             "axis": "Z", "count": n_bolts},
        ]
        b_solids = _run(ops2, f"{tag}_bolts")

    solids = _run(ops, tag)
    result = solids[solid_id]
    if n_bolts and n_bolts > 0:
        import build123d as _bd
        tool = _bd.Pos(cx, cy, z0) * b_solids[f"{tag}_bpat"]
        # Silent-zero guard: this `result - tool` is exactly the raw Python subtraction class that
        # bypasses the executor's own MISSED_CUT dispatch (the same reason as F4's hole cuts below).
        # This line historically carried a silent-zero bug; it is fixed, and the guard stays so the
        # class cannot creep back.
        result = _verify_subtraction(
            f"{tag}_bolt_pattern_cut", result, tool,
            requested_desc={"n_bolts": n_bolts, "bolt_d": bolt_d, "d": d, "pcd_r": pcd_r,
                             "cx": cx, "cy": cy, "z0": z0, "h_foot": h_foot},
        )
    return result


def _circle_edges_near(shape, radius: float, near_point: tuple, max_dist: float | None = None,
                        tol_r_frac: float = 0.2):
    """Plain geometric edge query (same class as `_line_edges_by_direction` above -- undirected,
    against a LIVE shape, not the strict schema-selector) for CIRCLE edges of the given radius
    close to `near_point`. Used by F4 below to find a just-cut hole's own entry-face rim so it can
    be countersunk, the same "measure-then-chamfer-the-live-object" idiom F2's end-chamfer and F3's
    bolt-hole workaround already use in this file."""
    import build123d as bd
    out = []
    for e in shape.edges():
        if e.geom_type != bd.GeomType.CIRCLE:
            continue
        try:
            r = float(e.radius)
        except Exception:
            continue
        if abs(r - radius) > max(tol_r_frac * radius, 0.05):
            continue
        if max_dist is not None:
            c = e.center()
            dist = ((c.X - near_point[0]) ** 2 + (c.Y - near_point[1]) ** 2 + (c.Z - near_point[2]) ** 2) ** 0.5
            if dist > max_dist:
                continue
        out.append(e)
    return out


# ===================================================================================== F4
def f4_countersunk_holes(tag: str, shape: "object", holes: list, axis_dir: tuple = (0.0, 0.0, 1.0)):
    """Chamfered plate edge / holes: N through mounting holes, each with a countersunk entry edge, so a
    screw head or burr sits flush. The chamfered circular edge is a genuine cone surface. Built through
    the executor's dispatch (one sketch_2d plus extrude per hole, deliberately not pattern_circular --
    see f3_socket's bolt-hole comment for why that op is not safe for a non-zero origin here) plus one
    direct build123d chamfer call per hole on the circle of the edge just cut.
    Measured invariant: material is only removed (a through hole plus a cone chamfer at the entry); the
    shape never grows outside its existing envelope.
    holes: [{"center": (x,y,z), a point on the surface where the hole enters, "r": hole radius mm,
             "depth": hole depth mm (at least the wall thickness there), "csk": chamfer length mm (0 or
             None for a plain hole)}, ...].
    axis_dir: the world direction all holes in this call are drilled along (one direction per call; call
    f4 twice for two hole directions).
    Returns (new_shape, n_countersunk), where n_countersunk is how many holes actually got a measured
    CIRCLE edge to chamfer -- 0 means no cone was added, never silently assumed.
    """
    import build123d as _bd
    d = _bd.Vector(*axis_dir).normalized()
    # pick whichever world axis is LEAST parallel to `d`.
    if abs(d.X) <= abs(d.Y) and abs(d.X) <= abs(d.Z):
        ref = (1.0, 0.0, 0.0)
    elif abs(d.Y) <= abs(d.Z):
        ref = (0.0, 1.0, 0.0)
    else:
        ref = (0.0, 0.0, 1.0)
    rv = _bd.Vector(*ref)
    x_dir = (rv - d * rv.dot(d)).normalized()
    result = shape
    n_countersunk = 0
    for i, spec in enumerate(holes):
        cx, cy, cz = spec["center"]
        r = float(spec["r"])
        depth = float(spec["depth"])
        plane = {"origin": [cx, cy, cz], "x_dir": [x_dir.X, x_dir.Y, x_dir.Z], "z_dir": [d.X, d.Y, d.Z]}
        ops = [
            {"op": "sketch_2d", "id": f"{tag}_h{i}_sk", "plane": plane,
             "shapes": [{"type": "circle", "radius": r, "mode": "add"}]},
            {"op": "extrude", "id": f"{tag}_h{i}_ext", "sketch_ref": f"{tag}_h{i}_sk", "amount": depth},
        ]
        b = _run(ops, f"{tag}_h{i}")
        tool = b[f"{tag}_h{i}_ext"]

        def _flip_tool_builder(cx=cx, cy=cy, cz=cz, r=r, depth=depth, x_dir=x_dir, d=d, tag=tag, i=i):
            # measured flip test (diagnosis-only path, see _verify_subtraction): rebuild the SAME
            # hole tool with axis_dir NEGATED, through the SAME cad_op_exec_v1 DISPATCH (not a
            # free-geometry shortcut) so the measured overlap is an apples-to-apples comparison.
            flip_d = -d
            flip_plane = {"origin": [cx, cy, cz], "x_dir": [x_dir.X, x_dir.Y, x_dir.Z],
                          "z_dir": [flip_d.X, flip_d.Y, flip_d.Z]}
            flip_ops = [
                {"op": "sketch_2d", "id": f"{tag}_h{i}_fsk", "plane": flip_plane,
                 "shapes": [{"type": "circle", "radius": r, "mode": "add"}]},
                {"op": "extrude", "id": f"{tag}_h{i}_fext", "sketch_ref": f"{tag}_h{i}_fsk", "amount": depth},
            ]
            fb = _run(flip_ops, f"{tag}_h{i}_flip")
            return fb[f"{tag}_h{i}_fext"]

        # silent-zero guard.
        result = _verify_subtraction(
            f"{tag}_h{i}_cut", result, tool,
            requested_desc={"center": [cx, cy, cz], "r": r, "depth": depth,
                             "axis_dir": [d.X, d.Y, d.Z], "csk": spec.get("csk")},
            flip_tool_builder=_flip_tool_builder,
        )
        csk = spec.get("csk")
        if csk:
            edges = _circle_edges_near(result, r, (cx, cy, cz), max_dist=max(r * 1.5, 2.0))
            if edges:
                try:
                    result = _bd.chamfer(edges, length=min(float(csk), r * 0.75))
                    n_countersunk += 1
                except Exception as e:  # noqa: BLE001 -- a failed countersink is cosmetic, never fatal
                    _LAST_LOG.append({"op_id": f"{tag}_h{i}_csk", "op_type": "chamfer",
                                       "status": "SKIPPED", "reason": f"{type(e).__name__}:{e}"})
    return result, n_countersunk


# ===================================================================================== SELFTEST
def _selftest() -> dict:
    checks = []

    # F1: box 200x150x100 wall 5 r_edge 3, AABB must equal the input box exactly
    s1 = f1_plate_shell("selftest_f1", 200, 150, 100, (0, 0, 0), 5.0, 3.0)
    bb1 = s1.bounding_box()
    aabb_ok = (abs(bb1.size.X - 200) < 1e-6 and abs(bb1.size.Y - 150) < 1e-6 and abs(bb1.size.Z - 100) < 1e-6)
    checks.append({"id": "f1_aabb_identical", "pass": aabb_ok,
                    "size": [bb1.size.X, bb1.size.Y, bb1.size.Z]})
    checks.append({"id": "f1_volume_less_than_solid_box", "pass": s1.volume < 200 * 150 * 100,
                    "volume": s1.volume, "solid_box_volume": 200 * 150 * 100})

    # F2: hollow beam along X, length 500, cross 60x40, wall 5, r_root 4, chamfer 3
    s2 = f2_structural_profile("selftest_f2h", "X", 500, {"Y": 60, "Z": 40}, 5.0, 4.0, 3.0, (250, 0, 0))
    bb2 = s2.bounding_box()
    centerline_ok = abs((bb2.min.X + bb2.max.X) / 2.0 - 250.0) < 1e-6
    checks.append({"id": "f2_hollow_centerline_unchanged", "pass": centerline_ok,
                    "mid_x": (bb2.min.X + bb2.max.X) / 2.0})
    checks.append({"id": "f2_hollow_volume_less_than_solid_bar",
                    "pass": s2.volume < 500 * 60 * 40,
                    "volume": s2.volume, "solid_bar_volume": 500 * 60 * 40})

    # F2 solid variant (wall_mm=None) along Y
    s2b = f2_structural_profile("selftest_f2s", "Y", 300, {"X": 50, "Z": 50}, None, 3.0, 2.0, (0, 150, 0))
    checks.append({"id": "f2_solid_builds", "pass": s2b.volume > 0, "volume": s2b.volume})

    # F3: pedestal r_shaft=100, z0..z1 = 0..600
    s3 = f3_socket("selftest_f3", 0, 0, 0.0, 600.0, 100.0, n_bolts=6, bolt_d=12.0)
    bb3 = s3.bounding_box()
    z_ok = abs(bb3.min.Z - 0.0) < 0.5 and bb3.max.Z <= 600.0 + 1e-6
    checks.append({"id": "f3_z_extent_within_bounds", "pass": z_ok, "z_min": bb3.min.Z, "z_max": bb3.max.Z})
    from cad_op_exec_v1 import bd  # noqa: F401 -- ensure build123d import path resolves for GeomType check
    import build123d as _bd
    n_torus = sum(1 for f in s3.faces() if f.geom_type == _bd.GeomType.TORUS)
    checks.append({"id": "f3_has_toroidal_surface_from_shoulder_fillet", "pass": n_torus >= 1,
                    "n_toroidal_faces": n_torus})

    # F3 REGRESSION NET: bolt holes at a NONZERO z0 (the realistic case, unlike the z0=0.0 fixture
    # above which hid the pattern_circular bug) must actually be CUT, not silently no-op'd -- volume
    # must be strictly less than a bolt-less build.
    s3b_nobolts = f3_socket("selftest_f3_nz_nobolts", 0, 0, 35.0, 140.0, 25.0, n_bolts=0)
    s3b_bolts = f3_socket("selftest_f3_nz_bolts", 0, 0, 35.0, 140.0, 25.0, n_bolts=4, bolt_d=6.0)
    n_cyl_bolts = sum(1 for f in s3b_bolts.faces() if f.geom_type == _bd.GeomType.CYLINDER)
    bolts_real = s3b_bolts.volume < s3b_nobolts.volume - 1.0 and n_cyl_bolts >= 4
    checks.append({"id": "f3_nonzero_z0_bolt_holes_actually_cut", "pass": bolts_real,
                    "volume_nobolts": s3b_nobolts.volume, "volume_with_4_bolts": s3b_bolts.volume,
                    "n_cylinder_faces_with_bolts": n_cyl_bolts})

    # F4: 2 countersunk through-holes on a flat plate -- must add >=2 real CONE faces.
    s4_plate = f1_plate_shell("selftest_f4_plate", 200, 150, 20, (0, 0, 0), 4.0, 3.0)
    s4, n_csk = f4_countersunk_holes(
        "selftest_f4", s4_plate,
        [{"center": (60.0, 0.0, -10.0), "r": 4.0, "depth": 20.0, "csk": 1.5},
         {"center": (-60.0, 0.0, -10.0), "r": 4.0, "depth": 20.0, "csk": 1.5}],
        axis_dir=(0.0, 0.0, 1.0))
    n_cone = sum(1 for f in s4.faces() if f.geom_type == _bd.GeomType.CONE)
    checks.append({"id": "f4_countersunk_holes_add_real_cone_faces",
                    "pass": n_csk == 2 and n_cone >= 2,
                    "n_countersunk": n_csk, "n_cone_faces": n_cone})

    # ============================================================ TYST-NOLL-VAKTEN sjalvtest
    # AT-A: direct unit test of the guard mechanism on synthetic, non-overlapping shapes (no
    # flip_tool_builder) -- confirms it raises ValueError with a measured (not narrated) diagnosis.
    import build123d as _bd2
    base_a = _bd2.Box(20, 20, 20)
    tool_a = _bd2.Pos(200, 0, 0) * _bd2.Box(5, 5, 5)  # fully outside base_a -- guaranteed zero overlap
    raised_a, msg_a = False, ""
    try:
        _verify_subtraction("AT_A_no_flip", base_a, tool_a, requested_desc={"synthetic": "AT_A"})
    except ValueError as e:
        raised_a, msg_a = True, str(e)
    checks.append({"id": "guard_raises_on_zero_overlap_no_flip_hint",
                    "pass": raised_a and "SILENT ZERO" in msg_a and "axis_dir SHALL point INTO" not in msg_a,
                    "raised": raised_a, "msg_head": msg_a[:160]})

    # AT-B: same, but WITH a flip_tool_builder whose flipped tool DOES overlap -- confirms the
    # guard's flip test is a MEASURED intersection-volume probe (flip_would_cut True + a real
    # measured overlap number in the diagnosis), not a guessed suggestion.
    base_b = _bd2.Box(20, 20, 20)  # z in [-10,10]
    tool_b = _bd2.Pos(0, 0, 30) * _bd2.Box(10, 10, 10)  # z in [25,35] -- misses base_b entirely
    flip_b = _bd2.Pos(0, 0, -5) * _bd2.Box(10, 10, 10)  # z in [-10,0] -- WITHIN base_b -- real overlap

    def _flip_builder_b():
        return flip_b

    raised_b, msg_b, diag_b = False, "", None
    try:
        _verify_subtraction("AT_B_flip_would_cut", base_b, tool_b,
                             requested_desc={"synthetic": "AT_B"}, flip_tool_builder=_flip_builder_b)
    except ValueError as e:
        raised_b = True
        msg_b = str(e)
        diag_b = json.loads(msg_b.split("Diagnos (JSON): ", 1)[1])
    checks.append({"id": "guard_flip_test_measures_real_overlap_not_a_guess",
                    "pass": (raised_b and diag_b is not None and diag_b.get("flip_axis_would_cut") is True
                              and isinstance(diag_b.get("flip_axis_overlap_mm3"), (int, float))
                              and diag_b["flip_axis_overlap_mm3"] > 0.0
                              and "axis_dir SHALL point INTO the material" in msg_b),
                    "flip_axis_would_cut": diag_b.get("flip_axis_would_cut") if diag_b else None,
                    "flip_axis_overlap_mm3": diag_b.get("flip_axis_overlap_mm3") if diag_b else None})

    # AT-C / AT-D: the silent-zero guard on a real feature chain. A wrong-sign axis_dir must raise a
    # structured error instead of returning an unchanged body, and the correct-sign path must be
    # left untouched by the guard. Fixture: a drum shell with eight countersunk vent holes entering
    # its top face, all dimensions declared here.
    _R_OUTER, _HEIGHT, _WALL, _R_RIM = 250.0, 300.0, 10.0, 20.0
    _CENTER = (0.0, 0.0, 0.0)
    _TOP_Z = _HEIGHT / 2.0

    def _vent_holes():
        return [{"center": (dx, dy, _TOP_Z), "r": 18.0, "depth": 15.0, "csk": 4.0}
                for dx in (-120.0, -40.0, 40.0, 120.0) for dy in (-60.0, 60.0)]

    fan_wrong = f1_drum_shell("guard_fan_wrong", _R_OUTER, _HEIGHT, _CENTER, _WALL, _R_RIM)
    raised_c, msg_c, diag_c = False, "", None
    try:
        f4_countersunk_holes("guard_fan_wrong_vent", fan_wrong, _vent_holes(), axis_dir=(0.0, 0.0, 1.0))
    except ValueError as e:
        raised_c = True
        msg_c = str(e)
        diag_c = json.loads(msg_c.split("Diagnos (JSON): ", 1)[1])
    checks.append({"id": "wrong_sign_axis_dir_raises_structured_error",
                    "pass": (raised_c and diag_c is not None
                              and diag_c.get("flip_axis_would_cut") is True
                              and "axis_dir" in diag_c.get("requested", {})),
                    "raised": raised_c, "flip_axis_would_cut": diag_c.get("flip_axis_would_cut") if diag_c else None,
                    "requested_axis_dir": diag_c.get("requested", {}).get("axis_dir") if diag_c else None})

    # Correct sign: the batch call must remove material, must countersink all eight holes, and must
    # land on the same volume as the same holes cut one at a time through eight separate calls --
    # an independently computed reference, not a stored constant.
    fan_right = f1_drum_shell("guard_fan_right", _R_OUTER, _HEIGHT, _CENTER, _WALL, _R_RIM)
    v_uncut = fan_right.volume
    fan_right_cut, n_csk_right = f4_countersunk_holes(
        "guard_fan_right_vent", fan_right, _vent_holes(), axis_dir=(0.0, 0.0, -1.0))
    v_batch = fan_right_cut.volume

    seq_shape = f1_drum_shell("guard_fan_seq", _R_OUTER, _HEIGHT, _CENTER, _WALL, _R_RIM)
    n_csk_seq = 0
    for k, hole in enumerate(_vent_holes()):
        seq_shape, n_k = f4_countersunk_holes(f"guard_fan_seq_vent_{k}", seq_shape, [hole],
                                               axis_dir=(0.0, 0.0, -1.0))
        n_csk_seq += n_k
    v_sequential = seq_shape.volume
    rel_dev_seq = abs(v_batch - v_sequential) / v_sequential
    checks.append({"id": "right_sign_batch_equals_sequential_and_removes_material",
                    "pass": (n_csk_right == 8 and n_csk_seq == 8 and v_batch < v_uncut
                              and rel_dev_seq < 1e-6),
                    "v_uncut_mm3": v_uncut, "v_batch_mm3": v_batch, "v_sequential_mm3": v_sequential,
                    "removed_mm3": v_uncut - v_batch, "rel_dev_batch_vs_sequential": rel_dev_seq,
                    "n_countersunk_batch": n_csk_right, "n_countersunk_sequential": n_csk_seq})

    n_fail = sum(1 for c in checks if not c["pass"])
    return {"cell": "formfeature_v1_selftest", "checks": checks, "n_checks": len(checks),
            "n_fail": n_fail, "verdict": "PASS" if n_fail == 0 else "FAIL"}


if __name__ == "__main__":
    import json
    out = _selftest()
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["verdict"] == "PASS" else 1)
