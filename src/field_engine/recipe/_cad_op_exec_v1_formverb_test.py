#!/usr/bin/env python3
"""_cad_op_exec_v1_formverb_test.py -- acceptance tests for the form verbs in cad_op_exec_v1's dispatch
table (fillet, chamfer, revolve, sweep, shell, pattern_linear, pattern_circular, mirror).

Each handler is checked against an analytical reference value (relative deviation < 1e-6) on a
synthetic fixture, and each has one deliberate-failure case that must raise an honest error rather
than crash or pass through silently. One case also checks that a recipe using the form verbs still
hits the recipe cache on an unchanged prefix.

The fillet signature in an exported STEP file is checked as CYLINDRICAL_SURFACE count == edges
filleted: a box filleted on all twelve edges exports twelve cylindrical surfaces (one per edge) and
eight spherical surfaces (one per orthogonal corner), and no toroidal surface, which is the measured
signature for this fixture.

Run it with `python _cad_op_exec_v1_formverb_test.py`; it writes its report under artifacts/ and
exits 0 only if every check passes.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _KERNEL_DIR)

import build123d as bd  # noqa: E402

from cad_op_exec_v1 import exec_ops, DISPATCH  # noqa: E402
from recept_cache_v1 import exec_ops_cached  # noqa: E402

PROBES = os.path.join(_KERNEL_DIR, "artifacts")


def _rel(measured, expected):
    return abs(measured - expected) / abs(expected)


# =================================================================================== AT_fillet
def _at_fillet() -> dict:
    L, r = 20.0, 2.0
    ops = [
        {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": L, "height": L, "mode": "add"}]},
        {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": L},
        {"op": "fillet", "id": "filleted", "target_ref": "box", "radius": r,
         "selector": {"entity": "edge", "expected_count": 12}},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["filleted"].volume
    a = L - 2 * r
    expected = a**3 + 6 * r * a**2 + 3 * math.pi * r**2 * a + (4.0 / 3.0) * math.pi * r**3  # edge- + hornterm
    rel_dev = _rel(measured, expected)

    step_path = os.path.join(PROBES, "_formverb_v1_fillet.step")
    bd.export_step(out["solids"]["filleted"], step_path)
    with open(step_path) as f:
        step_text = f.read()
    n_cyl = step_text.count("CYLINDRICAL_SURFACE")
    n_tor = step_text.count("TOROIDAL_SURFACE")
    n_sph = step_text.count("SPHERICAL_SURFACE")

    # fallbevis: r=11 > L/2=10 -- MUST raise, not crash silently or pass
    fallbevis_raised, fallbevis_type = False, None
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk2", "shapes": [{"type": "rectangle", "width": L, "height": L, "mode": "add"}]},
            {"op": "extrude", "id": "box2", "sketch_ref": "sk2", "amount": L},
            {"op": "fillet", "id": "bad", "target_ref": "box2", "radius": 11.0,
             "selector": {"entity": "edge", "expected_count": 12}},
        ])
    except Exception as e:
        fallbevis_raised, fallbevis_type = True, type(e).__name__

    return {
        "measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6,
        "cylindrical_surface_count": n_cyl, "toroidal_surface_count": n_tor, "spherical_surface_count": n_sph,
        "n_edges_filleted": 12, "cylindrical_eq_edges_filleted": n_cyl == 12,
        "reframe": "TOROIDAL_SURFACE (brief spec) measured 0 for orthogonal-corner cube; "
                   "CYLINDRICAL_SURFACE==12 (edges) is the measured, correct signature -- see module docstring",
        "fallbevis": {"pass": fallbevis_raised, "exception_type": fallbevis_type, "params": "radius=11.0 (> L/2=10)"},
    }


# =================================================================================== AT_chamfer
def _at_chamfer() -> dict:
    L, c = 20.0, 2.0
    ops = [
        {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": L, "height": L, "mode": "add"}]},
        {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": L},
        {"op": "chamfer", "id": "chamfered", "target_ref": "box", "distance": c,
         "selector": {"entity": "edge", "expected_count": 12}},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["chamfered"].volume
    expected = L**3 - 6 * c**2 * L + (16.0 / 3.0) * c**3  # edgeterm - dubbelraknad hornoverlappning
    rel_dev = _rel(measured, expected)

    fallbevis_raised, fallbevis_type = False, None
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk2", "shapes": [{"type": "rectangle", "width": L, "height": L, "mode": "add"}]},
            {"op": "extrude", "id": "box2", "sketch_ref": "sk2", "amount": L},
            {"op": "chamfer", "id": "bad", "target_ref": "box2", "distance": 11.0,
             "selector": {"entity": "edge", "expected_count": 12}},
        ])
    except Exception as e:
        fallbevis_raised, fallbevis_type = True, type(e).__name__

    return {
        "measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6,
        "fallbevis": {"pass": fallbevis_raised, "exception_type": fallbevis_type, "params": "distance=11.0 (> L/2=10)"},
    }


# =================================================================================== AT_revolve
def _at_revolve() -> dict:
    R, a, b = 10.0, 4.0, 6.0
    plane = {"origin": [0, 0, 0], "x_dir": [1, 0, 0], "z_dir": [0, 1, 0]}  # normal=Y -> XZ half-plane
    ops = [
        {"op": "sketch_2d", "id": "sk", "plane": plane,
         "shapes": [{"type": "rectangle", "width": a, "height": b, "center": [R, 0], "mode": "add"}]},
        {"op": "revolve", "id": "rev", "sketch_ref": "sk", "axis": "Z", "angle_deg": 360.0},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["rev"].volume
    expected = 2.0 * math.pi * R * a * b  # Pappus
    rel_dev = _rel(measured, expected)

    # fallbevis: profile straddling the axis (center x=0, half-width 2 > R=0) -- MUST raise
    plane_straddle = {"origin": [0, 0, 0], "x_dir": [1, 0, 0], "z_dir": [0, 1, 0]}
    fallbevis_raised, fallbevis_type = False, None
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk2", "plane": plane_straddle,
             "shapes": [{"type": "rectangle", "width": a, "height": b, "mode": "add"}]},
            {"op": "revolve", "id": "bad", "sketch_ref": "sk2", "axis": "Z", "angle_deg": 360.0},
        ])
    except Exception as e:
        fallbevis_raised, fallbevis_type = True, type(e).__name__

    return {
        "measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6,
        "fallbevis": {"pass": fallbevis_raised, "exception_type": fallbevis_type,
                      "params": "profile centered at axis (straddles it)"},
    }


# =================================================================================== AT_sweep
def _at_sweep() -> dict:
    # -- straight path: rectangle profile A=3*4 along a 20mm straight line -- V = A*L
    ops_straight = [
        {"op": "sketch_2d", "id": "path", "shapes": [{"type": "line", "start": [0, 0, 0], "end": [20, 0, 0]}]},
        {"op": "sketch_2d", "id": "prof",
         "plane": {"origin": [0, 0, 0], "x_dir": [0, 1, 0], "z_dir": [1, 0, 0]},
         "shapes": [{"type": "rectangle", "width": 3, "height": 4, "mode": "add"}]},
        {"op": "sweep", "id": "swept_straight", "sketch_ref": "prof", "path_ref": "path"},
    ]
    out_s = exec_ops(ops_straight)
    measured_straight = out_s["solids"]["swept_straight"].volume
    expected_straight = 3.0 * 4.0 * 20.0
    rel_dev_straight = _rel(measured_straight, expected_straight)

    # -- curved path: circular profile A=pi*r^2 along a circular arc radius Rp, 90deg -- Pappus V=A*L
    Rp, ang, r_prof = 5.0, 90.0, 1.0
    ops_curved = [
        {"op": "sketch_2d", "id": "arcpath",
         "shapes": [{"type": "arc", "center": [0, 0, 0], "radius": Rp, "start_angle_deg": 0.0, "arc_size_deg": ang}]},
        {"op": "sketch_2d", "id": "circprof",
         "plane": {"origin": [Rp, 0, 0], "x_dir": [1, 0, 0], "z_dir": [0, 1, 0]},  # normal=tangent at start (CCW)
         "shapes": [{"type": "circle", "radius": r_prof, "mode": "add"}]},
        {"op": "sweep", "id": "swept_curved", "sketch_ref": "circprof", "path_ref": "arcpath"},
    ]
    out_c = exec_ops(ops_curved)
    measured_curved = out_c["solids"]["swept_curved"].volume
    L_curved = Rp * math.radians(ang)
    expected_curved = math.pi * r_prof**2 * L_curved  # Pappus for the curved sweep
    rel_dev_curved = _rel(measured_curved, expected_curved)

    # fallbevis: profile plane NOT perpendicular to path tangent at start (silent-wrong-geometry guard)
    fallbevis_raised, fallbevis_type = False, None
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "arcpath2",
             "shapes": [{"type": "arc", "center": [0, 0, 0], "radius": Rp, "start_angle_deg": 0.0, "arc_size_deg": ang}]},
            {"op": "sketch_2d", "id": "badprof",
             "plane": {"origin": [Rp, 0, 0], "x_dir": [0, 1, 0], "z_dir": [1, 0, 0]},  # normal=radial, WRONG
             "shapes": [{"type": "circle", "radius": r_prof, "mode": "add"}]},
            {"op": "sweep", "id": "bad", "sketch_ref": "badprof", "path_ref": "arcpath2"},
        ])
    except Exception as e:
        fallbevis_raised, fallbevis_type = True, type(e).__name__

    return {
        "straight": {"measured": measured_straight, "expected": expected_straight,
                     "rel_dev": rel_dev_straight, "pass": rel_dev_straight < 1e-6},
        "curved_pappus": {"measured": measured_curved, "expected": expected_curved,
                           "rel_dev": rel_dev_curved, "pass": rel_dev_curved < 1e-6, "arc_length_mm": L_curved},
        "pass": (rel_dev_straight < 1e-6) and (rel_dev_curved < 1e-6),
        "fallbevis": {"pass": fallbevis_raised, "exception_type": fallbevis_type,
                      "params": "profile plane normal=radial (not tangent) -- degenerate ~0-volume sweep"},
    }


# =================================================================================== AT_shell
def _at_shell() -> dict:
    L, B, H, t = 30.0, 20.0, 10.0, 2.0
    ops = [
        {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": L, "height": B, "mode": "add"}]},
        {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": H},
        {"op": "shell", "id": "shelled", "target_ref": "box", "thickness": t},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["shelled"].volume
    expected = L * B * H - (L - 2 * t) * (B - 2 * t) * (H - 2 * t)
    rel_dev = _rel(measured, expected)

    fallbevis_raised, fallbevis_type = False, None
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk2", "shapes": [{"type": "rectangle", "width": L, "height": B, "mode": "add"}]},
            {"op": "extrude", "id": "box2", "sketch_ref": "sk2", "amount": H},
            {"op": "shell", "id": "bad", "target_ref": "box2", "thickness": 6.0},  # > H/2=5
        ])
    except Exception as e:
        fallbevis_raised, fallbevis_type = True, type(e).__name__

    return {
        "measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6,
        "fallbevis": {"pass": fallbevis_raised, "exception_type": fallbevis_type,
                      "params": "thickness=6.0 (> H/2=5)"},
    }


# =========================================================================== AT_pattern_linear
def _at_pattern_linear() -> dict:
    n, spacing = 5, 10.0
    ops = [
        {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": 3, "height": 3, "mode": "add"}]},
        {"op": "extrude", "id": "unit", "sketch_ref": "sk", "amount": 3},
        {"op": "pattern_linear", "id": "patt", "target_ref": "unit", "direction": [1, 0, 0],
         "count": n, "spacing": spacing},
    ]
    out = exec_ops(ops)
    result = out["solids"]["patt"]
    v0 = 3.0 * 3.0 * 3.0
    measured = result.volume
    expected = n * v0
    rel_dev = _rel(measured, expected)
    n_solids = len(result.solids())

    fallbevis_raised, fallbevis_type = False, None
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk2", "shapes": [{"type": "rectangle", "width": 3, "height": 3, "mode": "add"}]},
            {"op": "extrude", "id": "unit2", "sketch_ref": "sk2", "amount": 3},
            {"op": "pattern_linear", "id": "bad", "target_ref": "unit2", "direction": [1, 0, 0],
             "count": 0, "spacing": spacing},
        ])
    except Exception as e:
        fallbevis_raised, fallbevis_type = True, type(e).__name__

    return {
        "measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6,
        "n_instances_expected": n, "n_solids_in_tree": n_solids, "n_solids_pass": n_solids == n,
        "fallbevis": {"pass": fallbevis_raised, "exception_type": fallbevis_type, "params": "count=0"},
    }


# ========================================================================= AT_pattern_circular
def _at_pattern_circular() -> dict:
    n = 6
    ops = [
        {"op": "sketch_2d", "id": "sk", "plane": {"origin": [10, 0, 0], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]},
         "shapes": [{"type": "rectangle", "width": 2, "height": 2, "mode": "add"}]},
        {"op": "extrude", "id": "unit", "sketch_ref": "sk", "amount": 2},
        {"op": "pattern_circular", "id": "patt", "target_ref": "unit", "axis": "Z", "count": n, "angle_deg": 360.0},
    ]
    out = exec_ops(ops)
    result = out["solids"]["patt"]
    v0 = 2.0 * 2.0 * 2.0
    measured = result.volume
    expected = n * v0
    rel_dev = _rel(measured, expected)
    n_solids = len(result.solids())

    fallbevis_raised, fallbevis_type = False, None
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk2", "plane": {"origin": [10, 0, 0], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]},
             "shapes": [{"type": "rectangle", "width": 2, "height": 2, "mode": "add"}]},
            {"op": "extrude", "id": "unit2", "sketch_ref": "sk2", "amount": 2},
            {"op": "pattern_circular", "id": "bad", "target_ref": "unit2", "axis": "Z", "count": 0},
        ])
    except Exception as e:
        fallbevis_raised, fallbevis_type = True, type(e).__name__

    return {
        "measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6,
        "n_instances_expected": n, "n_solids_in_tree": n_solids, "n_solids_pass": n_solids == n,
        "fallbevis": {"pass": fallbevis_raised, "exception_type": fallbevis_type, "params": "count=0"},
    }


# =================================================================================== AT_mirror
def _at_mirror() -> dict:
    ops = [
        {"op": "sketch_2d", "id": "sk", "plane": {"origin": [5, 0, 0], "x_dir": [0, 1, 0], "z_dir": [1, 0, 0]},
         "shapes": [{"type": "rectangle", "width": 4, "height": 6, "mode": "add"}]},
        {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": 3},
        {"op": "mirror", "id": "mirrored", "target_ref": "box", "plane": "YZ"},
    ]
    out = exec_ops(ops)
    orig = out["solids"]["box"]
    mirr = out["solids"]["mirrored"]
    vol_dev = _rel(mirr.volume, orig.volume)
    cx_orig = orig.center().X
    cx_mirr = mirr.center().X
    centroid_dev = abs(cx_mirr - (-cx_orig)) / abs(cx_orig)

    fallbevis_raised, fallbevis_type = False, None
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk2", "shapes": [{"type": "rectangle", "width": 4, "height": 6, "mode": "add"}]},
            {"op": "extrude", "id": "box2", "sketch_ref": "sk2", "amount": 3},
            {"op": "mirror", "id": "bad", "target_ref": "box2", "plane": "QR"},
        ])
    except Exception as e:
        fallbevis_raised, fallbevis_type = True, type(e).__name__

    return {
        "volume_measured": mirr.volume, "volume_expected": orig.volume, "volume_rel_dev": vol_dev,
        "centroid_x_orig": cx_orig, "centroid_x_mirrored": cx_mirr, "centroid_rel_dev": centroid_dev,
        "pass": (vol_dev < 1e-6) and (centroid_dev < 1e-6),
        "fallbevis": {"pass": fallbevis_raised, "exception_type": fallbevis_type, "params": "plane='QR' (unknown)"},
    }


# =========================================================================== cache measurement
def _cache_measurement() -> dict:
    """Measured, not assumed: recept_cache_v1 imports DISPATCH directly from cad_op_exec_v1, so the 8 new
    handlers are cached identically -- verify prefix-reuse (AT5a pattern) with a NEW-verb (fillet)
    recipe: a late param change downstream of the fillet op must cache-HIT the fillet prefix."""
    tmp = os.path.join(PROBES, "_formverb_v1_cache_store")
    shutil.rmtree(tmp, ignore_errors=True)

    def make_ops(hole_d):
        return [
            {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": 20, "height": 20, "mode": "add"}]},
            {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": 20},
            {"op": "fillet", "id": "filleted", "target_ref": "box", "radius": 2.0,
             "selector": {"entity": "edge", "expected_count": 12}},
            {"op": "hole", "id": "holed", "target_ref": "filleted", "center": [0, 0, 10], "diameter": hole_d,
             "axis": "Z", "through": True},
        ]

    t0 = time.time()
    run1 = exec_ops_cached(make_ops(3.0), cache_dir=tmp)
    t_miss = time.time() - t0
    t0 = time.time()
    run2 = exec_ops_cached(make_ops(5.0), cache_dir=tmp)  # LATE param change (hole diameter only)
    t_hit = time.time() - t0

    # 16B (cache_16b_v1): recept_cache_v1's log["cache"] field now distinguishes HIT_MEM/HIT_DISK
    # (memory-tier hit vs disk-tier hit) instead of a single "HIT" -- startswith("HIT") preserves
    # this test's original intent (a cache hit, either tier) without caring WHICH tier served it.
    prefix_hit = (run2["log"][0]["cache"].startswith("HIT") and run2["log"][1]["cache"].startswith("HIT")
                  and run2["log"][2]["cache"].startswith("HIT"))
    fillet_op_cache_hit = run2["log"][2]["cache"].startswith("HIT")  # the NEW-verb op itself
    late_op_miss = run2["log"][3]["cache"] == "MISS"

    fresh = exec_ops(make_ops(5.0))
    cached_vol = run2["solids"]["holed"].volume
    fresh_vol = fresh["solids"]["holed"].volume
    roundtrip_identical = abs(cached_vol - fresh_vol) < 1e-6

    return {
        "pass": bool(prefix_hit and fillet_op_cache_hit and late_op_miss and roundtrip_identical),
        "prefix_hit": prefix_hit, "fillet_op_cache_hit": fillet_op_cache_hit, "late_op_miss": late_op_miss,
        "roundtrip_identical": roundtrip_identical, "cached_vol": cached_vol, "fresh_vol": fresh_vol,
        "wall_ms_miss_run": round(t_miss * 1000.0, 2), "wall_ms_hit_run": round(t_hit * 1000.0, 2),
        "cache_hits_run2": run2["cache_stats"]["cache_hits"],
    }


# =================================================================================== main
def run_all() -> dict:
    results = {}
    results["fillet"] = _at_fillet()
    results["chamfer"] = _at_chamfer()
    results["revolve"] = _at_revolve()
    results["sweep"] = _at_sweep()
    results["shell"] = _at_shell()
    results["pattern_linear"] = _at_pattern_linear()
    results["pattern_circular"] = _at_pattern_circular()
    results["mirror"] = _at_mirror()
    results["cache_measurement"] = _cache_measurement()

    results["reference_still_not_implemented"] = _reference_not_implemented()

    handler_names = ["fillet", "chamfer", "revolve", "sweep", "shell",
                      "pattern_linear", "pattern_circular", "mirror"]
    all_verb_pass = all(results[k]["pass"] for k in handler_names)
    all_fallbevis_pass = all(
        results[k].get("fallbevis", {}).get("pass", False) if k not in ("sweep",) else results[k]["fallbevis"]["pass"]
        for k in handler_names
    )
    # REFRAME DECLARED: dispatch_coverage's own expected handler count was hardcoded ==15 -- that
    # was MEASURED true for the 8 FORMVERBEN cell, is FALSE now that "draft" is a 9th real handler.
    # Bumped to 16, not silently left stale. REFRAME DECLARED AGAIN: bumped 16->18 --sheet_bend +
    # thread_cosmetic are 2 more real DISPATCH handlers, same "measured true then, false now"
    # pattern, same fix (bump the number, don't silently leave it stale). REFRAME DECLARED AGAIN:
    # bumped 18->19 -- "rib" (formrib_v1.rib_v1 DISPATCH-wiring) is a 19th real handler, same
    # pattern once more. REFRAME DECLARED AGAIN: bumped 19->20 -- "sketch_profile" (sketch becomes a
    # recipe citizen, cad_op_exec_v1._handle_sketch_profile) is a 20th real handler, same "measured
    # true then, false now" pattern once more.
    dispatch_now = sorted(DISPATCH.keys())
    all_pass = bool(
        all_verb_pass and all_fallbevis_pass and results["cache_measurement"]["pass"]
        and results["reference_still_not_implemented"]["pass"]
        and len(dispatch_now) == 20
    )
    results["dispatch_coverage"] = {"n_handlers": len(dispatch_now), "handlers": dispatch_now, "pass": len(dispatch_now) == 20}
    results["all_pass"] = all_pass
    return results


def _reference_not_implemented() -> dict:
    ok, msg = False, ""
    try:
        exec_ops([{"op": "reference", "id": "rf1", "kind": "plane"}])
    except NotImplementedError as e:
        ok, msg = True, str(e)
    return {"pass": ok, "raised_msg": msg}


def _build_atoms(results: dict, report_path: str) -> dict:
    # EXTERNALLY ANCHORED: each expected value below is the INDEPENDENTLY-DERIVED analytic literal
    # (Pappus/Minkowski/prism formulas, module docstring), NOT a re-read of the artifact's own
    # "expected" field -- a wrong handler must be able to flip these, which a self-consistent
    # rel_dev-vs-itself atom could not (one-sided "<1e-6" inequalities against a loose positive rhs
    # are also structurally TRIVIAL under the null-model, per ensidig-olikhetsatom-strukturellt-svag
    # -- replaced with tight two-sided value bands here).
    atoms = [
        {"type": "value-in-artifact", "artifact": report_path, "key": "fillet.measured",
         "expected": 7804.696111127531, "tol": 1e-6 * 7804.696111127531},
        {"type": "value-in-artifact", "artifact": report_path, "key": "chamfer.measured",
         "expected": 7562.666666666667, "tol": 1e-6 * 7562.666666666667},
        {"type": "value-in-artifact", "artifact": report_path, "key": "revolve.measured",
         "expected": 1507.9644737231006, "tol": 1e-6 * 1507.9644737231006},
        {"type": "value-in-artifact", "artifact": report_path, "key": "shell.measured",
         "expected": 3504.0, "tol": 1e-6 * 3504.0},
        {"type": "value-in-artifact", "artifact": report_path, "key": "sweep.straight.measured",
         "expected": 240.0, "tol": 1e-6 * 240.0},
        {"type": "value-in-artifact", "artifact": report_path, "key": "sweep.curved_pappus.measured",
         "expected": 24.674011002723393, "tol": 1e-6 * 24.674011002723393},
        {"type": "value-in-artifact", "artifact": report_path, "key": "pattern_linear.measured",
         "expected": 135.0, "tol": 1e-6 * 135.0},
        {"type": "value-in-artifact", "artifact": report_path, "key": "pattern_circular.measured",
         "expected": 48.0, "tol": 1e-6 * 48.0},
        {"type": "value-in-artifact", "artifact": report_path, "key": "mirror.volume_measured",
         "expected": 72.0, "tol": 1e-6 * 72.0},
        {"type": "value-in-artifact", "artifact": report_path, "key": "mirror.centroid_x_mirrored",
         "expected": -6.5, "tol": 1e-6 * 6.5},
    ]
    for verb in ["fillet", "chamfer", "revolve", "shell", "sweep", "pattern_linear", "pattern_circular", "mirror"]:
        atoms.append({"type": "value-in-artifact", "artifact": report_path, "key": f"{verb}.fallbevis.pass",
                      "expected": True})
    for verb in ["pattern_linear", "pattern_circular"]:
        atoms.append({"type": "value-in-artifact", "artifact": report_path, "key": f"{verb}.n_solids_pass",
                      "expected": True})
    atoms.append({"type": "value-in-artifact", "artifact": report_path, "key": "fillet.cylindrical_eq_edges_filleted",
                  "expected": True})
    atoms.append({"type": "value-in-artifact", "artifact": report_path, "key": "cache_measurement.pass",
                  "expected": True})
    atoms.append({"type": "value-in-artifact", "artifact": report_path, "key": "reference_still_not_implemented.pass",
                  "expected": True})
    atoms.append({"type": "value-in-artifact", "artifact": report_path, "key": "dispatch_coverage.n_handlers",
                  "expected": 20})
    atoms.append({"type": "command-exit-0", "command": "python cad_op_exec_v1.py"})
    return {"atoms": atoms}


if __name__ == "__main__":
    out = run_all()
    report_path_rel = os.path.join("artifacts", "formverb_v1.json")
    report_path_abs = os.path.join(_KERNEL_DIR, report_path_rel)
    out["ATOMS"] = _build_atoms(out, report_path_rel)
    os.makedirs(os.path.dirname(report_path_abs), exist_ok=True)
    with open(report_path_abs, "w") as f:
        json.dump(out, f, indent=2)
    printable = {k: v for k, v in out.items()}
    print(json.dumps({"all_pass": out["all_pass"], "dispatch_coverage": out["dispatch_coverage"]}, indent=2))
    sys.exit(0 if out["all_pass"] else 1)
