#!/usr/bin/env python3
"""formrib_v1.py -- the rib form verb, a recipe idiom in the same class as formfeature_v1.py's
F1-F4: it introduces ZERO new native `cad_op_schema_v1` op types. The strip is built from a
`sketch_2d` + `extrude` pair through `cad_op_exec_v1`'s dispatch (logged, per-op cache); the boolean
against the caller's already-live `target` shape is applied as a direct build123d operator call, the
same declared exception formfeature_v1.py's F2/F3/F4 already use, because re-routing a live external
shape through a second op list under an id it never had is outside this layer's contract (`+`/`-`
on live shapes is literally what `_handle_boolean_union`/`_handle_boolean_cut` do internally).

The welded material is `raw_strip - target`, not `raw_strip & target`. That choice is measured, not
stylistic: for any shapes A and B, `A & B` is a subset of B, so `B + (A & B) == B` identically and a
rib built that way would be a structural no-op with zero effect on the model at every placement.
With the cut form, all three invariants hold simultaneously and non-trivially:
  - V(union) - V(target_before) == V(added), exactly, at any placement (the two sets are disjoint
    by construction after the cut)
  - a strip fully inside the target adds nothing: V(added) == 0, which the planted-fault case below
    requires to raise rather than silently return the target unchanged
  - a strip flush against the target with zero volumetric overlap adds the whole raw prism:
    V(added) == thickness * run_length * depth
The geometric contract (the rib welds on, is additive and never protrudes past where it should) is
unchanged; only the boolean verb differs.

API:
    rib_v1(tag, target, midplane, thickness, run_length, depth) -> (result_shape, meta: dict)
        midplane: {"origin":[x,y,z], "x_dir":[..], "z_dir":[..]} -- local sketch x = thickness axis,
        local sketch y = run_length axis, z_dir (plane normal) = depth/extrude axis. The extrude is
        SYMMETRIC about the given midplane (origin +/- depth/2 along the normal), which is a rib's
        real geometric meaning (a stiffener straddling a wall or reference plane), not a one-sided
        pad. Raises ValueError if the strip adds no material, or if `target` is not a solid.
    meta carries the measured volumes (v_before, v_raw_strip, v_added, v_after) and the identity
    deviation, for the caller's own reporting.

Run the selftest with `python formrib_v1.py`; it prints a JSON report and exits non-zero if a check
fails.
"""
from __future__ import annotations

import math
import os
import sys

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)
ARTIFACTS = os.path.join(_KERNEL_DIR, "artifacts")

from cad_op_schema_v1 import validate_recipe  # noqa: E402
from cad_op_exec_v1 import exec_ops as _exec_ops_uncached  # noqa: E402

_LAST_LOG: list = []
_ZERO_MATERIAL_TOL_MM3 = 1e-6  # below this, "added" is a floating-point ghost, not real material


def _run(ops: list, tag: str) -> dict:
    """Same idiom as formfeature_v1.py's own _run: schema-validate, then execute through the
    uncached in-memory dispatch, and raise on any FAIL row."""
    global _LAST_LOG
    validate_recipe({"ops": ops})
    log_path = os.path.join(ARTIFACTS, "formrib_v1_exec_logs", f"{tag}.json")
    result = _exec_ops_uncached(ops, log_path=log_path)
    _LAST_LOG = result["log"]
    for row in result["log"]:
        if row.get("status") == "FAIL":
            raise ValueError(f"formrib_v1[{tag}]: op {row['op_id']} FAILED: {row.get('reason')}")
    return result["solids"]


def rib_v1(tag: str, target, midplane: dict, thickness: float, run_length: float, depth: float):
    """Build a thin rectangular rib strip (thickness x run_length x depth) centered on `midplane`,
    then weld ONLY the portion that is genuinely NEW material (raw_strip - target, see the module
    docstring) onto `target`. Returns (welded_shape, meta) -- meta carries the measured volumes
    (v_before, v_raw_strip, v_added, v_after) for the caller's own reporting."""
    import build123d as bd

    if thickness <= 0 or run_length <= 0 or depth <= 0:
        raise ValueError(f"rib_v1 requires thickness,run_length,depth all > 0, got "
                          f"{thickness},{run_length},{depth}")
    if target is None or not hasattr(target, "volume"):
        raise ValueError(
            "rib_v1 requires a target solid (the surface the rib welds onto) -- got "
            f"{target!r}; 'rib without a target face' is a configuration error, not a silent no-op"
        )

    origin = tuple(midplane.get("origin", (0, 0, 0)))
    x_dir = tuple(midplane.get("x_dir", (1, 0, 0)))
    z_dir = tuple(midplane.get("z_dir", (0, 0, 1)))
    normal = bd.Vector(*z_dir).normalized()
    # SYMMETRIC extrude about the given midplane: shift the sketch origin back by depth/2 along the
    # normal, then extrude the full `depth` forward -- result spans origin +/- depth/2 along normal.
    shifted_origin = (origin[0] - normal.X * depth / 2.0,
                       origin[1] - normal.Y * depth / 2.0,
                       origin[2] - normal.Z * depth / 2.0)
    ops = [
        {"op": "sketch_2d", "id": f"{tag}_sk",
         "plane": {"origin": list(shifted_origin), "x_dir": list(x_dir), "z_dir": list(z_dir)},
         "shapes": [{"type": "rectangle", "width": thickness, "height": run_length, "mode": "add"}]},
        {"op": "extrude", "id": f"{tag}_strip", "sketch_ref": f"{tag}_sk", "amount": depth},
    ]
    solids = _run(ops, tag)
    raw_strip = solids[f"{tag}_strip"]
    v_raw = raw_strip.volume

    # boolean_cut (see module docstring) -- direct live-object operators, same declared
    # idiom formfeature_v1.py's F2/F3/F4 already use for post-processing an external live shape.
    added = raw_strip - target
    v_added = getattr(added, "volume", 0.0) or 0.0

    if v_added < _ZERO_MATERIAL_TOL_MM3:
        raise ValueError(
            "rib fully contained in target, adds zero material "
            f"(v_raw_strip={v_raw:.6f}mm3, v_added={v_added:.6f}mm3 < tol {_ZERO_MATERIAL_TOL_MM3})"
        )

    v_before = target.volume
    result = target + added
    v_after = result.volume

    meta = {
        "v_before": v_before, "v_raw_strip": v_raw, "v_added": v_added, "v_after": v_after,
        "identity_rel_dev": abs((v_after - v_before) - v_added) / max(v_added, 1e-12),
    }
    return result, meta


# =================================================================================== SELFTEST
def _box_target(tag: str, L: float, B: float, H: float):
    """L x B x H box, X/Y centred at the world origin, Z in [0,H] -- the same convention the
    fillet/chamfer/shell fixtures in the neighbouring selftests use."""
    ops = [
        {"op": "sketch_2d", "id": f"{tag}_sk", "shapes": [{"type": "rectangle", "width": L, "height": B, "mode": "add"}]},
        {"op": "extrude", "id": f"{tag}_box", "sketch_ref": f"{tag}_sk", "amount": H},
    ]
    solids = _run(ops, tag)
    return solids[f"{tag}_box"]


def _at_rib_partial_overlap() -> dict:
    """A) HAPPY PATH: rib straddles the target's top face -- half embedded (welds), half protrudes
    (real new material). Reference value: closed-form AABB box-difference volume, independent of
    the execution machinery -- both target and raw strip are axis-aligned boxes, so the overlap and
    the protrusion are exact arithmetic, not re-read from the run's own numbers."""
    L, B, H = 40.0, 30.0, 20.0
    target = _box_target("rib_a_target", L, B, H)
    thickness, run_length, depth = 6.0, 12.0, 20.0
    midplane = {"origin": [0.0, 0.0, H], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]}
    result, meta = rib_v1("rib_a", target, midplane, thickness, run_length, depth)

    # independent closed-form: raw strip spans Z in [H-depth/2, H+depth/2] = [10,30]; target spans
    # Z in [0,H]=[0,20]; overlap Z-height = 20-10 = 10 -> embedded volume = thickness*run_length*10;
    # protruding (added) volume = raw - embedded = thickness*run_length*(depth - overlap_height)
    z_lo, z_hi = H - depth / 2.0, H + depth / 2.0
    overlap_h = max(0.0, min(z_hi, H) - max(z_lo, 0.0))
    expected_added = thickness * run_length * (depth - overlap_h)
    rel_dev = abs(meta["v_added"] - expected_added) / expected_added
    expected_v_after = L * B * H + expected_added
    v_after_rel_dev = abs(meta["v_after"] - expected_v_after) / expected_v_after

    return {
        "meta": meta, "expected_added_mm3": expected_added, "rel_dev": rel_dev,
        "expected_v_after_mm3": expected_v_after, "v_after_rel_dev": v_after_rel_dev,
        "identity_rel_dev": meta["identity_rel_dev"],
        "pass": bool(rel_dev < 1e-6 and v_after_rel_dev < 1e-6 and meta["identity_rel_dev"] < 1e-6),
    }


def _at_rib_untrimmed_full_prism() -> dict:
    """B) the untrimmed special case: rib entirely OUTSIDE the target (flush contact, zero
    volumetric overlap) -- the cut removes nothing and added == the raw prism volume exactly
    (thickness*depth*run_length), a second, independent anchor."""
    L, B, H = 40.0, 30.0, 20.0
    target = _box_target("rib_b_target", L, B, H)
    thickness, run_length, depth = 6.0, 12.0, 6.0
    midplane = {"origin": [0.0, 0.0, H + depth / 2.0], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]}
    result, meta = rib_v1("rib_b", target, midplane, thickness, run_length, depth)

    expected_added = thickness * run_length * depth  # prism formula
    rel_dev = abs(meta["v_added"] - expected_added) / expected_added
    return {
        "meta": meta, "expected_added_mm3": expected_added, "rel_dev": rel_dev,
        "pass": bool(rel_dev < 1e-6 and meta["identity_rel_dev"] < 1e-6),
    }


def _at_rib_fallbevis_fully_contained() -> dict:
    """C) planted fault: a rib strip fully INSIDE the target must raise, not silently return the
    target unchanged while pretending the rib was applied."""
    L, B, H = 40.0, 30.0, 20.0
    target = _box_target("rib_c_target", L, B, H)
    thickness, run_length, depth = 6.0, 12.0, 6.0
    midplane = {"origin": [0.0, 0.0, 10.0], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]}  # fully inside Z:[0,20]

    raised, exc_type, msg = False, None, ""
    try:
        rib_v1("rib_c", target, midplane, thickness, run_length, depth)
    except Exception as e:  # noqa: BLE001
        raised, exc_type, msg = True, type(e).__name__, str(e)

    return {
        "pass": bool(raised and exc_type == "ValueError" and "zero material" in msg),
        "exception_type": exc_type, "raised_msg": msg,
    }


def _at_rib_missing_target_face() -> dict:
    """D) planted fault: a rib with no target face at all (target=None) must raise a clear, typed
    error instead of leaking OCC internals."""
    thickness, run_length, depth = 6.0, 12.0, 6.0
    midplane = {"origin": [0.0, 0.0, 0.0], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]}
    raised, exc_type, msg = False, None, ""
    try:
        rib_v1("rib_d", None, midplane, thickness, run_length, depth)
    except Exception as e:  # noqa: BLE001
        raised, exc_type, msg = True, type(e).__name__, str(e)
    return {
        "pass": bool(raised and exc_type in ("ValueError", "TypeError", "AttributeError")),
        "exception_type": exc_type, "raised_msg": msg,
        "desc": "rib_v1(target=None) -- a rib without a target face must raise an understandable error",
    }


def selftest() -> dict:
    results = {
        "partial_overlap": _at_rib_partial_overlap(),
        "untrimmed_full_prism": _at_rib_untrimmed_full_prism(),
        "fallbevis_fully_contained": _at_rib_fallbevis_fully_contained(),
        "fallbevis_missing_target": _at_rib_missing_target_face(),
    }
    results["all_pass"] = bool(all(v["pass"] for v in results.values()))
    return results


if __name__ == "__main__":
    import json
    out = selftest()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["all_pass"] else 1)
