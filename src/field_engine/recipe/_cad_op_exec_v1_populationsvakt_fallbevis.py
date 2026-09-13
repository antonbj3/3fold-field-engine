#!/usr/bin/env python3
"""_cad_op_exec_v1_populationsvakt_fallbevis.py -- planted-fault evidence that the ALL-population guard
is wired into cad_op_exec_v1.exec_ops itself, not merely callable afterwards by a harness.

Three recipes each fold a population-drifting op and the later ALL-position op into a single
exec_ops() call, with the first solid's measured population declared as "expected_population" on the
later op, so the refusal comes from the executor's own internal guard before the destructive handler
touches OCC:
  - a box, fillet(selector=None), then chamfer(selector=None) declaring the pre-fillet edge count
  - a box, fillet with a narrow near_point selector, then the same undeclared-drift chamfer
  - a box, pattern_circular(count=3), then chamfer(selector=None) declaring the pre-pattern count
A fourth case is the backward-compatibility control: the same chain with no "expected_population" key
must not raise, must produce the same geometry, and must only warn and flag in the op log.

Run it with `python _cad_op_exec_v1_populationsvakt_fallbevis.py`; it prints a JSON report and exits
non-zero if a case does not behave as stated.
"""
from __future__ import annotations

import json
import os
import sys

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(_KERNEL_DIR, "artifacts")
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)

from cad_op_exec_v1 import exec_ops  # noqa: E402
from geometri_selektor_v1 import RefusalError, REASON_POPULATION_DRIFT, record_expected_population  # noqa: E402

REPORT_PATH = os.path.join(ARTIFACTS, "exec_populationsvakt_fallbevis_v1.json")


def _case_selector_none_chain_hazard_via_exec() -> dict:
    """Same recipe as _kernel_repro_suite_v1._case_selector_none_chain_hazard (300x300x300 box,
    fillet(selector=None,r=0.4), then chamfer(selector=None) declaring the PRE-fillet edge population
    (12) as expected_population) -- but the chamfer op is now IN THE SAME exec_ops(ops) call as the
    fillet that drifts it, so the guard fires FROM exec_ops itself, mid-chain, before the chamfer
    handler ever touches OCC."""
    fresh_ops = [
        {"id": "sb", "op": "sketch_2d", "shapes": [{"type": "rectangle", "width": 300.0, "height": 300.0}]},
        {"id": "d0", "op": "extrude", "sketch_ref": "sb", "amount": 300.0},
    ]
    fresh = exec_ops(fresh_ops)
    declared = record_expected_population(fresh["solids"]["d0"], "edge")

    chain = fresh_ops + [
        {"id": "d1", "op": "fillet", "target_ref": "d0", "radius": 0.4},
        {"id": "d2", "op": "chamfer", "target_ref": "d1", "distance": 0.05,
         "expected_population": declared["expected_population"]},
    ]
    try:
        exec_ops(chain)
        return {"name": "selector_none_chain_hazard_via_exec", "guard_raised_by_exec": False,
                "verdict": "FAIL (exec_ops did not refuse the drifted ALL population)"}
    except RefusalError as e:
        ok = (e.reason == REASON_POPULATION_DRIFT and e.diagnosis.get("op_id") == "d2"
              and "chain_position" in e.diagnosis.get("history", {})
              and e.diagnosis["history"]["chain_position"] == 3
              and e.diagnosis["expected_population"] == declared["expected_population"]
              and e.diagnosis["delta"] > 0)
        return {"name": "selector_none_chain_hazard_via_exec", "guard_raised_by_exec": True,
                "reason": e.reason, "op_id": e.diagnosis.get("op_id"),
                "chain_position": e.diagnosis.get("history", {}).get("chain_position"),
                "expected_population": e.diagnosis["expected_population"],
                "actual_population": e.diagnosis["actual_population"], "delta": e.diagnosis["delta"],
                "verdict": "PASS" if ok else "FAIL (refused but diagnosis payload wrong)"}


def _case_narrow_selector_still_redefines_downstream_all_via_exec() -> dict:
    """Same recipe as _kernel_repro_suite_v1._case_narrow_selector_still_redefines_downstream_all
    (200x120x80 box, fillet with a NARROW near_point selector, then chamfer(selector=None) declaring
    the pre-fillet edge population) -- folded into ONE exec_ops(ops) call."""
    near_sel = {"entity": "edge", "near_point": [100.0, 60.0, 0.0], "expected_count": 1}
    fresh_ops = [
        {"id": "sb", "op": "sketch_2d", "shapes": [{"type": "rectangle", "width": 200.0, "height": 120.0}]},
        {"id": "d0", "op": "extrude", "sketch_ref": "sb", "amount": 80.0},
    ]
    fresh = exec_ops(fresh_ops)
    declared = record_expected_population(fresh["solids"]["d0"], "edge")

    chain = fresh_ops + [
        {"id": "d1", "op": "fillet", "target_ref": "d0", "radius": 0.4, "selector": near_sel},
        {"id": "d2", "op": "chamfer", "target_ref": "d1", "distance": 0.05,
         "expected_population": declared["expected_population"]},
    ]
    try:
        exec_ops(chain)
        return {"name": "narrow_selector_still_redefines_downstream_all_via_exec",
                "guard_raised_by_exec": False,
                "verdict": "FAIL (exec_ops did not refuse the narrow-selector-induced downstream drift)"}
    except RefusalError as e:
        ok = (e.reason == REASON_POPULATION_DRIFT and e.diagnosis.get("op_id") == "d2"
              and e.diagnosis.get("history", {}).get("chain_position") == 3
              and e.diagnosis["expected_population"] == declared["expected_population"]
              and e.diagnosis["delta"] != 0)
        return {"name": "narrow_selector_still_redefines_downstream_all_via_exec",
                "guard_raised_by_exec": True, "reason": e.reason, "op_id": e.diagnosis.get("op_id"),
                "chain_position": e.diagnosis.get("history", {}).get("chain_position"),
                "expected_population": e.diagnosis["expected_population"],
                "actual_population": e.diagnosis["actual_population"], "delta": e.diagnosis["delta"],
                "verdict": "PASS" if ok else "FAIL (refused but diagnosis payload wrong)"}


def _case_pattern_multiplies_edges_then_fixed_forming_op_fails_via_exec() -> dict:
    """Same recipe as _kernel_repro_suite_v1._case_pattern_multiplies_edges_then_fixed_forming_op_fails
    (200x120x80 box, pattern_circular(count=3), then chamfer(selector=None) declaring the
    pre-pattern edge population) -- folded into ONE exec_ops(ops) call. The pattern op itself has NO
    guard (pattern_circular is not selector-addressable, per _ALL_CHAIN_ENTITY's own scope, see
    cad_op_exec_v1.py docstring) -- the guard fires at the FIRST selector=None chain position
    downstream of the multiplication, exactly the chamfer here."""
    fresh_ops = [
        {"id": "sb", "op": "sketch_2d", "shapes": [{"type": "rectangle", "width": 200.0, "height": 120.0}]},
        {"id": "d0", "op": "extrude", "sketch_ref": "sb", "amount": 80.0},
    ]
    fresh = exec_ops(fresh_ops)
    declared = record_expected_population(fresh["solids"]["d0"], "edge")

    chain = fresh_ops + [
        {"id": "m1", "op": "pattern_circular", "target_ref": "d0", "count": 3, "axis": "Z"},
        {"id": "m2", "op": "chamfer", "target_ref": "m1", "distance": 0.05,
         "expected_population": declared["expected_population"]},
    ]
    try:
        exec_ops(chain)
        return {"name": "pattern_multiplies_edges_then_fixed_forming_op_fails_via_exec",
                "guard_raised_by_exec": False,
                "verdict": "FAIL (exec_ops did not refuse the pattern-multiplied ALL population)"}
    except RefusalError as e:
        ok = (e.reason == REASON_POPULATION_DRIFT and e.diagnosis.get("op_id") == "m2"
              and e.diagnosis.get("history", {}).get("chain_position") == 3
              and e.diagnosis["expected_population"] == declared["expected_population"]
              and e.diagnosis["delta"] > 0)
        return {"name": "pattern_multiplies_edges_then_fixed_forming_op_fails_via_exec",
                "guard_raised_by_exec": True, "reason": e.reason, "op_id": e.diagnosis.get("op_id"),
                "chain_position": e.diagnosis.get("history", {}).get("chain_position"),
                "expected_population": e.diagnosis["expected_population"],
                "actual_population": e.diagnosis["actual_population"], "delta": e.diagnosis["delta"],
                "verdict": "PASS" if ok else "FAIL (refused but diagnosis payload wrong)"}


def _case_undeclared_legacy_never_raises_via_exec() -> dict:
    """Backward-compat planted-fault test: a legacy recipe's own fillet(selector=None) op with NO
    'expected_population' key must NOT raise (geometry proceeds unchanged, same box+fillet volume as
    pre-guard behaviour), only warn and flag in the log's population_guard row.
    """
    fresh_ops = [
        {"id": "sb", "op": "sketch_2d", "shapes": [{"type": "rectangle", "width": 300.0, "height": 300.0}]},
        {"id": "d0", "op": "extrude", "sketch_ref": "sb", "amount": 300.0},
    ]
    chain = fresh_ops + [{"id": "d1", "op": "fillet", "target_ref": "d0", "radius": 0.4}]
    out = exec_ops(chain)
    d1_row = next(r for r in out["log"] if r["op_id"] == "d1")
    pg = d1_row.get("population_guard") or {}
    expected_volume = 300.0 * 300.0 * 300.0  # fillet radius 0.4 removes a negligible sliver; check PASS+volume>0 only
    measured_volume = getattr(out["solids"]["d1"], "volume", None)
    ok = (d1_row["status"] == "PASS" and pg.get("declared") is False and pg.get("flag") is True
          and pg.get("population") == 12 and pg.get("auto_recorded_expected_population") == 12
          and measured_volume is not None and 0.0 < measured_volume < expected_volume)
    return {"name": "undeclared_legacy_never_raises_via_exec", "op_status": d1_row["status"],
            "population_guard": pg, "measured_volume_mm3": measured_volume,
            "verdict": "PASS" if ok else "FAIL"}


CASES = [
    _case_selector_none_chain_hazard_via_exec,
    _case_narrow_selector_still_redefines_downstream_all_via_exec,
    _case_pattern_multiplies_edges_then_fixed_forming_op_fails_via_exec,
    _case_undeclared_legacy_never_raises_via_exec,
]


def run() -> dict:
    results = [c() for c in CASES]
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    out = {"cell": "exec_populationsvakt_v1_fallbevis", "n_cases": len(results), "n_pass": n_pass,
           "all_pass": n_pass == len(results), "results": results}
    return out


if __name__ == "__main__":
    out = run()
    with open(REPORT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["all_pass"] else 1)
