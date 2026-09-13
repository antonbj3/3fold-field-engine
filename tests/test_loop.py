"""Tests for the recipe executor, the part harness contract and the surrogate."""
import json
import os
import random
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOP = os.path.join(ROOT, "src", "field_engine", "loop")
PARTS = os.path.join(ROOT, "examples", "parts")
for p in (LOOP, PARTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import bracket_part_v1 as BP  # noqa: E402
import surrogat_v1 as SUR  # noqa: E402


def test_recipe_builds_and_holes_are_cut():
    theta = {"plate_w": 150.0, "plate_h": 100.0, "thk": 10.0, "hole_d": 9.0}
    part = BP.build(theta)
    obj = BP.objectives(part, theta)
    # the slot and four holes must remove material against the plain slab
    slab = theta["plate_w"] * theta["plate_h"] * theta["thk"]
    assert 0.5 * slab < obj["volume_mm3"] < slab
    assert BP.constraints(part, theta)["deflection_margin"] < 0.0


def test_recipe_rejects_unknown_op():
    import recept_exec_v1 as RX
    bad = {"schema": "recept_v1", "part": "x", "params": {},
           "steps": [{"op": "teleport", "amount": 1}]}
    try:
        RX.execute(bad, {}, {})
    except ValueError as e:
        assert "unknown ops" in str(e)
    else:
        raise AssertionError("an unknown op must be rejected")


def test_part_harness_parallel_batch():
    out = subprocess.run([sys.executable, os.path.join(LOOP, "part_harness_v1.py"),
                          "bracket_part_v1", "--n", "4", "--workers", "2"],
                         capture_output=True, text=True, timeout=600,
                         env={**os.environ, "PYTHONPATH": f"{PARTS}:{LOOP}"})
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    rows = [json.loads(l) for l in out.stdout.strip().splitlines()]
    assert len(rows) == 4
    assert all(r["error"] is None and r["objectives"]["volume_mm3"] > 0 for r in rows)


def _synthetic_rows(n, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        theta = {"a": float(rng.uniform(0.0, 1.0)), "b": float(rng.uniform(0.0, 1.0))}
        y = (theta["a"] - 0.3) ** 2 + 0.5 * (theta["b"] - 0.7) ** 2 + shift
        rows.append({"theta": theta, "objectives": {"y": y}, "constraints": {}, "error": None})
    return rows


def test_surrogate_is_calibrated_and_poisoning_breaks_it():
    bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
    train = _synthetic_rows(24, seed=1)
    test = _synthetic_rows(24, seed=2)
    bank = SUR.GPBank(bounds, ["y"], seed=0)
    bank.fit(train)
    preds = []
    for r in test:
        mu, sigma = bank.predict(r["theta"])["y"]
        preds.append({"pred_mu": mu, "pred_sigma": sigma, "actual": r["objectives"]["y"]})
    rep = SUR.calibration_report(preds)
    assert rep["coverage_2sigma"] >= 0.8

    poisoned = [{**p, "actual": p["actual"] + 10.0} for p in preds]
    rep_bad = SUR.calibration_report(poisoned)
    assert rep_bad["coverage_2sigma"] < rep["coverage_2sigma"]


def test_surrogate_sigma_grows_out_of_distribution():
    bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
    bank = SUR.GPBank(bounds, ["y"], seed=0)
    bank.fit(_synthetic_rows(24, seed=3))
    _mu_in, sigma_in = bank.predict({"a": 0.5, "b": 0.5})["y"]
    _mu_out, sigma_out = bank.predict({"a": 6.0, "b": 6.0})["y"]
    assert sigma_out > 5.0 * sigma_in


def test_expected_improvement_and_feasibility_weighting():
    assert SUR.expected_improvement(mu=1.0, sigma=0.0, f_best=2.0) == 1.0
    p = SUR.prob_feasible({"c": (-2.0, 0.5)})
    assert p > 0.99
    p_bad = SUR.prob_feasible({"c": (2.0, 0.5)})
    assert p_bad < 0.01
