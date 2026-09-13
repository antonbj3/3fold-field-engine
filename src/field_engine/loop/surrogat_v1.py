#!/usr/bin/env python3
"""Small Gaussian-process surrogate for the part_harness_v1 contract (build/objectives/constraints).

Contract-independent: the same code drives any part module exposing BOUNDS, OBJECTIVE_NAMES,
CONSTRAINT_NAMES, build, objectives and constraints. The surrogate never returns a verdict; it proposes
the next theta and a real build plus gate decides.

Method: standard constrained Bayesian optimisation. One GaussianProcessRegressor (Matern 5/2 plus a
white kernel) per objective and per constraint;
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
for the objective, weighted by P(feasible)(x) = prod_c Phi(-mu_c(x)/sigma_c(x)) over the constraints
(independence assumed, declared). The candidate is chosen by dense random sampling over BOUNDS: the
space is low-dimensional, so this is sufficient and is deterministic given a seed.

Calibration is the gate on the surrogate itself: every proposed point is predicted (mu, sigma) before
the real build runs on it and before it enters the training set, so each point is a genuine
out-of-sample test. calibration_report() pools those (pred, actual, sigma) triples and computes the
empirical coverage against 1 and 2 sigma; a calibrated surrogate covers about 68% and 95%. Shifting y
by a constant after training must make that coverage collapse.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.abspath(__file__))


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def load_part_module(path: str):
    """Loads a part module from a file path as a standalone module and returns it."""
    abspath = os.path.join(ROOT, path) if not os.path.isabs(path) else path
    name = os.path.splitext(os.path.basename(abspath))[0]
    spec = importlib.util.spec_from_file_location(name, abspath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def real_eval(mod, theta: Dict[str, float]) -> Dict[str, Any]:
    """One real build()+objectives()+constraints() call, the only source of truth.

    Takes the part module and a theta dict; returns a flat record with theta, objectives, constraints,
    feasible, wall_s and error.
    """
    t0 = time.time()
    rec = {"theta": dict(theta)}
    try:
        shape = mod.build(theta)
        obj = mod.objectives(shape, theta)
        con = mod.constraints(shape, theta) if mod.CONSTRAINT_NAMES else {}
        feasible = all(v <= 0.0 for v in con.values()) if con else True
        rec.update({"objectives": obj, "constraints": con, "feasible": feasible, "error": None})
    except Exception as e:
        rec.update({"objectives": None, "constraints": None, "feasible": False, "error": str(e)})
    rec["wall_s"] = time.time() - t0
    return rec


def append_jsonl(path: str, rec: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# GP surrogate: one model per output (objectives and constraints), one kernel class.
# ---------------------------------------------------------------------------

class GPBank:
    """One GaussianProcessRegressor per output name; X is normalised to [0,1]^d through the bounds."""

    def __init__(self, bounds: Dict[str, Tuple[float, float]], output_names: List[str], seed: int = 0):
        self.param_names = list(bounds.keys())
        self.bounds = bounds
        self.output_names = list(output_names)
        self.seed = seed
        self.models: Dict[str, Any] = {}
        self.n_train = 0

    def _norm(self, theta: Dict[str, float]) -> np.ndarray:
        lo = np.array([self.bounds[k][0] for k in self.param_names])
        hi = np.array([self.bounds[k][1] for k in self.param_names])
        x = np.array([theta[k] for k in self.param_names], dtype=float)
        return (x - lo) / (hi - lo)

    def fit(self, rows: List[Dict[str, Any]]):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

        ok = [r for r in rows if r.get("error") is None]
        self.n_train = len(ok)
        if self.n_train < 2:
            self.models = {}
            return
        X = np.array([self._norm(r["theta"]) for r in ok])
        for name in self.output_names:
            y = np.array([_get_output(r, name) for r in ok], dtype=float)
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.3, nu=2.5,
                                     length_scale_bounds=(0.05, 3.0)) + WhiteKernel(1e-6, (1e-10, 1.0))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                                          random_state=self.seed)
            gp.fit(X, y)
            self.models[name] = gp

    def predict(self, theta: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Returns {output_name: (mu, sigma)}.

        Extrapolates beyond [0,1]^d when theta is outside the bounds: the kernel distance grows, so
        sigma grows with it. There is no special extrapolation code.
        """
        if not self.models:
            return {name: (0.0, 1e6) for name in self.output_names}
        x = self._norm(theta).reshape(1, -1)
        out = {}
        for name, gp in self.models.items():
            mu, sigma = gp.predict(x, return_std=True)
            out[name] = (float(mu[0]), float(max(sigma[0], 1e-9)))
        return out


def _get_output(row: Dict[str, Any], name: str) -> float:
    if row.get("objectives") and name in row["objectives"]:
        return row["objectives"][name]
    if row.get("constraints") and name in row["constraints"]:
        return row["constraints"][name]
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Acquisition function: constrained EI -- EI(objective) * P(all constraints <= 0)
# ---------------------------------------------------------------------------

def prob_feasible(con_pred: Dict[str, Tuple[float, float]]) -> float:
    p = 1.0
    for _, (mu, sigma) in con_pred.items():
        p *= float(norm.cdf(-mu / sigma))
    return p


def expected_improvement(mu: float, sigma: float, f_best: float) -> float:
    if sigma < 1e-12:
        return max(f_best - mu, 0.0)
    z = (f_best - mu) / sigma
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def propose_candidate(obj_bank: GPBank, con_bank: GPBank, bounds: Dict[str, Tuple[float, float]],
                      primary_objective: str, f_best: float, n_candidates: int = 4000,
                      rng: np.random.Generator = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Dense random sampling over BOUNDS ranked by P(feasible) * EI(primary objective).

    Takes the two GP banks, the bounds, the primary objective name, the incumbent best value, the
    candidate count and an rng. Returns (theta, debug dict with the best acquisition and its mu/sigma).
    """
    rng = rng or np.random.default_rng(0)
    names = list(bounds.keys())
    lo = np.array([bounds[k][0] for k in names])
    hi = np.array([bounds[k][1] for k in names])
    cand = lo + rng.random((n_candidates, len(names))) * (hi - lo)
    best_acq, best_theta, best_dbg = -1.0, None, None
    for row in cand:
        theta = {k: float(v) for k, v in zip(names, row)}
        con_pred = con_bank.predict(theta) if con_bank.output_names else {}
        pf = prob_feasible(con_pred) if con_pred else 1.0
        obj_pred = obj_bank.predict(theta)
        mu, sigma = obj_pred[primary_objective]
        ei = expected_improvement(mu, sigma, f_best)
        acq = ei * pf
        if acq > best_acq:
            best_acq, best_theta, best_dbg = acq, theta, {
                "acq": acq, "ei": ei, "p_feasible": pf, "obj_mu": mu, "obj_sigma": sigma,
                "con_pred": {k: v for k, v in con_pred.items()},
            }
    return best_theta, best_dbg


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_report(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Coverage report over a list of {"pred_mu", "pred_sigma", "actual"} records.

    Coverage is the fraction with |actual - mu| <= k*sigma for k = 1, 2; a calibrated GP gives about
    68% and 95%. Returns the counts, both coverages and the mean |z|, error and sigma.
    """
    if not preds:
        return {"n": 0}
    err = np.array([abs(p["actual"] - p["pred_mu"]) for p in preds])
    sig = np.array([max(p["pred_sigma"], 1e-9) for p in preds])
    z = err / sig
    return {
        "n": len(preds),
        "coverage_1sigma": float(np.mean(z <= 1.0)),
        "coverage_2sigma": float(np.mean(z <= 2.0)),
        "mean_abs_z": float(np.mean(z)),
        "mean_abs_err": float(np.mean(err)),
        "mean_sigma": float(np.mean(sig)),
    }
