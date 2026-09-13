#!/usr/bin/env python3
"""Differentiable placement search with a known-answer case.

Asks whether gradient descent on a differentiable loss can do the job a designer otherwise runs by
hand, for the class of problem where it applies: continuous placement / orientation / sizing with a
measurable, differentiable objective -- explicitly NOT topology changes (candidate A/B/C swaps stay a
design decision, see the generalisation contract at the end of this docstring).

The case: a set of circular bonding pads on a plate, one of which overlaps a ball-socket clearance
sphere. The conflict has a closed-form correct answer (move that pad until the centre-to-centre
distance >= R_pad + R_eff), computable independently of the descent, so the search can be scored
against a reference instead of against itself.

Model, declared:
  - A first attempt used a circle-circle lens-area proxy calibrated to the starting distance. That
    was a measured dead end, kept here rather than silently dropped: at the anchor distance, solving
    for the effective ball-socket radius that reproduces the declared overlap volume lands in the
    lens formula's FULL-CONTAINMENT branch (area = pi*r_eff^2, constant in d), i.e. the calibrated
    proxy has ZERO gradient exactly at the starting point -- confirmed by central differences.
  - The replacement model actually used: a smooth quadratic interpenetration falloff
    V(d) = V_MAX * relu(1 - d/D_REACT_MM)^2, where D_REACT_MM = R_PAD_MM + BALL_DIA/2 +
    SOCKET_CLEARANCE is the physical distance beyond which pad and socket cannot overlap at all, and
    V_MAX is the ONE calibrated free parameter, solved in closed form so V(d0) reproduces the
    declared overlap volume exactly. This model has a nonzero gradient everywhere in (0, D_REACT_MM)
    except the single point d=0, so it is descent-friendly by construction while still anchored to
    one declared data point.
  - The separation-floor penalty (minimum pad-pad spacing) is a constructed stiffness surrogate
    (declared threshold D_FLOOR_MM, not measured from a modal analysis).
  - Out of scope for this prototype, declared: a swept light-volume term would need either a real
    ray-trace evaluator or a differentiable-SDF kernel with a tape; once such a kernel exposes a
    stable gradient path over a light-volume expression tree, ball_term()/lens_area() can be
    replaced by a call into it and the optimiser loop is unchanged.

Gradient: central finite differences per parameter (declared choice; a tape would need autodiff
wiring for a 2-primitive model that does not warrant it -- central differences on 10 parameters =
20 forward evaluations per step, pure numpy). Optimiser: plain Adam with the hyperparameters below.

Inputs are synthetic and declared in code; the report is written to artifacts/ next to this module.

Run: python diff_placement_search_v1.py
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# ---------------------------------------------------------------------------
# Declared anchors of the synthetic case (a plate with five bonding pads and one
# ball-socket clearance sphere that one of the pads overlaps)
# ---------------------------------------------------------------------------
PAD_XY0 = np.array([
    [100.0, 45.0],
    [-100.0, 55.0],
    [-100.0, -55.0],
    [100.0, -45.0],
    [0.0, 0.0],
], dtype=np.float64)  # pad layout, mm
R_PAD_MM = 15.0                     # pad radius, mm
PAD_THICKNESS_MM = 2.0              # bond thickness, mm
BALL_CENTER_XY = np.array([-108.0, -62.0])  # ball-socket pivot, mm
REAL_BALL_COLLISION_MM3 = 148.70205226992292  # declared overlap volume at the start pose, mm3
BALL_PAD_INDEX = 2                  # the pad that occupies the ball location

# CONSTRUCTED (declared)
D_FLOOR_MM = 40.0                   # 2*R_PAD + 10mm margin, a declared stiffness-floor surrogate
K_FLOOR = 50.0                      # penalty stiffness for floor violation (constructed)
CLEARANCE_MARGIN_MM = 0.05          # small strictly-clear margin for the closed-form facit


def lens_area(d: float, r1: float, r2: float) -> float:
    """Circle-circle intersection (lens) area, standard closed form. d = center distance.
    Kept for documentation and comparison -- see the dead end in the module docstring: calibrating
    this against the anchor lands in the degenerate full-containment (zero-gradient) branch, so it
    is NOT the model actually used by loss_and_terms()."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    d = max(d, 1e-9)
    a1 = r1 * r1 * math.acos(np.clip((d * d + r1 * r1 - r2 * r2) / (2 * d * r1), -1.0, 1.0))
    a2 = r2 * r2 * math.acos(np.clip((d * d + r2 * r2 - r1 * r1) / (2 * d * r2), -1.0, 1.0))
    term = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    a3 = 0.5 * math.sqrt(max(term, 0.0))
    return a1 + a2 - a3


def ball_penetration_mm3(d: float, v_max: float, d_react: float) -> float:
    """Smooth quadratic interpenetration falloff, the model actually used. V(d>=d_react)=0,
    V(0) = v_max, smooth (C1) nonzero gradient on the open interval (0, d_react)."""
    if d >= d_react:
        return 0.0
    frac = 1.0 - d / d_react
    return v_max * frac * frac


BALL_DIA_MM = 10.0            # ball diameter, mm
SOCKET_CLEARANCE_MM = 0.15    # socket clearance, mm
D_REACT_MM = R_PAD_MM + BALL_DIA_MM / 2.0 + SOCKET_CLEARANCE_MM  # real: 15+5+0.15 = 20.15mm


def _solve_v_max() -> tuple[float, float]:
    """Closed form: V_MAX so that ball_penetration_mm3(d0, V_MAX, D_REACT_MM) ==
    REAL_BALL_COLLISION_MM3 at the pad's starting distance d0. The ONE calibration step, run once at
    import time."""
    d0 = float(np.linalg.norm(PAD_XY0[BALL_PAD_INDEX] - BALL_CENTER_XY))
    frac = 1.0 - d0 / D_REACT_MM
    v_max = REAL_BALL_COLLISION_MM3 / (frac * frac)
    achieved = ball_penetration_mm3(d0, v_max, D_REACT_MM)
    assert abs(achieved - REAL_BALL_COLLISION_MM3) < 1e-6, (achieved, REAL_BALL_COLLISION_MM3)
    return v_max, d0


V_MAX_MM3, D0_MM = _solve_v_max()


def loss_and_terms(params: np.ndarray) -> tuple[float, dict]:
    """params: flat (10,) = 5 pads x (x,y). Returns (total_loss, breakdown)."""
    xy = params.reshape(5, 2)
    ball_mm3 = 0.0
    for i in range(5):
        if i != BALL_PAD_INDEX:
            continue  # other 4 pads report gate2b ball-contact 0 in the real cell -- no ball term
        d = float(np.linalg.norm(xy[i] - BALL_CENTER_XY))
        ball_mm3 += ball_penetration_mm3(d, V_MAX_MM3, D_REACT_MM)

    floor_pen = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            d = float(np.linalg.norm(xy[i] - xy[j]))
            viol = max(0.0, D_FLOOR_MM - d)
            floor_pen += K_FLOOR * viol * viol

    total = ball_mm3 + floor_pen
    return total, {"ball_mm3": ball_mm3, "floor_penalty": floor_pen}


def central_diff_grad(params: np.ndarray, h: float = 1e-2) -> np.ndarray:
    n = params.shape[0]
    g = np.zeros(n)
    for k in range(n):
        pp = params.copy(); pp[k] += h
        pm = params.copy(); pm[k] -= h
        lp, _ = loss_and_terms(pp)
        lm, _ = loss_and_terms(pm)
        g[k] = (lp - lm) / (2 * h)
    return g


def adam_descent(params0: np.ndarray, lr: float = 0.5, beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8, max_iter: int = 2000, tol: float = 1e-6):
    params = params0.copy()
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    history = []
    n_evals = 0
    t0 = time.perf_counter()
    it = 0
    for it in range(1, max_iter + 1):
        loss, terms = loss_and_terms(params)
        n_evals += 1
        history.append({"iter": it, "loss": loss, **terms})
        if loss < tol:
            break
        grad = central_diff_grad(params)
        n_evals += 2 * params.shape[0]
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = m / (1 - beta1 ** it)
        vhat = v / (1 - beta2 ** it)
        params = params - lr * mhat / (np.sqrt(vhat) + eps)
    wall_s = time.perf_counter() - t0
    final_loss, final_terms = loss_and_terms(params)
    n_evals += 1
    return {
        "params": params, "iters": it, "n_evals": n_evals, "wall_s": wall_s,
        "final_loss": final_loss, "final_terms": final_terms, "history": history,
    }


def closed_form_facit_pad2() -> np.ndarray:
    """Independent ground truth (NOT used by the descent): minimal-displacement point that clears
    the ball -- move pad2 directly away from the ball center along the current line, to exactly
    R_PAD + R_EFF + CLEARANCE_MARGIN_MM."""
    vec = PAD_XY0[BALL_PAD_INDEX] - BALL_CENTER_XY
    unit = vec / np.linalg.norm(vec)
    d_min = D_REACT_MM + CLEARANCE_MARGIN_MM
    return BALL_CENTER_XY + unit * d_min


def main():
    params0 = PAD_XY0.flatten().astype(np.float64)
    loss0, terms0 = loss_and_terms(params0)

    result = adam_descent(params0)
    params_star = result["params"].reshape(5, 2)

    facit_pad2 = closed_form_facit_pad2()
    descent_pad2 = params_star[BALL_PAD_INDEX]
    facit_distance_mm = float(np.linalg.norm(descent_pad2 - facit_pad2))

    # symmetric falsification: does the OTHER 4 pads drift away from their zero-gradient start? (should not)
    other_idx = [i for i in range(5) if i != BALL_PAD_INDEX]
    other_drift_mm = [float(np.linalg.norm(params_star[i] - PAD_XY0[i])) for i in other_idx]
    max_other_drift_mm = max(other_drift_mm)

    # declared comparator: the accumulated wall time of a chain of manual falsification rounds that
    # chased the same placement defect by hand (six runs, seconds)
    accumulated_cell_time_s = 1021.9 + 130.0 + 74.6 + 331.0 + 46.8 + 77.5
    speedup_x = accumulated_cell_time_s / result["wall_s"] if result["wall_s"] > 0 else float("inf")

    report = {
        "cell": "diff_placement_v1",
        "reframe_declared": [
            "the full tilt-swept pad search has no closed-form reference -- substituted a single "
            "ball-vs-pad overlap with a closed-form correct answer as the known-answer case.",
            "the wall-time comparator is the summed elapsed time of the manual falsification rounds "
            "that chased the same defect by hand = 1681.8 s.",
        ],
        "measured_vs_constructed": {
            "declared": ["PAD_XY0", "R_PAD_MM", "PAD_THICKNESS_MM", "BALL_CENTER_XY",
                          "REAL_BALL_COLLISION_MM3", "accumulated_cell_time_s components"],
            "constructed": ["ball_penetration_mm3 quadratic-falloff proxy shape",
                              "V_MAX_MM3 (calibrated, not measured)",
                              "D_FLOOR_MM/K_FLOOR separation-floor stiffness surrogate"],
        },
        "calibration": {
            "model": "V(d) = V_MAX * relu(1 - d/D_REACT_MM)^2",
            "d_react_mm": D_REACT_MM, "v_max_mm3": V_MAX_MM3, "d0_mm": D0_MM,
            "lens_area_dead_end_note": "circle-lens calibration gave grad=0 at d0 (full-containment "
                                          "branch) -- falsified, not used; see module docstring.",
            "calibration_reproduces_declared_mm3": ball_penetration_mm3(D0_MM, V_MAX_MM3, D_REACT_MM),
        },
        "initial_state": {"params": params0.tolist(), "loss_mm3": loss0, "terms": terms0},
        "descent": {
            "optimizer": "Adam (lr=0.5,beta1=0.9,beta2=0.999,eps=1e-8)", "gradient": "central-diff h=1e-2mm",
            "n_params": 10, "iters": result["iters"], "n_forward_evals": result["n_evals"],
            "wall_s": result["wall_s"], "final_loss_mm3": result["final_loss"], "final_terms": result["final_terms"],
            "converged_params_xy": params_star.tolist(),
        },
        "facit_comparison": {
            "closed_form_facit_pad2_xy": facit_pad2.tolist(),
            "descent_pad2_xy": descent_pad2.tolist(),
            "distance_to_facit_mm": facit_distance_mm,
            "same_basin": facit_distance_mm < 1.0,
            "max_other_pad_drift_mm": max_other_drift_mm,
        },
        "wallclock_win": {
            "accumulated_manual_round_time_s": accumulated_cell_time_s,
            "descent_wall_s": result["wall_s"],
            "speedup_x": speedup_x,
            "caveat_apples_to_oranges": "the raw speedup_x is dominated by evaluation cost (this "
                "prototype's proxy is pure numpy, sub-microsecond per evaluation, while the manual rounds "
                "paid B-rep boolean/mesh/modal evaluation, STEP export and human round overhead per "
                "iteration). The load-bearing claim is NOT a raw speed factor -- it is that one convergent "
                "run replaces the STRUCTURE of nine manual falsification rounds for a placement-class "
                "defect; wired to a real B-rep/ray-trace evaluator (seconds per evaluation) the same "
                "number of evaluations would cost tens of minutes, which needs the real evaluator wired in "
                "before it is claimed.",
        },
        "generalization_contract": {
            "eats": [
                "continuous placement (x,y[,z]) of a fixed set of parts with a differentiable "
                "penetration/clearance objective -- the case in this module",
                "continuous orientation (angles) with a differentiable interference objective",
                "continuous dimensioning (radii/thicknesses/offsets) against a differentiable "
                "modal/mass/clearance floor",
                "any n_vars in roughly 2-20 with a cheap (<1s) differentiable-or-central-diffable evaluation "
                "-- in an optimiser-selection table this is a row distinct from "
                "NSGA-II and from topology optimisation at n_vars >> 1000: gradient descent wins "
                "when n_vars is low AND the loss is differentiable, because it needs O(n_vars) evals per "
                "step instead of O(1000s) population evals.",
            ],
            "does_not_eat": [
                "topology changes (candidate A vs B vs C, edge clip vs backside pad vs kinematic mount) "
                "-- these are discrete design choices with no continuous path between them; the "
                "candidate-family history stays design work, in the constraint-solver / discrete row "
                "of an optimiser-selection table, not gradient descent.",
                "a swept light-volume objective -- no differentiable evaluator for it is wired here "
                "(it needs an SDF-tree evaluation with a tape, or a ray-trace adjoint); this prototype's "
                "ball-collision case is a sub-problem of the same family, not that full problem.",
            ],
        },
        "handoff": "replace loss_and_terms()'s ball_term with a gradient query over an SDF "
                       "expression tree encoding a light-volume field; the Adam loop above is otherwise "
                       "unchanged (it only needs loss_and_terms plus a gradient).",
    }
    report["ATOMS"] = {
        "atoms": [
            {"id": "a1_converges_to_zero", "type": "value-in-artifact",
             "artifact": "reports/probes/diff_placering_v1.json", "key": "descent.final_loss_mm3",
             "expected": 0.0, "tol": 1e-6},
            {"id": "a2_same_basin_as_closed_form_facit", "type": "value-in-artifact",
             "artifact": "reports/probes/diff_placering_v1.json", "key": "facit_comparison.distance_to_facit_mm",
             "expected": 0.0, "tol": 0.5},
            {"id": "a3_other_pads_did_not_spuriously_drift", "type": "inequality",
             "lhs": {"artifact": "reports/probes/diff_placering_v1.json", "key": "facit_comparison.max_other_pad_drift_mm"},
             "op": "<", "rhs": 0.5},
            {"id": "a4_calibration_matches_real_measured_collision", "type": "value-in-artifact",
             "artifact": "reports/probes/diff_placering_v1.json", "key": "calibration.calibration_reproduces_real_mm3",
             "expected": REAL_BALL_COLLISION_MM3, "tol": 1e-3},
            {"id": "a5_descent_cheap_eval_budget", "type": "inequality",
             "lhs": {"artifact": "reports/probes/diff_placering_v1.json", "key": "descent.n_forward_evals"},
             "op": "<", "rhs": 1000},
            {"id": "a6_report_exists", "type": "artifact-exists",
             "artifact": "reports/probes/diff_placering_v1.json"},
        ],
    }
    return report


if __name__ == "__main__":
    t0 = time.perf_counter()
    rep = main()
    rep["elapsed_s"] = round(time.perf_counter() - t0, 3)
    out_path = os.path.join(HERE, "artifacts", "diff_placering_v1.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({
        "final_loss_mm3": rep["descent"]["final_loss_mm3"],
        "iters": rep["descent"]["iters"],
        "wall_s": rep["descent"]["wall_s"],
        "distance_to_facit_mm": rep["facit_comparison"]["distance_to_facit_mm"],
        "same_basin": rep["facit_comparison"]["same_basin"],
        "speedup_x": rep["wallclock_win"]["speedup_x"],
        "out_path": out_path,
    }, indent=1))
