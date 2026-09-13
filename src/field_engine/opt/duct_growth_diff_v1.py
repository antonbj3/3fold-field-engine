"""Differentiable duct growth against pressure drop.

A duct is parameterised differentiably -- one radius r_i per control point along the route
arclength -- and descended toward low pressure drop under two constraints: a minimum cross-section
(radius >= the task's minimum bend-radius floor) and no intrusion into the obstacles, measured with
the IKARUS SDF primitives (`ikarus_v1/expr.py`, reused, not rebuilt).

Gradient path: the lattice-Boltzmann solve is too expensive for every descent step (a fine run costs
tens of seconds; 100+ steps x 2*N_CTRL central-difference evaluations would cost hours). Chosen
instead: a differentiable SURROGATE -- the per-bin Darcy-Weisbach sum the LBM judge itself uses as
its reference, PLUS the bend-loss term the judge measures as missing from it. That term is
calibrated as a dimensionless K_BEND minor-loss coefficient against the judge's own bent-duct run
(read from its artifact, not guessed), the descent runs on the surrogate, and a real LBM run
(reusing lbm_domare_v1's run_lbm/lbm_step/unit_scale, not a new solver) judges before and after
every case: the descent proposes, the judge decides.

Declared: K_BEND is measured at the LBM's own creeping-flow operating point and applied here at the
declared design-flow regime as a dimensionless minor-loss coefficient (K*0.5*rho*v^2, the standard
duct-engineering form, approximately Re-independent for a fixed bend-geometry ratio). That reuse is
a constructed modelling choice, not a measurement in the new regime.

The LBM judge here (lbm_judge_two_leg) is a new geometry wrapper around the imported solver
primitives: each leg's cross-section is a free parameter (the per-leg mean radius of the descended
profile), a declared per-leg-uniform proxy of the continuous radius profile.

Cases: one reference L-bend (the same topology and cross-section as the judge's own validated
bent-duct calibration) plus three tasks from the duct corpus (data/corpus/duct_v1, written by
duct_task_gen_v1.py): real obstacle sets, real flange/mouth ports.

Run (one stage per invocation):
  python duct_growth_diff_v1.py <stage>   stage in {selftest, descend, judge_before, judge_after, report, all}
Requires the judge's artifact: run `python ../lbm_domare_v1.py calib` first.
Alternatively set FIELD_ENGINE_LBM_CALIBRATION to an explicit reusable cache file.
Its first use runs the full calibration; unchanged source, inputs and runtime
reuse it across work directories. Candidate descent and LBM judging still run.
Partials and the folded report go to artifacts/ next to this file.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD_ENGINE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(FIELD_ENGINE))
ARTIFACTS = os.path.join(HERE, "artifacts")
SCRATCH = os.path.join(ARTIFACTS, "duct_growth_diff_v1")
os.makedirs(SCRATCH, exist_ok=True)
RPT = os.path.join(ARTIFACTS, "duct_growth_diff_v1.json")

sys.path.insert(0, FIELD_ENGINE)
sys.path.insert(0, os.path.join(FIELD_ENGINE, "ikarus_v1"))
import lbm_domare_v1 as LD          # noqa: E402  -- reused solver primitives (run_lbm, lbm_step, unit_scale, ...)
import expr as IKE                  # noqa: E402  -- ikarus SDF primitives (intrusion measurement)

LBM_REPORT = os.path.join(FIELD_ENGINE, "artifacts", "lbm_domare_v1.json")
LBM_CALIB_PARTIAL = os.path.join(FIELD_ENGINE, "artifacts", "lbm_domare_v1", "calib.json")

RHO_AIR, MU_AIR = LD.RHO_AIR, LD.MU_AIR
Q_BRANCH_M3S = LD.Q_BRANCH_M3S          # the SAME declared branch flow lbm_domare_v1 uses
F_RE_CIRCULAR = LD.F_RE_CIRCULAR

CORPUS_DIR = os.environ.get("FIELD_ENGINE_DUCT_CORPUS", os.path.join(REPO, "data", "corpus", "duct_v1"))
SYNTH_TASK_IDS = ["t20260730_0000_00000000", "t20260730_0001_00000000", "t20260730_0002_00000000"]
MIN_BEND_R_FLOOR_MM = 24.0     # declared floor, the same constant duct_task_gen_v1.py bands its tasks around

N_CTRL = 6                    # control points per route (low-dimensional differentiable case, <= 20 variables)


# ══════════════════════════════ CALIBRATION: K_BEND from the real LBM measurement (read, not guessed) ═══

def load_k_bend():
    """K_BEND from the LBM judge's own bent-duct measurement (read, not guessed).

    Reads the folded report if it exists, otherwise the calibration partial; both are written by
    lbm_domare_v1.py, which must be run before this module."""
    cache = os.environ.get("FIELD_ENGINE_LBM_CALIBRATION")
    if cache:
        bent = LD.cached_calibration(cache)["bent"]
        source = "explicit source-bound LBM calibration:bent"
    elif os.path.exists(LBM_REPORT):
        bent = json.load(open(LBM_REPORT))["stage1_calibration"]["bent_duct_L90"]
        source = "artifacts/lbm_domare_v1.json:stage1_calibration.bent_duct_L90"
    elif os.path.exists(LBM_CALIB_PARTIAL):
        bent = json.load(open(LBM_CALIB_PARTIAL))["bent"]
        source = "artifacts/lbm_domare_v1/calib.json:bent"
    else:
        raise SystemExit("no LBM judge artifact found; run `python ../lbm_domare_v1.py calib` first "
                          f"(looked for {LBM_REPORT} and {LBM_CALIB_PARTIAL})")
    excess_pa = bent["excess_over_formula_pa"]           # MEASURED, LBM's own matched operating point
    re_matched = bent["Re_matched"]
    cross_mm = bent["cross_section_mm"]                  # [120,120]
    dh_m = math.sqrt(cross_mm[0] * cross_mm[1]) * 1e-3    # square duct: dh = side
    v_mean_matched = re_matched * MU_AIR / (RHO_AIR * dh_m)
    k_bend = excess_pa / (0.5 * RHO_AIR * v_mean_matched ** 2)
    return dict(k_bend=float(k_bend), excess_pa=excess_pa, re_matched=re_matched,
                v_mean_matched_lbm_case=float(v_mean_matched), dh_m_lbm_case=dh_m, source=source)


# ══════════════════════════════ CASE DEFINITIONS (route + obstacles + constraints) ═══════════════════════

def reference_case_L90():
    """The reference case: the same topology and cross-section as lbm_domare_v1's own validated
    bent-duct calibration (two 90 mm legs, 120x120 mm nominal cross-section, one 90deg corner). No
    obstacles -- that calibration geometry carries none; the minimum-radius floor is the declared
    MIN_BEND_R_FLOOR_MM."""
    leg_mm = 90.0
    p0 = np.array([0.0, 0.0, 0.0])
    corner = np.array([leg_mm, 0.0, 0.0])
    p1 = np.array([leg_mm, leg_mm, 0.0])
    min_r_mm = MIN_BEND_R_FLOOR_MM
    return dict(case_id="reference_L90", path_pts=[p0, corner, p1], obstacle_node=None,
                min_r_mm=min_r_mm, r_init_mm=60.0, Q_m3s=Q_BRANCH_M3S,
                leg1_mm=leg_mm, leg2_mm=leg_mm, source="lbm_domare_v1.py bent_duct_case topology")


AX_NAME = {0: "x", 1: "y", 2: "z"}


def _obstacle_node(obstacles):
    if not obstacles:
        return None
    nodes = []
    for ob in obstacles:
        c = tuple(ob["center_mm"])
        if ob["type"] == "box":
            s = ob["size_mm"]
            nodes.append(IKE.box(half_extents=(s[0] / 2, s[1] / 2, s[2] / 2), center=c))
        elif ob["type"] == "cylinder":
            nodes.append(IKE.cylinder(radius=ob["r_mm"], height=ob["height_mm"], axis=AX_NAME[ob["axis"]], center=c))
        else:  # cone (tapered cylinder, r0->r1)
            nodes.append(IKE.cone(radius1=ob["r0_mm"], radius2=ob["r1_mm"], height=ob["height_mm"],
                                   axis=AX_NAME[ob["axis"]], center=c))
    node = nodes[0]
    for n in nodes[1:]:
        node = IKE.union(node, n, k=0.0)
    return node


def synth_case(task_id):
    """A real duct_task_gen_v1 corpus task: real obstacles/ports, L-shaped route from flange_in through
    a single elbow to the mouth port (interior offset by a standoff so the route starts/ends inside the
    box, not glued to the wall plane)."""
    t = json.load(open(f"{CORPUS_DIR}/{task_id}/task.json"))
    p_in = np.array(t["ports"]["flange_in"]["center_mm"])
    n_in = np.array(t["ports"]["flange_in"]["normal"])
    p_out = np.array(t["ports"]["mynning_out"]["center_mm"])
    n_out = np.array(t["ports"]["mynning_out"]["normal"])
    standoff = 40.0
    p_in_i = p_in - n_in * standoff       # interior direction = -normal (normal points OUTWARD, task_gen convention)
    p_out_i = p_out - n_out * standoff
    ax_in = int(t["ports"]["flange_in"]["axis"])
    corner = p_in_i.copy()
    corner[ax_in] = p_out_i[ax_in]        # leg1: moves along ax_in only; leg2: corner -> p_out_i (remaining axes)
    leg1_mm = float(np.linalg.norm(corner - p_in_i))
    leg2_mm = float(np.linalg.norm(p_out_i - corner))
    min_r_mm = float(t["min_bend_r_floor_mm"])
    flange_half = t["ports"]["flange_in"]["size_mm"][0] / 2.0
    mynning_half = math.sqrt(t["ports"]["mynning_out"]["area_mm2"]) / 2.0
    r_init_mm = max(min_r_mm * 1.2, 0.5 * (flange_half + mynning_half))
    return dict(case_id=task_id, path_pts=[p_in_i, corner, p_out_i],
                obstacle_node=_obstacle_node(t["obstacles"]), min_r_mm=min_r_mm, r_init_mm=r_init_mm,
                Q_m3s=Q_BRANCH_M3S, leg1_mm=leg1_mm, leg2_mm=leg2_mm, source=f"data/corpus/duct_v1/{task_id}")


def control_points(case):
    """N_CTRL points evenly spaced by arclength along the 2-segment polyline; returns (pts(N,3), s_frac(N))."""
    p0, pc, p1 = case["path_pts"]
    l1 = float(np.linalg.norm(pc - p0))
    l2 = float(np.linalg.norm(p1 - pc))
    total = l1 + l2
    s = np.linspace(0.0, total, N_CTRL)
    pts = np.zeros((N_CTRL, 3))
    for i, si in enumerate(s):
        if si <= l1:
            frac = si / l1 if l1 > 1e-9 else 0.0
            pts[i] = p0 + frac * (pc - p0)
        else:
            frac = (si - l1) / l2 if l2 > 1e-9 else 0.0
            pts[i] = pc + frac * (p1 - pc)
    return pts, s / max(total, 1e-9), l1, l2


# ══════════════════════════════ SURROGATE LOSS (differentiable in r_i) ═══════════════════════════════

K_AREA = 5e2          # penalty stiffness, min-area floor (CONSTRUCTED, declared surrogate)
K_INTRUDE = 5e2        # penalty stiffness, obstacle intrusion


def _friction_dp(area_mm2, dh_mm, ds_mm, Q_m3s):
    area_m2 = area_mm2 * 1e-6
    dh_m = dh_mm * 1e-3
    ds_m = ds_mm * 1e-3
    v = Q_m3s / max(area_m2, 1e-9)
    re = RHO_AIR * v * dh_m / MU_AIR
    f_d = 64.0 / re if re < 2300 else 0.316 * re ** -0.25
    return f_d * (ds_m / max(dh_m, 1e-9)) * 0.5 * RHO_AIR * v ** 2, v, re


def loss_and_terms(radii_mm, case, k_bend):
    pts, s_frac, l1, l2 = control_points(case)
    n = len(radii_mm)
    dp_friction = 0.0
    seg_lens = np.diff(s_frac) * (l1 + l2)
    for i in range(n - 1):
        r_avg = 0.5 * (radii_mm[i] + radii_mm[i + 1])
        area = math.pi * r_avg ** 2
        dh = 2 * r_avg
        dp, _, _ = _friction_dp(area, dh, seg_lens[i], case["Q_m3s"])
        dp_friction += dp

    # bend term: applied at the control point nearest the corner (s_frac closest to l1/(l1+l2))
    corner_frac = l1 / max(l1 + l2, 1e-9)
    corner_idx = int(np.argmin(np.abs(s_frac - corner_frac)))
    r_corner = radii_mm[corner_idx]
    area_corner = math.pi * r_corner ** 2
    v_corner = case["Q_m3s"] / max(area_corner * 1e-6, 1e-9)
    dp_bend = k_bend * 0.5 * RHO_AIR * v_corner ** 2

    area_min = math.pi * case["min_r_mm"] ** 2
    pen_area = 0.0
    for r in radii_mm:
        a = math.pi * r ** 2
        viol = max(0.0, area_min - a)
        pen_area += K_AREA * (viol / max(area_min, 1e-9)) ** 2

    pen_intrude = 0.0
    if case["obstacle_node"] is not None:
        for i, r in enumerate(radii_mm):
            sdf = IKE.eval_sdf_py(case["obstacle_node"], tuple(pts[i]))
            viol = max(0.0, r - sdf)     # tube surface reaches into the obstacle if r > distance-to-surface
            pen_intrude += K_INTRUDE * (viol / max(case["min_r_mm"], 1e-9)) ** 2

    total = dp_friction + dp_bend + pen_area + pen_intrude
    return total, dict(dp_friction_pa=dp_friction, dp_bend_pa=dp_bend, pen_area=pen_area,
                        pen_intrude=pen_intrude, dp_total_unconstrained_pa=dp_friction + dp_bend)


def central_diff_grad(radii_mm, case, k_bend, h=0.5):
    n = len(radii_mm)
    g = np.zeros(n)
    for k in range(n):
        pp = radii_mm.copy(); pp[k] += h
        pm = radii_mm.copy(); pm[k] = max(1.0, pm[k] - h)
        lp, _ = loss_and_terms(pp, case, k_bend)
        lm, _ = loss_and_terms(pm, case, k_bend)
        g[k] = (lp - lm) / (2 * h)
    return g


def adam_descent(r0, case, k_bend, lr=0.4, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=600, tol=1e-10,
                  max_step_mm=3.0):
    """Adam with a per-step trust-region clip (max_step_mm): the loss landscape mixes a smooth friction
    term with quadratic penalty walls (area floor, obstacle intrusion) that can have very different local
    curvature at different control points simultaneously (one point needs to GROW to clear the area
    floor, a neighbour needs to SHRINK to clear an obstacle) -- an early unclipped run (lr=2.0, no clip)
    diverged on one corpus case (dP rose instead of falling, see reframe below); the clip keeps each
    control point's radius change bounded per iteration so Adam's momentum can settle instead of
    overshooting a penalty wall repeatedly."""
    r = r0.copy()
    m = np.zeros_like(r); v = np.zeros_like(r)
    hist = []
    t0 = time.perf_counter()
    it = 0
    best_r, best_loss = r.copy(), loss_and_terms(r, case, k_bend)[0]
    for it in range(1, max_iter + 1):
        loss, terms = loss_and_terms(r, case, k_bend)
        hist.append({"iter": it, "loss": loss, **terms})
        if loss < best_loss:
            best_loss, best_r = loss, r.copy()
        if it > 1 and abs(hist[-2]["loss"] - loss) < tol * max(abs(loss), 1e-12):
            break
        grad = central_diff_grad(r, case, k_bend)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = m / (1 - beta1 ** it); vhat = v / (1 - beta2 ** it)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        step = np.clip(step, -max_step_mm, max_step_mm)
        r = r - step
        r = np.maximum(r, 1.0)
    wall_s = time.perf_counter() - t0
    # report the BEST iterate seen (monotone-improving result), not necessarily the last (Adam can wander
    # near a penalty wall without a hard convergence criterion in the iteration budget)
    r = best_r
    final_loss, final_terms = loss_and_terms(r, case, k_bend)
    n_evals = it * (1 + 2 * len(r0)) + 1
    return dict(r_star=r, iters=it, n_evals=n_evals, wall_s=wall_s, final_loss=final_loss,
                final_terms=final_terms, history=hist)


# ══════════════════════════════ LBM JUDGE (real solver, per-leg uniform proxy) ═══════════════════════

def lbm_judge_two_leg(leg1_mean_r_mm, leg2_mean_r_mm, leg1_mm, leg2_mm, voxel_mm=12.0, tau=0.9, steps=1500):
    """Follows lbm_domare_v1.bent_duct_case's own pattern (a new function here, importing the
    validated D3Q19 primitives): each leg is a full-span slab (leg1 open for ALL x at
    its y/z-band; leg2 open for ALL y at its x/z-band); their overlap forms the corner, exactly as
    bent_duct_case does for the symmetric case. Generalized to INDEPENDENT per-leg widths (area-equivalent
    square proxy of the descended mean radius). z (height) is held at hmax=max(side1,side2) IDENTICALLY
    for both legs (declared simplification: only the in-plane bend width narrows per leg, not height) --
    this keeps the two legs' z-bands guaranteed to overlap at the corner (robust connectivity), at the
    cost of the narrower leg's true cross-section being a rectangle (side x hmax) rather than a square in
    the judge's proxy (the surrogate itself still assumes a true square/circle; only this judge geometry
    is height-shared)."""
    s1 = max(4, int(round(2 * leg1_mean_r_mm / voxel_mm)))
    s2 = max(4, int(round(2 * leg2_mean_r_mm / voxel_mm)))
    hmax = max(s1, s2)
    leg1_lu = max(6, int(round(leg1_mm / voxel_mm)))
    leg2_lu = max(6, int(round(leg2_mm / voxel_mm)))
    nx = leg1_lu + s2 + 2
    ny = leg2_lu + s1 + 2
    nz = hmax + 2
    shp = (nx, ny, nz)
    solid = np.ones(shp, dtype=bool)
    y0, z0 = 1, 1
    solid[:, y0:y0 + s1, z0:z0 + hmax] = False            # leg1: full x-span, inlet at x=0
    x0 = nx - 1 - s2
    solid[x0:x0 + s2, :, z0:z0 + hmax] = False            # leg2: full y-span, outlet at y=ny-1

    a_field = np.zeros(shp + (3,))
    drho = 0.01
    rho_in, rho_out = 1.0 + drho, 1.0 - drho
    inlet_idx = (0, slice(y0, y0 + s1), slice(z0, z0 + hmax))
    inlet_adj = (1, slice(y0, y0 + s1), slice(z0, z0 + hmax))
    outlet_idx = (slice(x0, x0 + s2), ny - 1, slice(z0, z0 + hmax))
    outlet_adj = (slice(x0, x0 + s2), ny - 2, slice(z0, z0 + hmax))
    bc_planes = [dict(idx=inlet_idx, adj=inlet_adj, rho=rho_in),
                 dict(idx=outlet_idx, adj=outlet_adj, rho=rho_out)]
    f, s = LD.run_lbm(shp, solid, tau, a_field, bc_planes, steps)
    rho = f.sum(-1)
    rho_safe = np.where(rho > 1e-9, rho, 1.0)
    u = (f @ LD.E) / rho_safe[..., None]
    u[solid] = 0.0
    uy_out = u[x0:x0 + s2, ny - 2, z0:z0 + hmax, 1]
    Q_lu = float(uy_out.sum())
    us = LD.unit_scale(voxel_mm, tau)
    area2_lu = s2 * hmax
    v_mean_lu = Q_lu / max(area2_lu, 1)
    dh2_lu = 4 * area2_lu / (2 * (s2 + hmax))
    nu_lu = us["nu_lu"]
    re_lu = v_mean_lu * dh2_lu / nu_lu if nu_lu > 0 else 0.0
    dP_lu = 2 * drho
    dP_lbm_pa = dP_lu * us["p_scale"]
    Q_phys_m3s = Q_lu * us["u_scale"] * (us["dx"] ** 2)
    R_pa_per_m3s = dP_lbm_pa / max(abs(Q_phys_m3s), 1e-30)
    return dict(dP_lbm_pa=float(dP_lbm_pa), Q_at_matched_point_m3s=float(Q_phys_m3s),
                R_pa_per_m3s=float(R_pa_per_m3s), n_cells=int(np.prod(shp)), steps_to_converge=int(s),
                grid_shape=list(shp), leg1_w_lu=s1, leg2_w_lu=s2, voxel_mm=voxel_mm,
                dP_pa_at_target_Q=float(R_pa_per_m3s * Q_BRANCH_M3S), re_lu=float(re_lu))


# ══════════════════════════════ STAGES ═══════════════════════════════

def all_cases():
    cases = [reference_case_L90()]
    for tid in SYNTH_TASK_IDS:
        cases.append(synth_case(tid))
    return cases


def stage_descend():
    t0 = time.time()
    kb = load_k_bend()
    print(f"[descend] K_BEND={kb['k_bend']:.4f} (dimensionless minor-loss, from {kb['source']})")
    out = {"k_bend_calibration": kb, "cases": []}
    for case in all_cases():
        r0 = np.full(N_CTRL, case["r_init_mm"])
        loss0, terms0 = loss_and_terms(r0, case, kb["k_bend"])
        res = adam_descent(r0, case, kb["k_bend"])
        pts, s_frac, l1, l2 = control_points(case)
        corner_frac = l1 / max(l1 + l2, 1e-9)
        corner_idx = int(np.argmin(np.abs(s_frac - corner_frac)))
        leg1_r0 = float(np.mean(r0[:corner_idx + 1]))
        leg2_r0 = float(np.mean(r0[corner_idx:]))
        leg1_rstar = float(np.mean(res["r_star"][:corner_idx + 1]))
        leg2_rstar = float(np.mean(res["r_star"][corner_idx:]))
        rec = dict(case_id=case["case_id"], source=case["source"], min_r_mm=case["min_r_mm"],
                   leg1_mm=case["leg1_mm"], leg2_mm=case["leg2_mm"], corner_idx=corner_idx,
                   r_init_mm=r0.tolist(), r_star_mm=res["r_star"].tolist(),
                   leg1_mean_r_init_mm=leg1_r0, leg2_mean_r_init_mm=leg2_r0,
                   leg1_mean_r_star_mm=leg1_rstar, leg2_mean_r_star_mm=leg2_rstar,
                   loss0=loss0, terms0=terms0, iters=res["iters"], n_evals=res["n_evals"],
                   wall_s=res["wall_s"], final_loss=res["final_loss"], final_terms=res["final_terms"],
                   dp_surrogate_drop_frac=float((terms0["dp_total_unconstrained_pa"]
                                                  - res["final_terms"]["dp_total_unconstrained_pa"])
                                                 / max(terms0["dp_total_unconstrained_pa"], 1e-12)),
                   constraints_ok=bool(res["final_terms"]["pen_area"] < 1e-6
                                        and res["final_terms"]["pen_intrude"] < 1e-6))
        out["cases"].append(rec)
        print(f"[descend] {case['case_id']}: dP {terms0['dp_total_unconstrained_pa']:.4e} -> "
              f"{res['final_terms']['dp_total_unconstrained_pa']:.4e} Pa "
              f"(drop {rec['dp_surrogate_drop_frac']*100:.1f}%) iters={res['iters']} "
              f"constraints_ok={rec['constraints_ok']}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{SCRATCH}/descend.json", "w"), indent=1)
    print(f"-> {SCRATCH}/descend.json")
    return out


def stage_judge(which):
    """which in {before, after}; run as two separate invocations."""
    t0 = time.time()
    desc = json.load(open(f"{SCRATCH}/descend.json"))
    results = []
    for rec in desc["cases"]:
        if which == "before":
            l1r, l2r = rec["leg1_mean_r_init_mm"], rec["leg2_mean_r_init_mm"]
        else:
            l1r, l2r = rec["leg1_mean_r_star_mm"], rec["leg2_mean_r_star_mm"]
        j = lbm_judge_two_leg(l1r, l2r, rec["leg1_mm"], rec["leg2_mm"])
        j["case_id"] = rec["case_id"]
        results.append(j)
        print(f"[judge:{which}] {rec['case_id']}: leg_r=({l1r:.1f},{l2r:.1f})mm "
              f"dP_at_target_Q={j['dP_pa_at_target_Q']:.4e}Pa n_cells={j['n_cells']} "
              f"t={time.time()-t0:.1f}s")
    out = dict(which=which, results=results, elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(f"{SCRATCH}/judge_{which}.json", "w"), indent=1)
    print(f"-> {SCRATCH}/judge_{which}.json")
    return out


def stage_report():
    desc = json.load(open(f"{SCRATCH}/descend.json"))
    jb = json.load(open(f"{SCRATCH}/judge_before.json"))
    ja = json.load(open(f"{SCRATCH}/judge_after.json"))
    jb_by_id = {r["case_id"]: r for r in jb["results"]}
    ja_by_id = {r["case_id"]: r for r in ja["results"]}

    cases_out = []
    n_pass_15 = 0
    n_lbm_confirms = 0
    for rec in desc["cases"]:
        cid = rec["case_id"]
        b, a = jb_by_id[cid], ja_by_id[cid]
        lbm_drop_frac = (b["dP_pa_at_target_Q"] - a["dP_pa_at_target_Q"]) / max(b["dP_pa_at_target_Q"], 1e-30)
        atom_15 = rec["dp_surrogate_drop_frac"] >= 0.15 and rec["constraints_ok"]
        lbm_confirms = lbm_drop_frac > 0.0
        n_pass_15 += int(atom_15)
        n_lbm_confirms += int(lbm_confirms)
        case_rec = dict(
            case_id=cid, source=rec["source"],
            surrogate_dp_drop_frac=rec["dp_surrogate_drop_frac"], constraints_ok=rec["constraints_ok"],
            atom_ge15pct_pass=bool(atom_15),
            lbm_before=b, lbm_after=a, lbm_dp_drop_frac=float(lbm_drop_frac),
            lbm_confirms_improvement=bool(lbm_confirms),
        )
        if not rec["constraints_ok"]:
            # honest-negative diagnosis: price the negative with a mechanism, not just a flag --
            # re-measure the obstacle SDF at every control point of the case for which the constrained
            # surrogate could not resolve (min-area floor vs obstacle clearance in conflict).
            case = synth_case(cid) if cid != "reference_L90" else reference_case_L90()
            pts, s_frac, l1, l2 = control_points(case)
            sdfs = ([float(IKE.eval_sdf_py(case["obstacle_node"], tuple(p))) for p in pts]
                    if case["obstacle_node"] is not None else [None] * N_CTRL)
            n_inside_obstacle_at_r0 = sum(1 for v in sdfs if v is not None and v < 0.0)
            case_rec["honest_negative_diagnosis"] = dict(
                min_r_mm=case["min_r_mm"], control_point_obstacle_sdf_mm=sdfs,
                n_control_points_inside_obstacle_at_centerline=n_inside_obstacle_at_r0,
                mechanism=("the route's own CENTERLINE (radius-independent, fixed by the naive 2-segment "
                            "elbow path) sits inside an obstacle at >=1 control point" if n_inside_obstacle_at_r0 > 0
                           else "min-area floor and obstacle clearance conflict at a positive-but-small "
                                "standoff -- no radius satisfies both simultaneously"),
            )
        cases_out.append(case_rec)

    report = {
        "_doc": "Differentiable duct growth against pressure drop -- a surrogate (Darcy-Weisbach plus "
                "an LBM-calibrated K_BEND bend term) descended with central-difference Adam, with a real "
                "LBM run (D3Q19, reused solver) judging before and after on a per-leg-uniform proxy of "
                "the descended radius profile. See the duct_growth_diff_v1.py docstring for the method.",
        "gradient_path_choice": {
            "chosen": "b_differentiable_surrogate_calibrated_against_lbm_corpus",
            "rejected_a_adjoint_on_lbm": "MEASURED: an LBM evaluation costs tens of seconds; 100+ descent "
                "steps x 2*N_CTRL central-difference evaluations would cost hours. Not chosen for the "
                "per-step gradient, but reused AS THE JUDGE (2 calls/case, 8 total).",
            "chosen_c_ikarus_gradient_batch": "USED for the constraint term (intrusion penalty via "
                "IKE.eval_sdf_py against the obstacle scene node), not for the dP objective itself "
                "(ikarus is a geometry/collision SDF kernel, not a fluid solver).",
        },
        "k_bend_calibration": desc["k_bend_calibration"],
        "reframe_declared": [
            "K_BEND measured at the LBM's own creeping-flow operating point is applied here at the "
            "declared turbulent design-flow regime (Re in the thousands) as a DIMENSIONLESS minor-loss "
            "coefficient (K*0.5*rho*v^2, standard duct-engineering form, approximately Re-independent for "
            "a fixed bend-geometry ratio) -- a CONSTRUCTED modelling choice, not a measurement in the new "
            "regime; declared, in the same spirit as lbm_domare_v1.py's own regime-mismatch declarations.",
            "the LBM judge sees a PER-LEG-UNIFORM proxy (area-equivalent mean radius per leg) of the "
            "descended per-control-point radius profile, not the full stepped geometry -- a declared "
            "simplification that keeps the 8 LBM calls (4 cases x before/after) cheap.",
        ],
        "measured_vs_constructed": {
            "measured": ["K_BEND source excess_over_formula_pa + Re_matched (lbm_domare_v1.json)",
                          "obstacle geometry (data/corpus/duct_v1 task.json, via ikarus SDF)",
                          "min_bend_r_floor_mm per case", "Q_BRANCH_M3S (declared branch flow)",
                          "LBM judge dP before/after (D3Q19, run_lbm/lbm_step reused unmodified)"],
            "constructed": ["circular cross-section approximation for the surrogate (LBM judge uses square, "
                              "declared mismatch, both area-equivalent)",
                              "K_AREA/K_INTRUDE penalty stiffness (declared surrogate)",
                              "K_BEND re-use across regimes (see reframe above)",
                              "LBM judge per-leg-uniform proxy of the descended profile"],
        },
        "cases": cases_out,
        "summary": {
            "n_cases": len(cases_out),
            "n_cases_surrogate_drop_ge_15pct_and_constrained": n_pass_15,
            "n_cases_lbm_confirms_improvement": n_lbm_confirms,
        },
        "optimizer_select_classification": {
            "n_vars": N_CTRL, "seconds_per_eval_surrogate": "<1ms (pure numpy)",
            "differentiable_physics": True,
            "table_row": "low-dimensional differentiable descent, N_CTRL=6 <= 20 variables.",
            "new_method_class_proposed": "descend-then-judge hybrid: gradient descent on a CALIBRATED "
                "cheap differentiable surrogate (proposer), periodic/final evaluation by the expensive "
                "high-fidelity solver on the SAME candidate (judge) -- distinct from row5's plain "
                "single-evaluator descent because the objective the descent optimizes is NOT the objective "
                "being certified.",
        },
        "generalization_contract": {
            "eats": ["continuous cross-section/radius growth along a fixed route topology, with an "
                     "expensive-to-evaluate ground-truth solver too costly for per-step gradients but "
                     "cheap enough for periodic/final judging -- calibrate a fast differentiable surrogate "
                     "against the solver's own measured discrepancy (here: the missing bend-loss term), "
                     "descend on the surrogate, certify with the solver."],
            "does_not_eat": ["route TOPOLOGY changes (which wall the ports sit on, how many bends) -- "
                              "a generative-sampling decision, handled by f33_1_topologi_v1.py.",
                              "full-fidelity per-control-point LBM judging (the judge here sees a "
                              "per-leg-uniform proxy, not the stepped profile -- declared, not silently "
                              "extrapolated)."],
        },
    }

    atoms = [
        {"id": "a0_report_exists", "type": "artifact-exists", "artifact": "artifacts/duct_growth_diff_v1.json"},
        {"id": "a1_kbend_positive_measured", "type": "inequality",
         "lhs": {"artifact": "artifacts/duct_growth_diff_v1.json", "key": "k_bend_calibration.k_bend"},
         "op": ">", "rhs": 0.0},
        {"id": "a2_reference_case_is_first_and_named", "type": "value-in-artifact",
         "artifact": "artifacts/duct_growth_diff_v1.json", "key": "cases.0.case_id",
         "expected": "reference_L90"},
        {"id": "a3_majority_surrogate_ge15pct_drop_constrained", "type": "inequality",
         "lhs": {"artifact": "artifacts/duct_growth_diff_v1.json",
                 "key": "summary.n_cases_surrogate_drop_ge_15pct_and_constrained"},
         "op": ">=", "rhs": max(1, len(cases_out) // 2)},
        {"id": "a4_reference_case_surrogate_drop_ge_15pct", "type": "inequality",
         "lhs": {"artifact": "artifacts/duct_growth_diff_v1.json", "key": "cases.0.surrogate_dp_drop_frac"},
         "op": ">=", "rhs": 0.15},
        {"id": "a5_reference_case_constrained", "type": "value-in-artifact",
         "artifact": "artifacts/duct_growth_diff_v1.json", "key": "cases.0.constraints_ok", "expected": True},
        {"id": "a6_lbm_judge_confirms_on_majority", "type": "inequality",
         "lhs": {"artifact": "artifacts/duct_growth_diff_v1.json", "key": "summary.n_cases_lbm_confirms_improvement"},
         "op": ">=", "rhs": max(1, len(cases_out) // 2)},
    ]
    report["ATOMS"] = {"atoms": atoms}

    json.dump(report, open(RPT, "w"), indent=1, ensure_ascii=False)
    print(f"-> {RPT}")
    return report


def stage_selftest():
    """Reduced gate: the reference case only, descended and judged before/after by the real LBM.

    Gates, all from the module's own numbers: the calibrated K_BEND is positive, the surrogate drops
    the reference case's pressure drop by at least 15 % with its constraints satisfied, and the LBM
    judge confirms the drop."""
    t0 = time.time()
    kb = load_k_bend()
    case = reference_case_L90()
    r0 = np.full(N_CTRL, case["r_init_mm"])
    loss0, terms0 = loss_and_terms(r0, case, kb["k_bend"])
    res = adam_descent(r0, case, kb["k_bend"])
    drop = float((terms0["dp_total_unconstrained_pa"] - res["final_terms"]["dp_total_unconstrained_pa"])
                  / max(terms0["dp_total_unconstrained_pa"], 1e-12))
    ok_constraints = bool(res["final_terms"]["pen_area"] < 1e-6 and res["final_terms"]["pen_intrude"] < 1e-6)
    pts, s_frac, l1, l2 = control_points(case)
    corner_idx = int(np.argmin(np.abs(s_frac - l1 / max(l1 + l2, 1e-9))))
    before = lbm_judge_two_leg(float(np.mean(r0[:corner_idx + 1])), float(np.mean(r0[corner_idx:])),
                                case["leg1_mm"], case["leg2_mm"])
    after = lbm_judge_two_leg(float(np.mean(res["r_star"][:corner_idx + 1])),
                               float(np.mean(res["r_star"][corner_idx:])), case["leg1_mm"], case["leg2_mm"])
    lbm_drop = float((before["dP_pa_at_target_Q"] - after["dP_pa_at_target_Q"])
                      / max(before["dP_pa_at_target_Q"], 1e-30))
    checks = [("k_bend_positive", kb["k_bend"], kb["k_bend"] > 0.0),
              ("surrogate_drop_ge_15pct", drop, drop >= 0.15),
              ("constraints_ok", 1.0 if ok_constraints else 0.0, ok_constraints),
              ("lbm_judge_confirms", lbm_drop, lbm_drop > 0.0)]
    ok = True
    for name, value, passed in checks:
        print(f"[selftest] {name}: {value:.4f} {'PASS' if passed else 'FAIL'}")
        ok = ok and passed
    out = dict(case_id=case["case_id"], k_bend=kb["k_bend"], surrogate_dp_drop_frac=drop,
               constraints_ok=ok_constraints, lbm_before=before, lbm_after=after,
               lbm_dp_drop_frac=lbm_drop, iters=res["iters"], n_evals=res["n_evals"],
               checks={n: dict(value=float(v), passed=bool(p)) for n, v, p in checks},
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(f"{SCRATCH}/selftest.json", "w"), indent=1)
    print(f"[selftest] {'ALL_PASS' if ok else 'FAIL'} in {out['elapsed_s']}s -> {SCRATCH}/selftest.json")
    return 0 if ok else 1


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "selftest":
        return stage_selftest()
    if stage in ("descend", "all"):
        stage_descend()
    if stage in ("judge_before", "all"):
        stage_judge("before")
    if stage in ("judge_after", "all"):
        stage_judge("after")
    if stage in ("report", "all"):
        stage_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
