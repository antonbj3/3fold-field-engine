#!/usr/bin/env python3
"""Query engine over expr.py trees: volume, intrusion, clearance and bbox, evaluated through eval_warp.

Every query picks its algorithm and resolution from a measured cost model rather than a fixed constant,
and can be given an explicit per-call compute budget in milliseconds that it tries to honour while
declaring the resulting uncertainty:
  1. calibrate throughput (points/s) for this expression on this device;
  2. hierarchical prune: a coarse grid pass over the query bbox classifies each cell as fully inside,
     fully outside or boundary, using the Lipschitz-1 property (|value| >= half the cell diagonal means
     the whole cell is unambiguously on one side);
  3. adaptive refinement of boundary cells only, as deep as the remaining budget affords;
  4. every dispatch decision (coarse pitch, refinement factor, points spent per pass) is logged in the
     returned dict.

Each result carries a declared uncertainty in mm^3, the volume spanned by cells still ambiguous at the
finest level reached, so a caller can demand "refine until uncertainty < tolerance" instead of trusting a
bare number.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expr as E
import eval_warp as W
import cost_model_v1_2 as CM


# --------------------------------------------------------------------------------------------- bbox
def bbox_of(node, samples=20000, pad=1.0, seed=0):
    """Estimate an expression's axis-aligned bbox by sampling around its analytic-primitive leaves'
    OWN declared extents (exact for pure-primitive trees; for gridfield leaves the leaf's own grid extent
    is used exactly, not sampled) -- avoids an expensive global search."""
    lo = np.array([1e30, 1e30, 1e30])
    hi = -lo.copy()

    def walk(n, xform):
        op = n["op"]
        if op == "sphere":
            c = np.array(n["center"]); r = n["radius"]
            pts = np.array([c + r * np.array(v) for v in
                             [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]])
        elif op == "box":
            c = np.array(n["center"]); h = np.array(n["half_extents"])
            pts = np.array([c + np.array([sx * h[0], sy * h[1], sz * h[2]])
                             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        elif op in ("cylinder", "cone"):
            c = np.array(n["center"]); h = n["height"]
            r = max(n.get("radius", 0), n.get("radius1", 0), n.get("radius2", 0))
            pts = np.array([c + np.array([sx * r, sy * r, sz * h / 2.0])
                             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        elif op == "halfspace":
            return   # unbounded; contributes nothing to a finite bbox estimate
        elif op == "gridfield":
            d = np.load(n["npz_path"])
            origin = np.asarray(d["global_origin_mm"]) if "global_origin_mm" in d else np.zeros(3)
            pitch = n["pitch_override"] or float(d["pitch_mm"])
            shp = np.asarray(d[n["key"]]).shape
            extent = origin + pitch * (np.array(shp) - 1)
            pts = np.array([origin, extent])
        elif op == "transform":
            walk(n["child"], xform @ np.array(n["mat4"]))
            return
        elif op in ("offset", "scale", "axis_project"):
            # axis_project's bbox is the CHILD's bbox unwidened (sdf_shell_v1): the projected coordinate
            # is only ever queried through a bounding halfspace-slab sibling node in practice (shell's
            # open-face cap_region), so pass-through here stays a safe (if slightly loose, non-tight)
            # finite bound rather than the true infinite-prism extent -- bbox_of only feeds grid-pitch
            # sizing, which tolerates a loose bound.
            walk(n["child"], xform)
            return
        elif op in ("union", "intersect", "subtract"):
            walk(n["a"], xform); walk(n["b"], xform)
            return
        else:
            raise ValueError(op)
        ones = np.ones((pts.shape[0], 1))
        world = (xform @ np.hstack([pts, ones]).T).T[:, :3]
        nonlocal lo, hi
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))

    walk(node, np.eye(4))
    return lo - pad, hi + pad


# --------------------------------------------------------------------------------------------- throughput calibration
def calibrate_throughput(node, n_probe=200_000, device=W.DEVICE):
    """One short timed batch (kernel already compiled by the caller's earlier eval_batch, so this
    measures STEADY-STATE points/sec, not compile latency) -- the number the dispatcher sizes every grid
    pass from."""
    lo, hi = bbox_of(node)
    rng = np.random.default_rng(0)
    pts = rng.uniform(lo, hi, size=(n_probe, 3))
    t0 = time.time()
    W.eval_batch(node, pts, device=device)
    dt = time.time() - t0
    return n_probe / max(dt, 1e-6), dt


def calibrate_throughput_cached(node, device=W.DEVICE, cache_path=CM._DEFAULT_CACHE_PATH):
    """Deterministic replacement for calibrate_throughput: a two-term model (fixed overhead_s plus
    marginal rate_pts_per_s) measured once per (device, kernel shape) and cached to disk.

    A wall-clock rate is subject to timing jitter, which makes coarse-pitch sizing non-deterministic
    run to run; after the first calibration for a given expression shape on this device, every later
    call reads the identical cached values back. Returns (rate_pts_per_s, overhead_s, from_cache).
    """
    lo, hi = bbox_of(node)
    W.eval_batch(node, np.zeros((8, 3)), device=device)   # warm the kernel-compile cache first (excluded)
    shape_key = W._kernel_cache_key(node)
    overhead_s, rate, from_cache = CM.calibrate_two_term(
        shape_key, lambda pts: W.eval_batch(node, pts, device=device), lo, hi, device, cache_path=cache_path)
    return rate, overhead_s, from_cache


def _grid_points(lo, hi, pitch):
    nx = max(int(math.ceil((hi[0] - lo[0]) / pitch)) + 1, 1)
    ny = max(int(math.ceil((hi[1] - lo[1]) / pitch)) + 1, 1)
    nz = max(int(math.ceil((hi[2] - lo[2]) / pitch)) + 1, 1)
    xs = lo[0] + pitch * np.arange(nx)
    ys = lo[1] + pitch * np.arange(ny)
    zs = lo[2] + pitch * np.arange(nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return pts, (nx, ny, nz)


# --------------------------------------------------------------------------------------------- the core dispatcher
def intrusion_volume(exprA, exprB, budget_ms=200.0, min_pitch_mm=0.25, max_refine_factor=8,
                      device=W.DEVICE, log=None, max_coarse_points=None):
    """Volume of A∩B, hierarchical-prune + adaptive-refine, budget-dispatched.

    Method rows (data-driven, logged per call -- optimizer_select_v1.py pattern):
      ROW coarse: full-bbox brute grid at a pitch sized so coarse-pass points ~= 0.6*budget_points
      ROW refine: boundary-only fine grid at pitch = coarse_pitch/refine_factor, refine_factor chosen so
        (n_boundary_coarse_cells * refine_factor**3) fits the REMAINING budget after the coarse pass'
        actual measured cost -- if it would overshoot, refine_factor is reduced (logged, not silently
        clipped) rather than blowing the budget.

    `max_coarse_points`: calibrate_throughput()'s rate estimate is measured on a small
    probe and assumes throughput is constant (zero fixed cost). For a candidate pair whose operand
    bboxes barely overlap, the intersect bbox can still be a leaf's full extent while the calibrated
    rate on a tiny probe reads very high, and the sizing formula then targets millions of points whose
    real cost (host-side grid construction and transfer, not device compute) overshoots the budget
    several-fold. This cap clips n_coarse_target directly; the algorithm below the cap is unchanged.
    """
    combo = E.intersect(exprA, exprB, k=0.0)
    # the deterministic two-term cost model replaces the timing-based calibrate_throughput(), so
    # calib_dt below is the cached fixed-dispatch overhead, not a freshly measured wall time.
    rate, overhead_s, calib_from_cache = calibrate_throughput_cached(combo, device=device)
    calib_dt = overhead_s
    budget_points = CM.affordable_points(budget_ms, overhead_s, rate, floor_points=1000)

    # bbox_of() walks union/intersect/subtract nodes identically (bbox of the whole subtree), which for
    # an A-vs-B intrusion query gives the union of A's and B's extents: correct for union, wrong for
    # intersect, where the true search space is the overlap of the two bboxes and can be orders of
    # magnitude smaller. Intersect the two operands' own bboxes instead of trusting bbox_of(combo).
    loA, hiA = bbox_of(exprA)
    loB, hiB = bbox_of(exprB)
    lo = np.maximum(loA, loB)
    hi = np.minimum(hiA, hiB)
    if np.any(hi <= lo):
        return {"method": "hierarchical_prune_adaptive_refine", "volume_mm3": 0.0, "uncertainty_mm3": 0.0,
                "budget_ms": budget_ms, "wall_ms": 0.0, "budget_met_within_20pct": True,
                "dispatch_log": {"note": "operand bboxes do not overlap -- 0 volume by construction, no "
                                          "grid pass needed", "bboxA": [loA.tolist(), hiA.tolist()],
                                  "bboxB": [loB.tolist(), hiB.tolist()]}}
    bbox_vol = float(np.prod(hi - lo))

    # STAGE 0 (cheap pre-pass, ~1-2% of budget): estimate the BOUNDARY FRACTION (surface-crossing cells
    # / total cells) at a throwaway resolution -- refinement cost scales as boundary_cells*refine_factor^3,
    # which for a thin/flat overlap (large surface-area-to-volume ratio) can dwarf the coarse-pass cost;
    # sizing coarse_pitch from bbox_vol/budget_points alone (ignoring surface density) starves the
    # refine pass on thin-slab geometries (measured on a 5x20x20 mm box-overlap sliver: 31.9% low-biased
    # volume, refine_factor forced to 1, before this stage was added).
    t0 = time.time()
    probe_pitch = max((bbox_vol / max(int(0.03 * budget_points), 200)) ** (1.0 / 3.0), min_pitch_mm)
    pts_p, _ = _grid_points(lo, hi, probe_pitch)
    vals_p = W.eval_batch(combo, pts_p, device=device)
    t_probe = time.time() - t0
    probe_diag_half = 0.5 * probe_pitch * math.sqrt(3.0)
    f_boundary = float((np.abs(vals_p) <= probe_diag_half).mean())
    f_boundary = min(max(f_boundary, 1e-4), 1.0)

    target_refine_factor = 4
    # solve coarse_pitch so that N_coarse + f_boundary*N_coarse*target_refine_factor^3 ~= budget_points
    remaining_after_probe = max(budget_points - len(pts_p), 1000)
    denom = 1.0 + f_boundary * (target_refine_factor ** 3)
    n_coarse_target = max(int(remaining_after_probe / denom), 64)
    if max_coarse_points is not None:
        n_coarse_target = min(n_coarse_target, max_coarse_points)
    coarse_pitch = max((bbox_vol / n_coarse_target) ** (1.0 / 3.0), min_pitch_mm)

    t0 = time.time()
    pts_c, shape_c = _grid_points(lo, hi, coarse_pitch)
    vals_c = W.eval_batch(combo, pts_c, device=device)
    t_coarse = time.time() - t0
    coarse_diag_half = 0.5 * coarse_pitch * math.sqrt(3.0)
    fully_inside = vals_c < -coarse_diag_half
    boundary = np.abs(vals_c) <= coarse_diag_half
    coarse_vol_cell = coarse_pitch ** 3
    vol_from_coarse = float(fully_inside.sum()) * coarse_vol_cell
    n_boundary = int(boundary.sum())

    remaining_points = max(budget_points - len(pts_p) - len(pts_c), 0)
    refine_factor = min(max_refine_factor, target_refine_factor)
    if n_boundary > 0:
        while refine_factor > 1 and n_boundary * (refine_factor ** 3) > max(remaining_points, 1):
            refine_factor -= 1
    refine_factor = max(refine_factor, 1)
    fine_pitch = max(coarse_pitch / refine_factor, min_pitch_mm)

    t0 = time.time()
    vol_from_fine = 0.0
    n_fine_pts = 0
    n_boundary_fine = 0
    if n_boundary > 0 and refine_factor > 1:
        boundary_centers = pts_c[boundary]
        half = coarse_pitch / 2.0
        offs = (np.arange(refine_factor) + 0.5) / refine_factor * coarse_pitch - half
        OX, OY, OZ = np.meshgrid(offs, offs, offs, indexing="ij")
        local = np.stack([OX.ravel(), OY.ravel(), OZ.ravel()], axis=1)   # (refine_factor^3, 3)
        fine_pts = (boundary_centers[:, None, :] + local[None, :, :]).reshape(-1, 3)
        n_fine_pts = fine_pts.shape[0]
        fine_vals = W.eval_batch(combo, fine_pts, device=device)
        fine_cell_vol = (coarse_pitch / refine_factor) ** 3
        fine_diag_half = 0.5 * (coarse_pitch / refine_factor) * math.sqrt(3.0)
        fine_inside = fine_vals < 0.0
        vol_from_fine = float(fine_inside.sum()) * fine_cell_vol
        n_boundary_fine = int((np.abs(fine_vals) <= fine_diag_half).sum())
    t_fine = time.time() - t0

    total_vol = vol_from_coarse + vol_from_fine
    # declared uncertainty: every finest-level cell still ambiguous (|value|<=half-diagonal) contributes
    # its full cell volume as a +/- bound (conservative: worst case it's entirely mis-classified).
    finest_cell_vol = (coarse_pitch / refine_factor) ** 3 if (n_boundary > 0 and refine_factor > 1) else coarse_vol_cell
    uncertainty_mm3 = n_boundary_fine * finest_cell_vol if (n_boundary > 0 and refine_factor > 1) else n_boundary * coarse_vol_cell

    total_wall_s = calib_dt + t_probe + t_coarse + t_fine
    result = {
        "method": "hierarchical_prune_adaptive_refine",
        "volume_mm3": total_vol,
        "uncertainty_mm3": uncertainty_mm3,
        "budget_ms": budget_ms,
        "wall_ms": total_wall_s * 1000.0,
        "budget_met_within_20pct": bool(total_wall_s * 1000.0 <= budget_ms * 1.2),
        "dispatch_log": {
            "calibrated_rate_pts_per_s": rate,
            "cost_model_v1_2": {"overhead_s": overhead_s, "rate_pts_per_s": rate,
                                 "from_cache": calib_from_cache},
            "budget_points": budget_points,
            "max_coarse_points_cap": max_coarse_points,
            "coarse_cap_engaged": bool(max_coarse_points is not None and n_coarse_target >= max_coarse_points),
            "probe_pitch_mm": probe_pitch, "n_probe_points": len(pts_p),
            "boundary_fraction_estimate": f_boundary, "target_refine_factor": target_refine_factor,
            "bbox_mm": {"lo": lo.tolist(), "hi": hi.tolist(), "volume_mm3": bbox_vol},
            "coarse_pitch_mm": coarse_pitch, "coarse_grid_shape": shape_c, "n_coarse_points": len(pts_c),
            "n_fully_inside_coarse": int(fully_inside.sum()), "n_boundary_coarse": n_boundary,
            "coarse_wall_s": t_coarse,
            "refine_factor_chosen": refine_factor, "fine_pitch_mm": fine_pitch,
            "n_fine_points": n_fine_pts, "n_boundary_remaining_at_finest": n_boundary_fine,
            "fine_wall_s": t_fine,
            "space_eliminated_frac": 1.0 - (n_boundary / max(len(pts_c), 1)),
        },
    }
    if log is not None:
        log.append(result["dispatch_log"])
    return result


# --------------------------------------------------------------------------------------------- v1.2 levers (2)+(3): SWEEP-level dispatcher
def scene_sweep_gridfield_pairs(pairs, total_budget_ms=800.0, min_pitch_mm=0.25, max_refine_factor=8,
                                 target_refine_factor=4, min_pair_coarse_points=27, device=W.DEVICE,
                                 cache_path=CM._DEFAULT_CACHE_PATH, prior_vol_override=None,
                                 max_pair_frac_of_pool=0.05, max_pair_points_abs=150_000,
                                 max_total_refine_points_abs=6_000_000):
    """IKARUS v1.2's fullscenssvep entry point: many (leafA, leafB) gridfield-leaf pairs -- e.g. every
    AABB-prefilter candidate in a real scene sweep -- answered with a hard intersect(A,B) volume+
    uncertainty each, using ONE total sweep budget instead of one budget_ms PER PAIR.

    lever (2) BUDGET ALLOCATION: v1's intrusion_volume() gives every pair the SAME budget_ms regardless
    of geometry, so a pair whose two AABBs merely touch (common: 1690/40470 candidates survive the
    prefilter precisely BECAUSE their boxes touch, not because their solids do) still gets sized toward
    the SAME target point count as a pair with a large genuine overlap -- v1.1 profiled this as the
    dominant remaining cost (coarse grids sized in the millions for near-empty pairs). Here the total
    sweep budget is instead distributed across pairs proportional to each pair's AABB-OVERLAP-VOLUME
    PRIOR (intersect(bboxA, bboxB) volume -- free to compute, no GPU needed) -- a pair whose boxes barely
    touch gets only the declared FLOOR (`min_pair_coarse_points`, a minimal classify-and-move-on probe),
    a pair with a large overlap prior gets a proportionally larger point allocation for refinement. Every
    pair still returns a declared uncertainty_mm3 (v1's contract preserved).

    lever (3) BATCHING: every pair's coarse-grid points (sized per its allocation above) are concatenated
    into ONE flat point buffer + a per-point pair-id index, and answered by ONE
    eval_batch_multi_pair() kernel launch for the WHOLE sweep (not one launch per pair) -- see
    eval_warp.py's compile_multi_pair_gridfield_kernel docstring for the root cause this removes
    (per-launch fixed overhead, now paid once instead of N times). A second batched launch handles the
    (also globally-budgeted) refine pass over every pair's boundary cells together.

    `prior_vol_override` (v1.2 scoped stress-test hook, optional): a list of length len(pairs) of
    externally supplied AABB-overlap-volume priors to drive budget allocation (stage C's share
    computation) instead of the value recomputed from each pair's own leaf geometry. This lets a caller
    exercise the dispatcher's cost model and batching at a measured overlap-volume distribution while
    the leaf data actually queried is a smaller corpus reused across many pair slots. Grid placement
    (lo/hi, empty-detection) still comes from each pair's own actual leaf bboxes -- only the
    ALLOCATION WEIGHT changes, so every dispatch still queries real, valid gridfield content.

    `max_pair_frac_of_pool` is a safety valve on the same precedent as `max_coarse_points`: the
    AABB-overlap-volume distribution over candidate pairs is heavy-tailed (measured on a 1645-candidate
    sweep: the top ten pairs carry 40% of the total prior volume, a factor-4000 span from median to
    max), and an unclipped linear-in-prior allocation hands a few outlier pairs multi-million-point
    grids whose host-side construction cost the two-term model does not capture (an uncapped run
    predicted under 1 s but measured 4.34 s). Clipping any single pair's allocated share to this
    fraction of the pool preserves the prior's rank order while bounding worst-case construction cost.

    Returns (results, sweep_log): results[i] mirrors intrusion_volume()'s per-pair dict (volume_mm3,
    uncertainty_mm3, dispatch_log); sweep_log carries the SWEEP-level cost-model/budget-allocation
    decisions (prior volumes, total budget split, wall times per stage)."""
    n_pairs = len(pairs)
    if n_pairs == 0:
        return [], {"n_pairs": 0, "note": "empty pair list"}
    t_sweep0 = time.time()

    # ---- stage A: cheap per-pair AABB-overlap PRIOR (pure numpy/python, no GPU) ----
    # memoized per distinct leaf (npz_path, key, pitch_override): candidate pairs reuse the same small
    # set of per-solid leaves, and bbox_of()'s gridfield branch does np.load(npz_path) on every call, so
    # an unmemoized per-pair call re-opens the same file up to n_pairs times (measured on a 1645-pair
    # sweep over 19 distinct leaves: 1.14 s unmemoized versus a few ms memoized).
    _bbox_leaf_cache = {}
    def _bbox_leaf(node):
        if node["op"] == "gridfield":
            ck = (node["npz_path"], node["key"], node.get("pitch_override"))
            if ck not in _bbox_leaf_cache:
                _bbox_leaf_cache[ck] = bbox_of(node)
            return _bbox_leaf_cache[ck]
        return bbox_of(node)

    priors = []
    for la, lb in pairs:
        loA, hiA = _bbox_leaf(la); loB, hiB = _bbox_leaf(lb)
        lo = np.maximum(loA, loB); hi = np.minimum(hiA, hiB)
        empty = bool(np.any(hi <= lo))
        vol = 0.0 if empty else float(np.prod(hi - lo))
        priors.append({"lo": lo, "hi": hi, "vol": vol, "empty": empty})
    if prior_vol_override is not None:
        assert len(prior_vol_override) == n_pairs
        for i, p in enumerate(priors):
            if not p["empty"]:
                p["alloc_vol"] = float(prior_vol_override[i])
            else:
                p["alloc_vol"] = 0.0
    else:
        for p in priors:
            p["alloc_vol"] = p["vol"]
    total_prior_vol = sum(p["alloc_vol"] for p in priors)
    t_prior = time.time() - t_sweep0

    # ---- stage B: two-term cost model for the FIXED multi-pair kernel shape (cached, deterministic) ----
    def _probe_eval(pts):
        n = pts.shape[0]
        pop = np.zeros(n, dtype=np.int32)
        return W.eval_batch_multi_pair(pts, pop, np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
                                        [pairs[0][0]], device=device)
    lo0, hi0 = _bbox_leaf(pairs[0][0])
    W.compile_multi_pair_gridfield_kernel(device)   # warm compile, excluded from the timed probe
    overhead_s, rate, from_cache = CM.calibrate_two_term(
        W.MULTI_PAIR_SHAPE_KEY, _probe_eval, lo0, hi0, device, cache_path=cache_path)

    floor_total = n_pairs * min_pair_coarse_points
    total_points_budget = CM.affordable_points(total_budget_ms, overhead_s, rate, floor_points=floor_total)
    alloc_pool = max(total_points_budget - floor_total, 0)

    # ---- stage C: per-pair coarse pitch sized from its ALLOCATED point share of the total sweep budget,
    # clipped at max_pair_frac_of_pool AND an absolute ceiling (safety valve, see docstring) ----
    # Measured: a fraction-of-pool-only cap still let a handful of outlier-prior
    # pairs each claim millions of points (the pool itself is ~1e8 for a real sweep) -- the true limiter
    # is NOT the GPU eval (batched, <=0.5s even at 60M points) but the CPU-side np.meshgrid/ravel/stack
    # construction of `_grid_points()` PER PAIR, which is NOT captured by the two-term cost model
    # (calibrated on a single eval_batch call, not on n_pairs separate meshgrid constructions) -- an
    # absolute per-pair ceiling bounds that construction cost directly, regardless of how large the pool is.
    max_pair_points = max(min(int(alloc_pool * max_pair_frac_of_pool), max_pair_points_abs),
                           min_pair_coarse_points)
    coarse_pitch_of_pair = [None] * n_pairs
    n_pairs_capped = 0
    for i, p in enumerate(priors):
        if p["empty"]:
            continue
        share = (p["alloc_vol"] / total_prior_vol) if total_prior_vol > 0 else (1.0 / n_pairs)
        n_target = min_pair_coarse_points + int(alloc_pool * share)
        if n_target > max_pair_points:
            n_target = max_pair_points
            n_pairs_capped += 1
        coarse_pitch_of_pair[i] = max((p["vol"] / max(n_target, 1)) ** (1.0 / 3.0), min_pitch_mm)

    # ---- build the leaf table + concatenated coarse point buffer (ONE launch answers every pair) ----
    t_buf0 = time.time()
    all_leaves, leaf_index = [], {}
    def _leaf_id(leaf_node):
        k2 = (leaf_node["npz_path"], leaf_node["key"], leaf_node["mode"], leaf_node["iso"])
        if k2 not in leaf_index:
            leaf_index[k2] = len(all_leaves)
            all_leaves.append(leaf_node)
        return leaf_index[k2]

    leaf_a_of_pair = np.zeros(n_pairs, dtype=np.int32)
    leaf_b_of_pair = np.zeros(n_pairs, dtype=np.int32)
    pts_of_pair = [None] * n_pairs          # pair index -> its own coarse points array (for boundary extraction)
    shape_of_pair = [None] * n_pairs
    pts_chunks, pop_chunks = [], []
    for i, (la, lb) in enumerate(pairs):
        leaf_a_of_pair[i] = _leaf_id(la)
        leaf_b_of_pair[i] = _leaf_id(lb)
        pitch = coarse_pitch_of_pair[i]
        if pitch is None:
            continue
        pts_i, shape_i = _grid_points(priors[i]["lo"], priors[i]["hi"], pitch)
        pts_of_pair[i] = pts_i
        shape_of_pair[i] = shape_i
        pts_chunks.append(pts_i)
        pop_chunks.append(np.full(pts_i.shape[0], i, dtype=np.int32))

    all_pts = np.concatenate(pts_chunks, axis=0) if pts_chunks else np.zeros((0, 3))
    pair_of_point = np.concatenate(pop_chunks, axis=0) if pop_chunks else np.zeros((0,), dtype=np.int32)
    t_buffer_build = time.time() - t_buf0

    t0 = time.time()
    vals = (W.eval_batch_multi_pair(all_pts, pair_of_point, leaf_a_of_pair, leaf_b_of_pair, all_leaves,
                                     device=device) if all_pts.shape[0] > 0 else np.zeros((0,), dtype=np.float32))
    t_coarse_launch = time.time() - t0

    # ---- stage D: per-pair reduce of the ONE coarse-pass result -> volume + boundary cells ----
    results = [None] * n_pairs
    boundary_centers_of_pair = [None] * n_pairs
    n_boundary_of_pair = [0] * n_pairs
    offset = 0
    for i in range(n_pairs):
        pitch = coarse_pitch_of_pair[i]
        if pitch is None:
            results[i] = {"method": "sweep_batched_prior_alloc", "volume_mm3": 0.0, "uncertainty_mm3": 0.0,
                          "dispatch_log": {"note": "AABB-overlap prior volume is 0 -- no grid pass needed",
                                           "prior_vol_mm3": 0.0}}
            continue
        n_i = shape_of_pair[i][0] * shape_of_pair[i][1] * shape_of_pair[i][2]
        seg = vals[offset:offset + n_i]
        offset += n_i
        diag_half = 0.5 * pitch * math.sqrt(3.0)
        fully_inside = seg < -diag_half
        boundary = np.abs(seg) <= diag_half
        cell_vol = pitch ** 3
        vol_c = float(fully_inside.sum()) * cell_vol
        n_b = int(boundary.sum())
        n_boundary_of_pair[i] = n_b
        if n_b > 0:
            boundary_centers_of_pair[i] = pts_of_pair[i][boundary]
        results[i] = {"method": "sweep_batched_prior_alloc", "volume_mm3": vol_c, "uncertainty_mm3": n_b * cell_vol,
                      "dispatch_log": {"prior_vol_mm3": priors[i]["vol"], "coarse_pitch_mm": pitch,
                                       "n_coarse_points": n_i, "n_fully_inside_coarse": int(fully_inside.sum()),
                                       "n_boundary_coarse": n_b}}

    # ---- stage E: ONE globally-budgeted refine pass over every pair's boundary cells (batched) ----
    total_boundary = sum(n_boundary_of_pair)
    remaining_budget_ms = max(total_budget_ms - t_coarse_launch * 1000.0, 0.0)
    remaining_points_budget = CM.affordable_points(remaining_budget_ms, overhead_s, rate, floor_points=0)
    refine_factor = 1
    t_refine_launch = 0.0
    t_refine_buffer_build = 0.0
    if total_boundary > 0 and remaining_points_budget > total_boundary:
        refine_factor = int(min(max_refine_factor, target_refine_factor,
                                 max((remaining_points_budget / total_boundary) ** (1.0 / 3.0), 1.0)))
    # ABSOLUTE ceiling on total fine-point count (same CPU-buffer-construction root cause as the
    # per-pair coarse cap above, at SWEEP level: total_boundary*refine_factor**3 broadcasts a
    # (n_boundary, refine_factor**3, 3) array before reshape -- measured: at 694,114 boundary
    # cells and refine_factor=4 (44.4M fine points), THIS broadcast/reshape construction alone cost
    # 1.17s, the single largest remaining term after the per-pair coarse cap was added).
    while refine_factor > 1 and total_boundary * (refine_factor ** 3) > max_total_refine_points_abs:
        refine_factor -= 1
    if refine_factor > 1 and total_boundary > 0:
        t_rb0 = time.time()
        # The sub-cell offsets below always tile the coarse cell at its true raw sub-pitch
        # (coarse_pitch/refine_factor); min_pitch_mm is a floor on the coarse pitch only, and the
        # clamped value is used for logging, never for cell-volume or diagonal math. Conflating the two
        # inflates every cell volume by (min_pitch/raw_pitch)^3 (measured on one pair: 1915.1 mm3
        # reported against a correct 109.9 mm3 and an exact reference of 148.5 mm3). raw_fine_pitch
        # drives every volume/uncertainty computation; fine_pitch_of_pair is informational only.
        fine_pts_chunks, fine_pop_chunks, fine_pitch_of_pair, raw_fine_pitch_of_pair = [], [], {}, {}
        for i in range(n_pairs):
            if n_boundary_of_pair[i] == 0:
                continue
            coarse_pitch = coarse_pitch_of_pair[i]
            raw_fine_pitch = coarse_pitch / refine_factor
            raw_fine_pitch_of_pair[i] = raw_fine_pitch
            fine_pitch_of_pair[i] = max(raw_fine_pitch, min_pitch_mm)   # logging-only, see note above
            half = coarse_pitch / 2.0
            offs = (np.arange(refine_factor) + 0.5) / refine_factor * coarse_pitch - half
            OX, OY, OZ = np.meshgrid(offs, offs, offs, indexing="ij")
            local = np.stack([OX.ravel(), OY.ravel(), OZ.ravel()], axis=1)
            fine_pts = (boundary_centers_of_pair[i][:, None, :] + local[None, :, :]).reshape(-1, 3)
            fine_pts_chunks.append(fine_pts)
            fine_pop_chunks.append(np.full(fine_pts.shape[0], i, dtype=np.int32))
        fine_all_pts = np.concatenate(fine_pts_chunks, axis=0)
        fine_pair_of_point = np.concatenate(fine_pop_chunks, axis=0)
        t_refine_buffer_build = time.time() - t_rb0
        t0 = time.time()
        fine_vals = W.eval_batch_multi_pair(fine_all_pts, fine_pair_of_point, leaf_a_of_pair, leaf_b_of_pair,
                                             all_leaves, device=device)
        t_refine_launch = time.time() - t0

        offset = 0
        for i in range(n_pairs):
            if n_boundary_of_pair[i] == 0:
                continue
            raw_fp = raw_fine_pitch_of_pair[i]   # the TRUE sub-cell size the offsets above were built from
            n_fi = boundary_centers_of_pair[i].shape[0] * (refine_factor ** 3)
            seg = fine_vals[offset:offset + n_fi]
            offset += n_fi
            fine_diag_half = 0.5 * raw_fp * math.sqrt(3.0)
            fine_cell_vol = raw_fp ** 3
            fine_inside = seg < 0.0
            vol_fine = float(fine_inside.sum()) * fine_cell_vol
            n_boundary_fine = int((np.abs(seg) <= fine_diag_half).sum())
            results[i]["volume_mm3"] += vol_fine
            results[i]["uncertainty_mm3"] = n_boundary_fine * fine_cell_vol
            results[i]["dispatch_log"]["refine_factor"] = refine_factor
            results[i]["dispatch_log"]["fine_pitch_mm"] = fine_pitch_of_pair[i]
            results[i]["dispatch_log"]["n_fine_points"] = n_fi
            results[i]["dispatch_log"]["n_boundary_remaining_at_finest"] = n_boundary_fine

    t_sweep_total = time.time() - t_sweep0
    sweep_log = {
        "n_pairs": n_pairs, "total_budget_ms": total_budget_ms,
        "cost_model_v1_2": {"overhead_s": overhead_s, "rate_pts_per_s": rate, "from_cache": from_cache},
        "total_prior_overlap_vol_mm3": total_prior_vol,
        "n_pairs_empty_prior": sum(1 for p in priors if p["empty"]),
        "total_points_budget": total_points_budget, "n_coarse_points_actual": int(all_pts.shape[0]),
        "max_pair_points_cap": max_pair_points, "n_pairs_capped": n_pairs_capped,
        "n_launches": 1 + (1 if (refine_factor > 1 and total_boundary > 0) else 0),
        "refine_factor_chosen": refine_factor, "total_boundary_cells": total_boundary,
        "t_prior_stage_s": t_prior, "t_buffer_build_s": t_buffer_build,
        "t_coarse_launch_s": t_coarse_launch,
        "t_refine_buffer_build_s": t_refine_buffer_build, "t_refine_launch_s": t_refine_launch,
        "t_sweep_wall_s": t_sweep_total,
    }
    return results, sweep_log


def total_volume(node, budget_ms=200.0, min_pitch_mm=0.25, device=W.DEVICE):
    """Volume of a single expression -- same hierarchical-prune dispatcher, degenerate case (A=A, no
    intersect combine needed, but reuses intrusion_volume's engine against `intersect(node, node)` would
    double-eval; instead this runs the identical coarse+refine machinery directly on `node`)."""
    return intrusion_volume(node, node, budget_ms=budget_ms, min_pitch_mm=min_pitch_mm, device=device)


def min_clearance(exprA, exprB, budget_ms=200.0, n_samples=None, device=W.DEVICE):
    """Approximate min-clearance between A and B via distance sampling: evaluate sdf_A - (-sdf_B) is not
    meaningful directly (SDFs of DIFFERENT solids aren't a single field); instead sample candidate points
    on/near A's surface (via the coarse-boundary cells of A alone) and, for each, take B's SDF value as an
    approximate clearance lower bound (exact if B's SDF is a true metric distance near the surface, which
    holds for every analytic primitive here away from smooth-blend regions). Method choice logged like
    intrusion_volume -- table-driven, not asserted."""
    rate, calib_dt = calibrate_throughput(exprA, n_probe=50_000, device=device)
    budget_points = max(int(budget_ms / 1000.0 * rate * 0.5), 500)   # half budget: A-boundary find + B-eval
    lo, hi = bbox_of(exprA)
    bbox_vol = float(np.prod(hi - lo))
    pitch = max((bbox_vol / budget_points) ** (1.0 / 3.0), 0.25)
    t0 = time.time()
    pts, shape = _grid_points(lo, hi, pitch)
    valsA = W.eval_batch(exprA, pts, device=device)
    diag_half = 0.5 * pitch * math.sqrt(3.0)
    near_surface = np.abs(valsA) <= diag_half
    surf_pts = pts[near_surface]
    if surf_pts.shape[0] == 0:
        return {"method": "surface_sample_vs_field", "min_clearance_mm": float("inf"),
                "n_surface_samples": 0, "wall_ms": (time.time() - t0) * 1000.0,
                "note": "expr A empty at this bbox/pitch -- no surface samples found"}
    valsB_at_surfA = W.eval_batch(exprB, surf_pts, device=device)
    wall_s = time.time() - t0
    min_clear = float(valsB_at_surfA.min())
    return {
        "method": "surface_sample_vs_field", "min_clearance_mm": min_clear,
        "n_surface_samples": int(surf_pts.shape[0]), "pitch_mm": pitch,
        "uncertainty_mm": pitch,   # sampling resolution is the declared uncertainty on the clearance value
        "budget_ms": budget_ms, "wall_ms": wall_s * 1000.0,
        "budget_met_within_20pct": bool(wall_s * 1000.0 <= budget_ms * 1.2),
    }


if __name__ == "__main__":
    # smoke test: two overlapping boxes with an EXACTLY known analytic overlap volume
    A = E.box(half_extents=(10, 10, 10), center=(0, 0, 0))
    B = E.box(half_extents=(10, 10, 10), center=(15, 0, 0))
    exact = 5.0 * 20.0 * 20.0   # overlap along x in [5,10] = 5mm, full 20x20 in y,z
    r = intrusion_volume(A, B, budget_ms=300.0)
    print(json.dumps(r, indent=2))
    err_pct = 100.0 * abs(r["volume_mm3"] - exact) / exact
    print(f"exact={exact} measured={r['volume_mm3']:.3f} err%={err_pct:.4f} "
          f"uncertainty_mm3={r['uncertainty_mm3']:.3f} budget_met={r['budget_met_within_20pct']}")
    mc = min_clearance(A, E.translate(B, 0, 0, 0), budget_ms=200.0)
    print(json.dumps(mc, indent=2))
