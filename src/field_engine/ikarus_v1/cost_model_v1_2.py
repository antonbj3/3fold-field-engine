#!/usr/bin/env python3
"""Deterministic two-term dispatch cost model for the query engine.

A single-probe wall-clock calibration divides n_probe/dt and yields a rate with no fixed-cost term, so a
sizing rule of the form wall_time = n_points / rate over-estimates how many points a budget affords
whenever the probe itself was overhead-dominated. This module models

    wall_time(n) = overhead_s + n / rate_pts_per_s

fitted once from two probe sizes via the line through (n_lo, t_lo) and (n_hi, t_hi):
    rate = (n_hi - n_lo) / (t_hi - t_lo);  overhead = t_lo - n_lo / rate
Each probe point is the median of several repeated timings.

The fitted (overhead_s, rate_pts_per_s) is cached to disk keyed by (device, kernel-shape hash), so the
wall clock is sampled only the first time a given pair is calibrated and every later call in any process
reads the same numbers back; sizing decisions are then bit-identical run to run.

Main entry points: calibrate_two_term(shape_key, eval_fn, lo, hi, device) -> (overhead_s, rate,
from_cache); affordable_points(budget_ms, overhead_s, rate) -> point count.
"""
from __future__ import annotations

import json
import os
import statistics
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CACHE_PATH = os.path.join(HERE, "_cost_model_cache_v1_2.json")


def _load_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(path, cache):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _median_timed_eval(eval_fn, n_points, lo, hi, rng, reps=5):
    """Median wall time (seconds) of `reps` repeated eval_fn(random (n_points,3) points) calls --
    median (not mean) so a single OS-scheduling outlier doesn't move the fitted line."""
    times = []
    for _ in range(reps):
        pts = rng.uniform(lo, hi, size=(n_points, 3))
        t0 = time.time()
        eval_fn(pts)
        times.append(time.time() - t0)
    return statistics.median(times)


def calibrate_two_term(shape_key, eval_fn, bbox_lo, bbox_hi, device,
                        n_lo=20_000, n_hi=400_000, reps=5,
                        cache_path=_DEFAULT_CACHE_PATH, force_remeasure=False):
    """Returns (overhead_s, rate_pts_per_s, from_cache: bool) for the given (shape_key, device).

    `eval_fn(points_nx3) -> values` is the ALREADY-COMPILED evaluator to probe (caller is responsible
    for warming the kernel cache with one throwaway call first, exactly as v1's calibrate_throughput did
    -- this function measures STEADY-STATE dispatch cost, not compile latency).
    """
    cache = _load_cache(cache_path)
    key = f"{shape_key}::{device}"
    if not force_remeasure and key in cache:
        e = cache[key]
        return e["overhead_s"], e["rate_pts_per_s"], True

    rng = np.random.default_rng(0)
    t_lo = _median_timed_eval(eval_fn, n_lo, bbox_lo, bbox_hi, rng, reps=reps)
    t_hi = _median_timed_eval(eval_fn, n_hi, bbox_lo, bbox_hi, rng, reps=reps)
    dn = n_hi - n_lo
    dt = t_hi - t_lo
    if dn <= 0 or dt <= 0:
        # degenerate probe (can happen on a near-instant no-op kernel) -- fall back to the single-point
        # zero-intercept estimate rather than a negative/undefined rate; declared, not silently clipped.
        rate = n_hi / max(t_hi, 1e-6)
        overhead = 0.0
    else:
        rate = dn / dt
        overhead = max(t_lo - n_lo / rate, 0.0)

    cache[key] = {
        "overhead_s": overhead, "rate_pts_per_s": rate,
        "n_lo": n_lo, "t_lo_s": t_lo, "n_hi": n_hi, "t_hi_s": t_hi, "reps": reps,
        "measured_at_unix": time.time(),
    }
    _save_cache(cache_path, cache)
    return overhead, rate, False


def affordable_points(budget_ms, overhead_s, rate_pts_per_s, floor_points=64):
    """Two-term inverse: how many points fit in budget_ms given the fitted (overhead, rate)? Floors at
    `floor_points` (never zero -- a dispatch that can't even afford the floor still runs it and the
    caller declares the budget as blown, rather than silently returning an empty/undefined grid)."""
    remaining_s = budget_ms / 1000.0 - overhead_s
    if remaining_s <= 0:
        return floor_points
    return max(int(remaining_s * rate_pts_per_s), floor_points)


def predicted_wall_ms(n_points, overhead_s, rate_pts_per_s):
    return (overhead_s + n_points / max(rate_pts_per_s, 1e-9)) * 1000.0
