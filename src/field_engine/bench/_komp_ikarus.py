#!/usr/bin/env python3
"""Benchmark component: the SDF expression-tree kernel (`ikarus_v1`, code-generated Warp evaluator).

Sizes: N = 1e5 / 1e6 / 1e7 points, batch-evaluated against a sphere tree. Correctness:
max |sdf - (|p| - r)| and `queries.total_volume` (hierarchical prune) against 4/3 pi r^3.
Compilation cost: a union tree of 10 and 50 leaves, cold (new shape) against warm (cached kernel),
because the build dominates for large trees. Part: analytic primitives only, none read from disk.
Writes bench/artifacts/_parts/ikarus.json.
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bench_common as BC  # noqa: E402

BC.repo_paths()
SNABB = BC.snabb()
DEV = BC.device()
HAVE_CUDA = BC.have_cuda()

import expr as E  # noqa: E402
import eval_warp as W  # noqa: E402

R = 20.0
NS = [100_000, 1_000_000] + ([10_000_000] if HAVE_CUDA else [])
if SNABB:
    NS = [100_000]
LABEL = {100_000: "S", 1_000_000: "M", 10_000_000: "L"}


def _pts(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-30, 30, size=(n, 3)).astype(np.float32)


def del_storlekar():
    node = E.sphere(R)
    rows = []
    for n in NS:
        pts = _pts(n)
        t, vals = BC.timeit(lambda: W.eval_batch(node, pts, device=DEV), n_reps=3 if SNABB else 5)
        err = np.abs(vals - (np.linalg.norm(pts, axis=1) - R))
        rows.append(dict(label=LABEL.get(n, str(n)), n_punkter=n, p50_s=t["p50_s"], p95_s=t["p95_s"],
                         punkter_per_s=n / t["p50_s"], max_abs_fel_mm=float(err.max()),
                         GRON=bool(err.max() < 1e-4)))
    return dict(rader=rows, device=DEV,
                not_="eval_batch includes the host-to-device copy of the points and the copy back, "
                     "which is how a caller invokes it")


def del_kompilering():
    """Compilation time per tree size (10 and 50 leaves), cold (new shape) against cached."""
    rows = []
    for n_leaf in ((10,) if SNABB else (10, 50)):
        node = E.box((3.0, 3.0, 3.0), center=(0.0, 0.0, 0.0))
        for i in range(1, n_leaf):
            node = E.union(node, E.box((3.0, 3.0, 3.0), center=(i * 2.5 + 0.01 * n_leaf, 0.0, 0.0)))
        pts = _pts(100_000)
        t0 = time.perf_counter()
        W.eval_batch(node, pts, device=DEV)
        t_kall = time.perf_counter() - t0
        t, _ = BC.timeit(lambda: W.eval_batch(node, pts, device=DEV), n_reps=3)
        rows.append(dict(n_noder=E.node_count(node), t_kall_kompilering_s=t_kall, t_varm_p50_s=t["p50_s"],
                         kompilering_over_eval=t_kall / t["p50_s"]))
    return dict(rader=rows, device=DEV)


def del_total_volume():
    import queries as Q
    node = E.sphere(R)
    va = 4.0 / 3.0 * math.pi * R ** 3
    t0 = time.perf_counter()
    r = Q.total_volume(node, budget_ms=200.0, device=DEV)
    wall = time.perf_counter() - t0
    vol = r.get("intrusion_volume_mm3", r.get("volume_mm3"))
    if vol is None:
        for k, v in r.items():
            if "volume" in k and isinstance(v, (int, float)):
                vol = v
                break
    return dict(volym_mm3=vol, volym_analytisk_mm3=va,
                rel_fel=(abs(vol - va) / va) if vol is not None else None, wall_s=wall,
                GRON=bool(vol is not None and abs(vol - va) / va < 0.02))


def del_determinism():
    node = E.sphere(R)
    pts = _pts(100_000 if SNABB else 1_000_000)
    v1 = W.eval_batch(node, pts, device=DEV)
    v2 = W.eval_batch(node, pts, device=DEV)
    return dict(n_punkter=int(len(pts)), sha_run1=BC.sha_arr(v1), sha_run2=BC.sha_arr(v2),
                bit_identisk=BC.sha_arr(v1) == BC.sha_arr(v2), device=DEV)


def del_gpu_vs_cpu():
    """The same batch on the CUDA device and on the Warp CPU backend. CUDA-only."""
    node = E.sphere(R)
    rows = []
    for n in NS[:2]:
        pts = _pts(n)
        tg, vg = BC.timeit(lambda: W.eval_batch(node, pts, device="cuda:0"), n_reps=5)
        tc, vc = BC.timeit(lambda: W.eval_batch(node, pts, device="cpu"), n_reps=3)
        rows.append(dict(n_punkter=n, gpu_p50_s=tg["p50_s"], cpu_p50_s=tc["p50_s"],
                         gpu_speedup=tc["p50_s"] / tg["p50_s"],
                         max_abs_diff_cpu_gpu=float(np.max(np.abs(vg - vc)))))
    return dict(rader=rows)


if __name__ == "__main__":
    P = BC.Part("ikarus")
    P.kor("storlekar_SML", del_storlekar)
    P.kor("kompilering_tradstorlek", del_kompilering)
    P.kor("total_volume_sfar", del_total_volume)
    P.kor("determinism", del_determinism)
    P.kor("gpu_vs_cpu", del_gpu_vs_cpu, gpu=True)
    P.d["vram_process_mb"] = BC.gpu_proc_used_mb()
    P.skriv()
