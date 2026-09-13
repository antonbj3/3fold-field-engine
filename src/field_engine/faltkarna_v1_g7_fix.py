#!/usr/bin/env python3
"""GPU-resident variant of the sparse field reconstruction, with correctness and benchmark gates.

rekonstruera_tat_gpu() in faltkarna_v1.py takes sf.kind/sf.tiles, which have already been downloaded
to numpy, and uploads them again before the scatter kernels run: two unnecessary host round trips on
the large tile array. The variant here keeps the tile array on the device between classification,
evaluation and reconstruction; the only host transfers are the small kind array (needed for np.where
compaction) and the final padded array (needed by skimage marching cubes).

Three variants are compared:
  V0 = klassificera_och_evaluera_testfalt + rekonstruera_tat        (host scatter)
  V1 = klassificera_och_evaluera_testfalt + rekonstruera_tat_gpu    (device scatter, two round trips)
  V2 = klassificera_evaluera_rekonstruera_gpu_resident (this file)  (device resident)

Takes no arguments; writes artifacts/faltkarna_v1_g7_fix.json next to this file with the correctness
gate (all three dense arrays must be bit-identical), the two planted-fault tests re-run on V2, and a
three-size benchmark against a trivial dense one-thread-per-voxel reference.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import faltkarna_v1 as FK  # noqa: E402
import warp as wp  # noqa: E402

REPORT = os.path.join(HERE, "artifacts", "faltkarna_v1_g7_fix.json")

HX, HY, HZ, R = 20.0, 15.0, 10.0, 6.0
VOL_ANALYTIC = 8.0 * HX * HY * HZ - math.pi * R * R * (2.0 * HZ)
BLOCK = 8
DEVICE = "cuda:0" if FK.HAVE_CUDA else "cpu"


def make_meta(pitch):
    pad = 1.0
    nx = int(round((2 * HX + 2 * pad) / pitch)) + 1
    ny = int(round((2 * HY + 2 * pad) / pitch)) + 1
    nz = int(round((2 * HZ + 2 * pad) / pitch)) + 1
    return FK.GridMeta(x0=-HX - pad, y0=-HY - pad, z0=-HZ - pad, pitch=pitch, nx=nx, ny=ny, nz=nz,
                        block=BLOCK)


# ================================================================================================
# V2: classify + evaluate + reconstruct inside one wp.ScopedDevice context; no tile array leaves the
# device before the final padded array. It reuses the kernels in faltkarna_v1.py unchanged, so this is
# an orchestration change (which .numpy() calls are made), not new mathematics.
# ================================================================================================
def klassificera_evaluera_rekonstruera_gpu_resident(meta: FK.GridMeta, hx, hy, hz, r,
                                                      corrupt_index=False, device=DEVICE):
    b = meta.block
    b3 = b ** 3
    pnx, pny, pnz = meta.n_bx * b, meta.n_by * b, meta.n_bz * b
    t = {}
    with wp.ScopedDevice(device):
        t0 = time.time()
        kind_w = wp.zeros(meta.n_blocks, dtype=wp.int32)
        cma_w = wp.zeros(meta.n_blocks, dtype=wp.float32)
        wp.launch(FK.k_classify_testfalt, dim=meta.n_blocks,
                  inputs=[meta.n_bx, meta.n_by, meta.n_bz, meta.block,
                          meta.x0, meta.y0, meta.z0, meta.pitch, hx, hy, hz, r, meta.margin_mm],
                  outputs=[kind_w, cma_w])
        wp.synchronize()
        t["t_classify_s"] = time.time() - t0

        # the only host download needed for compaction: the small kind array (n_blocks int32),
        # not the tile data.
        t1 = time.time()
        kind_np = kind_w.numpy()
        active_ids = np.where(kind_np == 0)[0].astype(np.int32)
        n_active = len(active_ids)
        active_w = wp.array(active_ids, dtype=wp.int32)
        t["t_compact_s"] = time.time() - t1

        t2 = time.time()
        tiles_w = wp.zeros(n_active * b3, dtype=wp.float32) if n_active > 0 else wp.zeros(1, dtype=wp.float32)
        if n_active > 0:
            wp.launch(FK.k_eval_tile_testfalt, dim=n_active * b3,
                      inputs=[active_w, meta.n_by, meta.n_bz, meta.block,
                              meta.x0, meta.y0, meta.z0, meta.pitch, hx, hy, hz, r,
                              1 if corrupt_index else 0],
                      outputs=[tiles_w])
        wp.synchronize()
        t["t_eval_s"] = time.time() - t2

        # tiles_w/kind_w go straight into the scatter kernels: no .numpy()/wp.array() round trip
        # (compare faltkarna_v1.rekonstruera_tat_gpu, which does exactly that).
        t3 = time.time()
        out_w = wp.zeros(pnx * pny * pnz, dtype=wp.float32)
        wp.launch(FK.k_fill_background, dim=meta.n_blocks * b3,
                  inputs=[kind_w, meta.n_by, meta.n_bz, b, pnx, pny, pnz, FK.BIG], outputs=[out_w])
        if n_active > 0:
            wp.launch(FK.k_scatter_tiles, dim=n_active * b3,
                      inputs=[active_w, tiles_w, meta.n_by, meta.n_bz, b, pny, pnz], outputs=[out_w])
        wp.synchronize()
        t["t_reconstruct_s"] = time.time() - t3

        t4 = time.time()
        padded = out_w.numpy().reshape(pnx, pny, pnz)
        t["t_final_download_s"] = time.time() - t4
    t["t_total_s"] = sum(t.values())
    dense = padded[:meta.nx, :meta.ny, :meta.nz]
    stats = dict(n_blocks=int(meta.n_blocks), n_active=int(n_active),
                 active_voxels=int(n_active * b3), dense_voxels=int(meta.nx * meta.ny * meta.nz))
    return dense, t, stats


# ================================================================================================
# Correctness gate: V0/V1/V2 must give a bit-identical dense array (same input, deterministic CSG).
# ================================================================================================
def korrekthetsgrind(meta):
    sf = FK.klassificera_och_evaluera_testfalt(meta, HX, HY, HZ, R, corrupt_index=False, device=DEVICE)
    dense_v0 = FK.rekonstruera_tat(sf)
    dense_v1 = FK.rekonstruera_tat_gpu(sf, device=DEVICE)
    dense_v2, _t, _s = klassificera_evaluera_rekonstruera_gpu_resident(meta, HX, HY, HZ, R, device=DEVICE)
    d01 = float(np.max(np.abs(dense_v0 - dense_v1)))
    d02 = float(np.max(np.abs(dense_v0 - dense_v2)))
    vol_v0 = float(np.sum(dense_v0 < 0.0)) * meta.pitch ** 3
    vol_v2 = float(np.sum(dense_v2 < 0.0)) * meta.pitch ** 3
    check_v2 = FK.konservering_check(vol_v2, VOL_ANALYTIC, tol_frac=0.03)
    return dict(max_abs_dev_v0_vs_v1=d01, max_abs_dev_v0_vs_v2=d02,
                bit_identical_v0_vs_v1=bool(d01 == 0.0), bit_identical_v0_vs_v2=bool(d02 == 0.0),
                vol_v0_mm3=vol_v0, vol_v2_mm3=vol_v2, konservering_v2=check_v2,
                PASS=bool(d01 == 0.0 and d02 == 0.0 and check_v2["GRON"]))


# ================================================================================================
# The inherited planted-fault tests, re-run on the V2 path.
# ================================================================================================
def fallbevis_v2(meta):
    dense_bad, _t, _s = klassificera_evaluera_rekonstruera_gpu_resident(meta, HX, HY, HZ, R,
                                                                         corrupt_index=True, device=DEVICE)
    vol_bad = float(np.sum(dense_bad < 0.0)) * meta.pitch ** 3
    check_bad = FK.konservering_check(vol_bad, VOL_ANALYTIC, tol_frac=0.03)
    fb_i = dict(vol_voxel_mm3=vol_bad, check=check_bad, GRIND_FALLER_SOM_FORVANTAT=bool(not check_bad["GRON"]))

    sf = FK.klassificera_och_evaluera_testfalt(meta, HX, HY, HZ, R, corrupt_index=False, device=DEVICE)
    det = FK.volym_via_gpu_reduktion(sf, n_korningar=6)
    fb_ii = dict(det, GRIND_FALLER_F32_SOM_FORVANTAT=bool(not det["f32_unsafe"]["bit_identiska"]),
                 GRIND_HALLER_I64_SOM_FORVANTAT=bool(det["i64_fixpunkt_safe"]["bit_identiska"]))
    return dict(fallbevis_i_korrupt_index_pa_v2=fb_i, fallbevis_ii_determinism_pa_v2=fb_ii)


# ================================================================================================
# Benchmark gate: three sizes, warm median of five, V0/V1/V2 plus a trivial dense reference.
# ================================================================================================
def _dense_ref_kernel_time(meta, n_reps=5):
    """Trivial one-thread-per-voxel dense reference (same mathematics, no sparsity).

    Takes grid metadata and a repetition count, returns median/all kernel times and the voxel count.
    """
    @wp.kernel
    def k_dense_ref(nx: int, ny: int, nz: int, x0: float, y0: float, z0: float, pitch: float,
                     hx: float, hy: float, hz: float, r: float, out: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        k = tid % nz
        j = (tid // nz) % ny
        i = tid // (nz * ny)
        x = x0 + float(i) * pitch
        y = y0 + float(j) * pitch
        z = z0 + float(k) * pitch
        out[tid] = FK.testfalt(x, y, z, hx, hy, hz, r)

    n = meta.nx * meta.ny * meta.nz
    times = []
    with wp.ScopedDevice(DEVICE):
        out_w = wp.zeros(n, dtype=wp.float32)
        for _ in range(n_reps):
            wp.synchronize()
            t0 = time.time()
            wp.launch(k_dense_ref, dim=n, inputs=[meta.nx, meta.ny, meta.nz, meta.x0, meta.y0, meta.z0,
                                                    meta.pitch, HX, HY, HZ, R], outputs=[out_w])
            wp.synchronize()
            times.append(time.time() - t0)
    return dict(median_s=float(np.median(times)), all_s=times, dense_voxels=int(n))


def benchmark_size(label, pitch, n_reps=5):
    meta = make_meta(pitch)
    n_dense = meta.nx * meta.ny * meta.nz
    print(f"[g7-fix] size={label} pitch={pitch} dense_voxels={n_dense:.3g} ...", file=sys.stderr)

    # warm-up (JIT compilation, device init), not measured
    FK.klassificera_och_evaluera_testfalt(meta, HX, HY, HZ, R, device=DEVICE)
    klassificera_evaluera_rekonstruera_gpu_resident(meta, HX, HY, HZ, R, device=DEVICE)

    v0_times, v1_times, v2_times = [], [], []
    n_active_last = None
    for _ in range(n_reps):
        sf = FK.klassificera_och_evaluera_testfalt(meta, HX, HY, HZ, R, device=DEVICE)
        t0 = time.time()
        _ = FK.rekonstruera_tat(sf)
        v0_times.append(time.time() - t0)

        sf2 = FK.klassificera_och_evaluera_testfalt(meta, HX, HY, HZ, R, device=DEVICE)
        t1 = time.time()
        _ = FK.rekonstruera_tat_gpu(sf2, device=DEVICE)
        v1_times.append(time.time() - t1)

        _, t_v2, stats = klassificera_evaluera_rekonstruera_gpu_resident(meta, HX, HY, HZ, R, device=DEVICE)
        v2_times.append(t_v2["t_total_s"])
        n_active_last = stats["n_active"]

    ref = _dense_ref_kernel_time(meta, n_reps=n_reps)
    dense_MB = n_dense * 4 / 1e6
    gles_MB = (n_active_last or 0) * BLOCK ** 3 * 4 / 1e6
    return dict(label=label, pitch=pitch, n_dense_voxels=int(n_dense), n_active_blocks=n_active_last,
                dense_MB=dense_MB, gles_MB=gles_MB,
                gles_over_dense_frac=(gles_MB / dense_MB if dense_MB else None),
                v0_cpu_scatter_median_s=float(np.median(v0_times)), v0_all_s=v0_times,
                v1_gpu_scatter_roundtrip_median_s=float(np.median(v1_times)), v1_all_s=v1_times,
                v2_gpu_resident_g7fix_median_s=float(np.median(v2_times)), v2_all_s=v2_times,
                dense_ref_kernel_median_s=ref["median_s"],
                gles_ge_tat_v0=bool(np.median(v0_times) <= ref["median_s"]),
                gles_ge_tat_v2=bool(np.median(v2_times) <= ref["median_s"]),
                v2_speedup_over_v1=(float(np.median(v1_times)) / float(np.median(v2_times))
                                     if np.median(v2_times) > 0 else None))


def main():
    out = {"cell": "faltkarna_v1_g7_fix", "device": DEVICE, "params": dict(hx=HX, hy=HY, hz=HZ, r=R,
                                                                            vol_analytic_mm3=VOL_ANALYTIC)}
    meta_pilot = make_meta(0.25)
    out["korrekthetsgrind"] = korrekthetsgrind(meta_pilot)
    out["fallbevis"] = fallbevis_v2(meta_pilot)

    sizes = [("S_liten", 0.9), ("M_f4pilot_skala", 0.25), ("L_stor", 0.09)]
    if os.environ.get("FALTKARNA_BENCH_SIZES") == "SM":
        sizes = sizes[:2]
    out["benchmarkgrind"] = [benchmark_size(lbl, pitch) for lbl, pitch in sizes]

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "benchmarkgrind"}, indent=2))
    for b in out["benchmarkgrind"]:
        print(f"  {b['label']}: dense={b['n_dense_voxels']:.3g}vox gles/dense={b['gles_over_dense_frac']:.3f} "
              f"V0(cpu)={b['v0_cpu_scatter_median_s']*1e3:.2f}ms V1(gpu_roundtrip)={b['v1_gpu_scatter_roundtrip_median_s']*1e3:.2f}ms "
              f"V2(g7fix)={b['v2_gpu_resident_g7fix_median_s']*1e3:.2f}ms dense_ref={b['dense_ref_kernel_median_s']*1e3:.2f}ms "
              f"v2_speedup_over_v1={b['v2_speedup_over_v1']:.2f}x")
    print("\nwrote", REPORT)


if __name__ == "__main__":
    main()
