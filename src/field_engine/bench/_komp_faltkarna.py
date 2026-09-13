#!/usr/bin/env python3
"""Benchmark component: the sparse block-SDF field kernel.

Reuses the kernel's own entry points rather than reimplementing them:
`faltkarna_v1_g7_fix.benchmark_size` (declared S/M/L sizes, variants V0/V1/V2 plus a dense
reference), `faltkarna_v1.klassificera_och_evaluera_testfalt` (one op on the CPU or the CUDA
device), `faltkarna_v1.volym_via_gpu_reduktion` (float32-atomic vs int64 fixed-point determinism)
and `faltkarna_v1_mesh_to_sdf.klassificera_och_evaluera_fran_tatt_falt` (sphere-SDF correctness).

Part: the analytic test field (a box minus a through cylinder) and an analytic sphere; no geometry
is read from disk. Sizes S/M run on the CPU Warp backend, size L (41 M voxels) needs a CUDA device.
Writes bench/artifacts/_parts/faltkarna.json.
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
P = BC.Part("faltkarna")
SNABB = BC.snabb()

import warp as wp  # noqa: E402
import faltkarna_v1 as FK  # noqa: E402
import faltkarna_v1_g7_fix as G7  # noqa: E402
import faltkarna_v1_mesh_to_sdf as M2S  # noqa: E402

DEV = BC.device()
HAVE_CUDA = BC.have_cuda()
# the same declared sizes as the kernel's own benchmark gate (44 K / 1.94 M / 41 M voxels)
SIZES = [("S", 0.9), ("M", 0.25)] + ([("L", 0.09)] if HAVE_CUDA else [])
if SNABB:
    SIZES = [("S", 0.9)]


def _volym_mc(dense, pitch):
    from skimage import measure
    import trimesh
    v, f, _, _ = measure.marching_cubes(dense, level=0.0, spacing=(pitch, pitch, pitch))
    m = trimesh.Trimesh(v, f, process=False)
    return float(abs(m.volume)), int(len(f))


def del_analytiskt_testfalt():
    """Throughput, p50/p95 latency and memory per declared size, via the kernel's benchmark_size."""
    out = dict(storlekar=[], device=DEV)
    for lbl, pitch in SIZES:
        free0 = wp.get_device(DEV).free_memory if HAVE_CUDA else None
        b = G7.benchmark_size(lbl, pitch, n_reps=3 if SNABB else 5)
        free1 = wp.get_device(DEV).free_memory if HAVE_CUDA else None
        n = b["n_dense_voxels"]
        v2 = np.array(b["v2_all_s"])
        out["storlekar"].append(dict(
            label=lbl, pitch_mm=pitch, n_dense_voxels=n, n_active_blocks=b["n_active_blocks"],
            dense_MB=b["dense_MB"], gles_MB=b["gles_MB"], gles_over_dense=b["gles_over_dense_frac"],
            v2_resident_p50_s=BC.pct(v2, 50), v2_resident_p95_s=BC.pct(v2, 95),
            v0_cpu_scatter_p50_s=float(np.median(b["v0_all_s"])),
            v1_roundtrip_p50_s=float(np.median(b["v1_all_s"])),
            dense_ref_kernel_p50_s=b["dense_ref_kernel_median_s"],
            voxlar_per_s_v2=n / BC.pct(v2, 50),
            voxlar_per_s_dense_ref=n / b["dense_ref_kernel_median_s"],
            gles_slar_tat_i_tid=b["gles_ge_tat_v2"],
            vram_delta_MB=((free0 - free1) / 1e6) if (free0 is not None and free1 is not None) else None))
    return out


def del_korrekthet_konservering():
    """Correctness: the CSG volume of the box minus a through cylinder, marching-cubed from the
    reconstructed dense field, against the closed form. Tolerance 1e-3 relative, from the kernel."""
    pitch = 0.5 if SNABB else 0.25
    meta = G7.make_meta(pitch)
    dense, t, stats = G7.klassificera_evaluera_rekonstruera_gpu_resident(meta, G7.HX, G7.HY, G7.HZ, G7.R, device=DEV)
    t0 = time.perf_counter()
    vol, nf = _volym_mc(dense, pitch)
    t_mc = time.perf_counter() - t0
    k = FK.konservering_check(vol, G7.VOL_ANALYTIC, tol_frac=1e-3)
    return dict(pitch_mm=pitch, n_dense_voxels=int(dense.size), volym_mc_mm3=vol,
                volym_analytisk_mm3=G7.VOL_ANALYTIC, rel_fel=k["diff_frac"], tol=1e-3, GRON=k["GRON"],
                n_faces_mc=nf, t_marching_cubes_cpu_s=t_mc, t_falt_s=t["t_total_s"],
                andel_mc_av_total=t_mc / (t_mc + t["t_total_s"]))


def del_determinism():
    """Two runs bit-identical (sha over the dense array) plus the float32-atomic against the
    int64 fixed-point reduction over three processes."""
    pitch = 0.5 if SNABB else 0.25
    meta = G7.make_meta(pitch)
    d1, _, _ = G7.klassificera_evaluera_rekonstruera_gpu_resident(meta, G7.HX, G7.HY, G7.HZ, G7.R, device=DEV)
    d2, _, _ = G7.klassificera_evaluera_rekonstruera_gpu_resident(meta, G7.HX, G7.HY, G7.HZ, G7.R, device=DEV)
    s1, s2 = BC.sha_arr(d1), BC.sha_arr(d2)
    sf = FK.klassificera_och_evaluera_testfalt(meta, G7.HX, G7.HY, G7.HZ, G7.R, device=DEV)
    tmp = os.environ.get("BENCH_TMP", "/tmp")
    red = FK.volym_via_gpu_reduktion(sf, n_korningar=3, tmp_dir=tmp)
    return dict(dense_sha_run1=s1, dense_sha_run2=s2, bit_identisk=s1 == s2, device=DEV,
                reduktion_f32_atomic_bit_identisk=red["f32_unsafe"]["bit_identiska"],
                reduktion_f32_varden=red["f32_unsafe"]["varden"],
                reduktion_i64_fixpunkt_bit_identisk=red["i64_fixpunkt_safe"]["bit_identiska"],
                n_processer=red["n_processer"], n_active_voxels=red["n_active_voxels"])


def del_gpu_vs_cpu():
    """The same op on the CUDA device and on the Warp CPU backend, sizes S and M. CUDA-only."""
    rows = []
    for lbl, pitch in SIZES[:2]:
        meta = G7.make_meta(pitch)
        n = meta.nx * meta.ny * meta.nz
        tg, _ = BC.timeit(lambda: FK.klassificera_och_evaluera_testfalt(meta, G7.HX, G7.HY, G7.HZ, G7.R, device="cuda:0"), n_reps=5)
        tc, _ = BC.timeit(lambda: FK.klassificera_och_evaluera_testfalt(meta, G7.HX, G7.HY, G7.HZ, G7.R, device="cpu"), n_reps=3)
        rows.append(dict(label=lbl, n_dense_voxels=int(n), gpu_p50_s=tg["p50_s"], cpu_p50_s=tc["p50_s"],
                         gpu_speedup=tc["p50_s"] / tg["p50_s"],
                         cpu_tradar="1 (the Warp CPU device is single-threaded; 1/2/4 scaling does not apply)"))
    return dict(rader=rows)


def del_sfar_sdf():
    """Correctness of the generic classifier on an analytic sphere field, three checks:
    (1) the Lipschitz classification never lies (INTERIOR blocks all negative, EXTERIOR all
    positive), (2) active tiles are bit-exact against the source field, (3) the marching-cubes
    volume of the reconstruction against 4/3 pi r^3."""
    r = 20.0
    rows = []
    pitches = [1.0] if SNABB else [1.0, 0.5]
    for pitch in pitches:
        n = int(round(2 * (r + 4.0) / pitch)) + 1
        ax = -(r + 4.0) + pitch * np.arange(n)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        dense = (np.sqrt(X * X + Y * Y + Z * Z) - r).astype(np.float32)
        t0 = time.perf_counter()
        sf = M2S.klassificera_och_evaluera_fran_tatt_falt(wp, dense, pitch, 8, DEV)
        t_cls = time.perf_counter() - t0
        kind = sf["kind"].reshape(sf["n_bx"], sf["n_by"], sf["n_bz"])
        B = 8
        pad = np.full((sf["n_bx"] * B, sf["n_by"] * B, sf["n_bz"] * B), np.nan, np.float32)
        pad[:n, :n, :n] = dense
        bv = pad.reshape(sf["n_bx"], B, sf["n_by"], B, sf["n_bz"], B).transpose(0, 2, 4, 1, 3, 5)
        inre_ok = bool(np.all(np.nan_to_num(bv[kind == -1], nan=-1.0) < 0.0)) if np.any(kind == -1) else True
        ytre_ok = bool(np.all(np.nan_to_num(bv[kind == 1], nan=1.0) > 0.0)) if np.any(kind == 1) else True
        bi, bj, bk = np.unravel_index(sf["active_ids"], (sf["n_bx"], sf["n_by"], sf["n_bz"]))
        src_tiles = bv[bi, bj, bk]
        m = ~np.isnan(src_tiles)
        tiles_exakta = bool(np.array_equal(src_tiles[m], sf["tiles"][m]))
        rec = np.where(kind < 0, -FK.BIG, FK.BIG).astype(np.float32)
        padr = np.empty_like(pad)
        bvr = padr.reshape(sf["n_bx"], B, sf["n_by"], B, sf["n_bz"], B).transpose(0, 2, 4, 1, 3, 5)
        bvr[:] = rec[:, :, :, None, None, None]
        bvr[bi, bj, bk] = sf["tiles"]
        vol, nf = _volym_mc(padr[:n, :n, :n], pitch)
        va = 4.0 / 3.0 * math.pi * r ** 3
        rows.append(dict(pitch_mm=pitch, n_dense_voxels=int(dense.size), n_blocks=sf["n_blocks"],
                         n_active=sf["n_active"], andel_aktiva=sf["n_active"] / sf["n_blocks"],
                         t_klassificera_evaluera_s=t_cls, voxlar_per_s=dense.size / t_cls,
                         inre_block_alla_negativa=inre_ok, ytre_block_alla_positiva=ytre_ok,
                         aktiva_tiles_bit_exakta=tiles_exakta, volym_mc_mm3=vol, volym_analytisk_mm3=va,
                         rel_fel=abs(vol - va) / va,
                         GRON=bool(inre_ok and ytre_ok and tiles_exakta and abs(vol - va) / va < 2e-3)))
    return dict(r_mm=r, rader=rows)


if __name__ == "__main__":
    P.kor("analytiskt_testfalt_SML", del_analytiskt_testfalt)
    P.kor("korrekthet_konservering", del_korrekthet_konservering)
    P.kor("determinism", del_determinism)
    P.kor("gpu_vs_cpu", del_gpu_vs_cpu, gpu=True)
    P.kor("sfar_sdf_generisk_klassificerare", del_sfar_sdf)
    P.d["vram_process_mb"] = BC.gpu_proc_used_mb()
    P.skriv()
