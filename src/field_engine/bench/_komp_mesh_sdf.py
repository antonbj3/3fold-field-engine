#!/usr/bin/env python3
"""Benchmark component: the mesh-to-SDF op (`faltkarna_v1_mesh_to_sdf`).

Parts: the holed plate from `examples/parts/holed_plate_v1.py` (five through holes of radius
1 to 16 mm in one plate) for the size sweep, the determinism check, the watertightness gate and the
open-mesh probe; closed analytic bodies with an exactly known volume (axis-aligned and rotated box,
axis-aligned and rotated cylinder, sphere) for the solid-filling accuracy, so that an axis-aligned
coincidence cannot pass for correctness.

Measured: (a) triangles/s and cells/s per declared pitch, (b) p50/p95 latency, (c) peak RSS,
(d) process scaling 1/2/4 on the CPU stage (surface raster + flood + EDT), which dominates the cost,
(e) voxel volume against the closed form for three signing methods, (f) two runs bit-identical,
(g) the watertightness gate and the fail-open probe on a mesh with a hole cut in it.
Writes bench/artifacts/_parts/mesh_sdf.json.
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bench_common as BC  # noqa: E402

BC.repo_paths()
import faltkarna_v1_mesh_to_sdf as M2S  # noqa: E402
import holed_plate_v1 as PLATE  # noqa: E402

SNABB = BC.snabb()
DEV = BC.device()
HAVE_CUDA = BC.have_cuda()
# declared pitches for the plate; L (0.5 mm) is a 26 M cell grid and is run only on a CUDA device
SIZES = [("S", 2.0), ("M", 1.0)] + ([("L", 0.5)] if HAVE_CUDA else [])
if SNABB:
    SIZES = [("S", 4.0)]
METODER_JMF = ("skal_floodfill", "raypar_vindning", "winding_gpu")
TOL_ANALYTISK = {1.0: 0.02, 2.0: 0.05, 4.0: 0.10}   # declared: <=2 % at pitch 1 mm, <=5 % at 2 mm


def _plate():
    return PLATE.mesh(regenerate=not os.path.exists(PLATE.STL_PATH))


def _cpu_steg(args):
    """The pure CPU stage (surface raster + flood + EDT) for one body, run in a worker process."""
    namn, pitch = args
    V, T = _kropp(namn)
    lo = V.min(0) - 3 * pitch
    t0 = time.perf_counter()
    gmin, shape_l, yta, solid, sd = M2S.surface_raster_and_flood(V, T, pitch, lo)
    return dict(namn=namn, t_s=time.perf_counter() - t0, n_cells=int(np.prod(shape_l)), n_tris=int(len(T)))


def _kropp(namn):
    """Vertices and triangles of one declared body, by name."""
    import trimesh
    if namn == "holed_plate":
        return _plate()
    if namn == "lada":
        m = trimesh.creation.box(extents=(60.0, 40.0, 30.0))
    elif namn == "lada_roterad30":
        m = trimesh.creation.box(extents=(60.0, 40.0, 30.0))
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30.0), [1, 1, 0]))
    elif namn == "cylinder_r15h40":
        m = trimesh.creation.cylinder(radius=15.0, height=40.0, sections=256)
    elif namn == "cylinder_roterad37":
        m = trimesh.creation.cylinder(radius=15.0, height=40.0, sections=256)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(37.0), [0, 1, 0]))
    elif namn == "sfar_r20":
        m = trimesh.creation.icosphere(subdivisions=5, radius=20.0)
    else:
        raise KeyError(namn)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def _analytiska_kroppar():
    """Closed bodies with an exactly known volume. The rotated variants separate an axis-aligned
    coincidence (the voxel lattice lines up with the surface) from real correctness."""
    ut = []
    for pitch in ((4.0,) if SNABB else (2.0, 1.0)):
        ut.append(("lada", pitch, 60.0 * 40.0 * 30.0))
        ut.append(("lada_roterad30", pitch, 60.0 * 40.0 * 30.0))
        ut.append(("cylinder_r15h40", pitch, np.pi * 15.0 ** 2 * 40.0))
        ut.append(("cylinder_roterad37", pitch, np.pi * 15.0 ** 2 * 40.0))
        ut.append(("sfar_r20", pitch, 4.0 / 3.0 * np.pi * 20.0 ** 3))
    return ut


def del_storlekar():
    import warp as wp
    wp.init()
    V, T = _plate()
    rows = []
    for lbl, pitch in SIZES:
        lo = V.min(0) - 3 * pitch
        free0 = wp.get_device(DEV).free_memory if HAVE_CUDA else None
        reps, res = [], None
        for _ in range(2 if lbl == "L" else 3):
            t0 = time.perf_counter()
            res = M2S.mesh_to_sdf_del(wp, V, T, pitch, 0.0, lo, DEV)
            reps.append(time.perf_counter() - t0)
        free1 = wp.get_device(DEV).free_memory if HAVE_CUDA else None
        n_cells = int(np.prod(res["shape_l"]))
        vol_vox = float(res["solid_final"].sum()) * pitch ** 3
        try:
            import trimesh
            m = trimesh.Trimesh(V, T, process=False)
            m.merge_vertices()
            vol_mesh, watertight = float(abs(m.volume)), bool(m.is_watertight)
        except Exception:  # noqa: BLE001
            vol_mesh, watertight = None, None
        rows.append(dict(label=lbl, del_namn="holed_plate", pitch_mm=pitch, signeringsmetod=M2S.METOD_STANDARD,
                         n_tris=int(len(T)), n_verts=int(len(V)), grid_shape=list(map(int, res["shape_l"])),
                         n_cells=n_cells, p50_s=BC.pct(reps, 50), p95_s=BC.pct(reps, 95), all_s=reps,
                         t_cpu_flood_edt_s=res["t_flood_s"], t_gpu_klassificera_s=res["t_gpu_classify_s"],
                         andel_cpu=res["t_flood_s"] / (res["t_flood_s"] + res["t_gpu_classify_s"]),
                         trianglar_per_s=len(T) / BC.pct(reps, 50), celler_per_s=n_cells / BC.pct(reps, 50),
                         celler_per_s_klassificering=n_cells / max(res["t_gpu_classify_s"], 1e-9),
                         n_blocks=res["n_blocks"], n_active=res["n_active"],
                         andel_aktiva=res["n_active"] / res["n_blocks"],
                         n_diff_gles_vs_cpu_solid=res["n_diff_pre_margin_gpu_vs_cpu"],
                         GRON_bitidentisk=res["n_diff_pre_margin_gpu_vs_cpu"] == 0,
                         volym_voxel_mm3=vol_vox, volym_trimesh_mm3=vol_mesh, mesh_vattentat=watertight,
                         rel_avvikelse_voxel_vs_trimesh=(abs(vol_vox - vol_mesh) / vol_mesh) if vol_mesh else None,
                         vram_delta_MB=((free0 - free1) / 1e6) if (free0 is not None and free1 is not None) else None,
                         rss_peak_mb_hittills=BC.rss_peak_mb()))
    return dict(rader=rows, device=DEV)


def del_determinism():
    import warp as wp
    pitch = SIZES[0][1]
    V, T = _plate()
    lo = V.min(0) - 3 * pitch
    r1 = M2S.mesh_to_sdf_del(wp, V, T, pitch, 0.0, lo, DEV)
    r2 = M2S.mesh_to_sdf_del(wp, V, T, pitch, 0.0, lo, DEV)
    s1, s2 = BC.sha_arr(r1["solid_final"]), BC.sha_arr(r2["solid_final"])
    return dict(del_namn="holed_plate", pitch_mm=pitch, sha_run1=s1, sha_run2=s2, bit_identisk=s1 == s2)


def del_tradskalning():
    """Process scaling of the CPU stage over six bodies; the device stage is not included."""
    pitch = 4.0 if SNABB else 2.0
    jobb = [(n, pitch) for n in ("holed_plate", "lada", "lada_roterad30", "cylinder_r15h40",
                                 "cylinder_roterad37", "sfar_r20")]
    rows = []
    ctx = mp.get_context("fork")
    for n_proc in (1, 2, 4):
        t0 = time.perf_counter()
        with ctx.Pool(n_proc) as pool:
            res = pool.map(_cpu_steg, jobb)
        wall = time.perf_counter() - t0
        rows.append(dict(n_proc=n_proc, wall_s=wall, sum_cpu_s=sum(r["t_s"] for r in res),
                         n_cells_tot=sum(r["n_cells"] for r in res), n_tris_tot=sum(r["n_tris"] for r in res),
                         celler_per_s=sum(r["n_cells"] for r in res) / wall))
    base = rows[0]["wall_s"]
    for r in rows:
        r["speedup_vs_1"] = base / r["wall_s"]
        r["effektivitet"] = r["speedup_vs_1"] / r["n_proc"]
    return dict(pitch_mm=pitch, n_delar=len(jobb), rader=rows,
                not_="CPU stage (surface raster + flood + EDT) in multiprocessing; the device stage is excluded")


def del_solidfyllning_analytiska_kroppar():
    """Voxel volume against the closed form for three signing methods in one run:
      skal_floodfill  point raster plus a label flood fill. It returns a hollow shell; kept as the
                      before column, so the size of the defect it had is on the record.
      raypar_vindning the standard path: a +Z ray winding number per voxel centre.
      winding_gpu     Warp BVH plus the generalised winding number (exact sd, no EDT).
    Declared tolerance: rel. error <= 2 % at pitch 1 mm, <= 5 % at 2 mm."""
    import warp as wp
    wp.init()
    rader = []
    for namn, pitch, vol_an in _analytiska_kroppar():
        V, T = _kropp(namn)
        lo = V.min(0) - 3 * pitch
        rad = dict(fall=namn, pitch_mm=pitch, n_tri=int(len(T)), volym_analytisk_mm3=vol_an,
                   tol=TOL_ANALYTISK[pitch])
        for metod in METODER_JMF:
            d = {}
            t0 = time.perf_counter()
            gmin, shape_l, yta, solid, sd = M2S.surface_raster_and_flood(V, T, pitch, lo, metod=metod, wp=wp, device=DEV, diag_ut=d)
            dt = time.perf_counter() - t0
            vv = float(np.count_nonzero(solid)) * pitch ** 3
            rel = abs(vv - vol_an) / vol_an
            rad[metod] = dict(volym_voxel_mm3=vv, rel_fel=rel, t_s=dt, n_solid=int(np.count_nonzero(solid)),
                              n_ytlager=int(np.count_nonzero(yta)),
                              solid_ar_bara_skalet=bool(np.count_nonzero(solid) == np.count_nonzero(yta)),
                              n_vindning_vs_paritet_diff=d.get("n_vindning_vs_paritet_diff"),
                              GRON=bool(rel <= TOL_ANALYTISK[pitch]))
        rader.append(rad)
    sam = {m: dict(varsta_rel_fel=max(r[m]["rel_fel"] for r in rader),
                   n_roda=sum(0 if r[m]["GRON"] else 1 for r in rader), n_fall=len(rader),
                   t_tot_s=sum(r[m]["t_s"] for r in rader),
                   GRON_alla=all(r[m]["GRON"] for r in rader)) for m in METODER_JMF}
    std = M2S.METOD_STANDARD
    return dict(rader=rader, sammanfattning=sam, metod_standard=std,
                varsta_rel_fel=sam[std]["varsta_rel_fel"], GRON_alla=sam[std]["GRON_alla"],
                not_="the skal_floodfill column IS the old algorithm and is expected red: it is the "
                     "before picture, not a regression")


def del_vattentathetsgrind():
    """The op's watertightness gate on the plate: volume before (mesh as generated) and after a
    declared repair (fill_holes before rasterising), plus the gate's verdict."""
    import warp as wp
    wp.init()
    V, T = _plate()
    rows = []
    for lbl, pitch in SIZES:
        lo = V.min(0) - 3 * pitch
        r0 = M2S.mesh_to_sdf_del(wp, V, T, pitch, 0.0, lo, DEV)
        r1 = M2S.mesh_to_sdf_del(wp, V, T, pitch, 0.0, lo, DEV, laga_oppen=True)
        v0 = float(np.count_nonzero(r0["solid_final"])) * pitch ** 3
        v1 = float(np.count_nonzero(r1["solid_final"])) * pitch ** 3
        g0, g1 = r0["vattentathet"], r1["vattentathet"]
        rows.append(dict(label=lbl, del_namn="holed_plate", pitch_mm=pitch,
                         status_fore=g0["status"], status_efter=g1["status"],
                         vattentat_fore=g0.get("vattentat"), vattentat_efter=g1.get("vattentat"),
                         kanter_ej_par_fore=g0.get("n_kanter_ej_delade_av_2_trianglar"),
                         kanter_ej_par_efter=g1.get("n_kanter_ej_delade_av_2_trianglar"),
                         volym_voxel_fore_mm3=v0, volym_voxel_efter_mm3=v1,
                         volym_rel_andring=(v1 - v0) / v0 if v0 else None,
                         volym_trimesh_fore_mm3=g0.get("volym_trimesh_mm3"),
                         rel_avvikelse_efter=g1.get("rel_avvikelse_voxel_vs_trimesh"),
                         paritet_divergens_fore=g0.get("paritet_divergens_frac"),
                         paritet_divergens_efter=g1.get("paritet_divergens_frac"),
                         grind_wall_s=g0.get("grind_wall_s")))
    return dict(rader=rows, n_flaggade_fore=sum(r["status_fore"] != "OK" for r in rows),
                n_flaggade_efter=sum(r["status_efter"] != "OK" for r in rows),
                not_="fail-closed: every mesh_to_sdf_del result carries the field 'vattentathet'; "
                     "FALTKARNA_VATTENTATHET_STRIKT=1 raises instead of flagging")


def del_fail_open_probe():
    """The fail-open layer: a central disc of triangles is removed from the plate, so the flood fill
    leaks. Measured: how much volume leaks, and whether the op flags the mesh itself."""
    import warp as wp
    pitch = SIZES[0][1]
    V, T = _plate()
    lo = V.min(0) - 3 * pitch
    r_full = M2S.mesh_to_sdf_del(wp, V, T, pitch, 0.0, lo, DEV)
    c = V[T].mean(1)
    keep = np.linalg.norm(c - c.mean(0), axis=1) > 0.35 * np.ptp(V, axis=0).max()
    T2 = T[keep]
    r_hal = M2S.mesh_to_sdf_del(wp, V, T2, pitch, 0.0, lo, DEV)
    v0, v1 = float(r_full["solid_final"].sum()), float(r_hal["solid_final"].sum())
    g_full, g_hal = r_full["vattentathet"], r_hal["vattentathet"]
    return dict(del_namn="holed_plate", pitch_mm=pitch, n_tris_borttagna=int((~keep).sum()),
                volym_frac_efter_hal=v1 / v0, flaggar_openen_sjalv=bool(g_hal and g_hal["status"] != "OK"),
                grind_status_hel_mesh=g_full["status"], grind_status_hal_mesh=g_hal["status"],
                grind_flaggor_hal_mesh=g_hal.get("flaggor"),
                kanter_ej_par_hel=g_full.get("n_kanter_ej_delade_av_2_trianglar"),
                kanter_ej_par_hal=g_hal.get("n_kanter_ej_delade_av_2_trianglar"),
                paritet_divergens_hal=g_hal.get("paritet_divergens_frac"))


if __name__ == "__main__":
    P = BC.Part("mesh_sdf")
    P.kor("storlekar_SML", del_storlekar)
    P.kor("determinism", del_determinism)
    P.kor("tradskalning_cpu_1_2_4", del_tradskalning)
    P.kor("solidfyllning_analytiska_kroppar", del_solidfyllning_analytiska_kroppar)
    P.kor("vattentathetsgrind", del_vattentathetsgrind)
    P.kor("fail_open_probe_oppen_mesh", del_fail_open_probe)
    P.d["vram_process_mb"] = BC.gpu_proc_used_mb()
    P.skriv()
