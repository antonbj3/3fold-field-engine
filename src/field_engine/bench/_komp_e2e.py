#!/usr/bin/env python3
"""Benchmark component: the end-to-end chain on one part.

B-rep -> STL -> mesh-to-SDF -> sparse block field -> dense reconstruction -> marching cubes (mesh
out, volume) -> occupancy at a coarser pitch. Part: the holed plate from
`examples/parts/holed_plate_v1.py`, whose volume and per-hole volume are known in closed form, so
every stage has a reference and not only a time. Each stage is timed, and the stage with the largest
share is reported as the bottleneck. Writes bench/artifacts/_parts/e2e.json.
"""
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
PITCH_SDF = 2.0 if SNABB else 1.0
PITCH_OCC = 4.0 if SNABB else 2.0


def _closed_form_volume_mm3():
    length, width = PLATE.plate_size_mm()
    v = length * width * PLATE.THICKNESS_MM
    for r in PLATE.HOLE_RADII_MM:
        v -= PLATE.brep_hole_volume_mm3(r)
    return float(v)


def del_kedja():
    import warp as wp
    wp.init()
    tl = []

    def mark(namn, t0, **kw):
        tl.append(dict(steg=namn, t_s=time.perf_counter() - t0, **kw))
        return time.perf_counter()

    t = time.perf_counter()
    V, T = PLATE.mesh(regenerate=not os.path.exists(PLATE.STL_PATH))
    t = mark("0_brep_till_mesh", t, n_tris=int(len(T)), n_verts=int(len(V)))

    lo = V.min(0) - 3 * PITCH_SDF
    res = M2S.mesh_to_sdf_del(wp, V, T, PITCH_SDF, 0.0, lo, DEV)
    t = mark("1_mesh_till_sdf_gles", t, pitch_mm=PITCH_SDF, grid=list(map(int, res["shape_l"])),
             n_cells=int(np.prod(res["shape_l"])), n_blocks=res["n_blocks"], n_active=res["n_active"],
             andel_aktiva=res["n_active"] / res["n_blocks"], t_cpu_flood_edt_s=res["t_flood_s"],
             t_klassificera_s=res["t_gpu_classify_s"])

    gmin, shape_l, yta, solid, sd = M2S.surface_raster_and_flood(V, T, PITCH_SDF, lo, device=DEV)
    t = mark("2_tat_falt_for_export", t, n_cells=int(sd.size), MB=sd.nbytes / 1e6)

    from skimage import measure
    import trimesh
    vv, ff, _, _ = measure.marching_cubes(sd, level=0.0, spacing=(PITCH_SDF,) * 3)
    m = trimesh.Trimesh(vv, ff, process=False)
    vol_mc = float(abs(m.volume))
    vol_vox = float(solid.sum()) * PITCH_SDF ** 3
    vol_cf = _closed_form_volume_mm3()
    t = mark("3_marching_cubes_mesh_ut", t, n_faces_ut=int(len(ff)), n_verts_ut=int(len(vv)),
             volym_mc_mm3=vol_mc, volym_voxel_mm3=vol_vox, volym_sluten_form_mm3=vol_cf,
             rel_mc_vs_sluten_form=abs(vol_mc - vol_cf) / vol_cf,
             rel_voxel_vs_sluten_form=abs(vol_vox - vol_cf) / vol_cf)

    gmin2, shape2, yta2, solid2, sd2 = M2S.surface_raster_and_flood(V, T, PITCH_OCC, V.min(0) - 2 * PITCH_OCC, device=DEV)
    nx, ny, nz = solid2.shape
    t = mark("4_occupancy_grov_pitch", t, grid=[int(nx), int(ny), int(nz)], n_cells=int(solid2.size),
             pitch_mm=PITCH_OCC, fyllnadsgrad=float(solid2.mean()),
             volym_mm3=float(solid2.sum()) * PITCH_OCC ** 3)

    tot = sum(s["t_s"] for s in tl)
    for s in tl:
        s["andel"] = s["t_s"] / tot
    return dict(objekt="holed_plate", pitch_sdf_mm=PITCH_SDF, pitch_occ_mm=PITCH_OCC, tidslinje=tl,
                total_s=tot, flaskhals=max(tl, key=lambda s: s["t_s"])["steg"],
                volym_sluten_form_mm3=_closed_form_volume_mm3(), rss_peak_mb=BC.rss_peak_mb(), device=DEV)


if __name__ == "__main__":
    P = BC.Part("e2e")
    P.kor("kedja_brep_mesh_sdf_falt_mesh", del_kedja)
    P.d["vram_process_mb"] = BC.gpu_proc_used_mb()
    P.skriv()
