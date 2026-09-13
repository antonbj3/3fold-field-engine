#!/usr/bin/env python3
"""Self-test for faltkarna_v1: three planted-fault tests plus a substrate check.

  (i)   corrupted block indexing -> the conservation gate must fail
  (ii)  float32 accumulation without fixed point -> the determinism gate must fail
        (int64 fixed point must hold), measured over six separate processes
  (iii) CSG of two known primitives (box minus cylinder) -> volume against the analytic value

Test field: box [-hx,hx]x[-hy,hy]x[-hz,hz] minus an infinite Z cylinder of radius r through the
centre. Analytic volume = 8*hx*hy*hz - pi*r^2*(2*hz).

Takes no arguments; writes src/field_engine/artifacts/faltkarna_v1_selftest.json and prints it.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
import math
import os
import time

import sys

import numpy as np
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import faltkarna_v1 as FK

HERE = str(_probe_root / 'src/field_engine')
REPORT = os.path.join(HERE, "artifacts", "faltkarna_v1_selftest.json")

HX, HY, HZ = 20.0, 15.0, 10.0
R = 6.0
PITCH = 0.25
BLOCK = 8
VOL_ANALYTIC = 8.0 * HX * HY * HZ - math.pi * R * R * (2.0 * HZ)


def make_meta():
    pad = 1.0
    nx = int(round((2 * HX + 2 * pad) / PITCH)) + 1
    ny = int(round((2 * HY + 2 * pad) / PITCH)) + 1
    nz = int(round((2 * HZ + 2 * pad) / PITCH)) + 1
    return FK.GridMeta(x0=-HX - pad, y0=-HY - pad, z0=-HZ - pad, pitch=PITCH, nx=nx, ny=ny, nz=nz,
                        block=BLOCK)


def main():
    out = {"cell": "faltkarna_v1_selftest", "params": dict(hx=HX, hy=HY, hz=HZ, r=R, pitch=PITCH,
                                                             block=BLOCK, vol_analytic_mm3=VOL_ANALYTIC)}

    # ---- (iii) normal rasterisation + conservation ----------------------------------------------
    meta = make_meta()
    t0 = time.time()
    sf = FK.klassificera_och_evaluera_testfalt(meta, HX, HY, HZ, R, corrupt_index=False)
    t_raster_s = time.time() - t0
    dense = FK.rekonstruera_tat(sf)
    verts, faces, _n, _v = measure.marching_cubes(dense, level=0.0, spacing=(PITCH,) * 3)
    verts = verts + np.array([meta.x0, meta.y0, meta.z0])
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    # This body (box minus a through cylinder, genus 1, one connected surface) has no loose
    # fragments, so it is deliberately not component-split: on a not-perfectly-manifold mesh
    # trimesh.split(only_watertight=False) returns duplicated/degenerate "components". The raw
    # signed volume over the whole surface matches the analytic value and is used directly.
    vol_mesh = abs(mesh.volume)
    # the dense array already holds -BIG for interior and +BIG for exterior blocks, so counting
    # voxels < 0 covers both the near-surface tiles and the constant blocks.
    vol_voxel = float(np.sum(dense < 0.0)) * PITCH ** 3

    check_mesh = FK.konservering_check(vol_mesh, VOL_ANALYTIC, tol_frac=0.03)
    check_voxel = FK.konservering_check(vol_voxel, VOL_ANALYTIC, tol_frac=0.03)
    out["fallbevis_iii_normal"] = {
        "vol_mesh_mm3": vol_mesh, "vol_voxel_mm3": vol_voxel, "vol_analytic_mm3": VOL_ANALYTIC,
        "check_mesh": check_mesh, "check_voxel": check_voxel,
        "t_raster_s": t_raster_s, "n_verts": len(mesh.vertices), "n_faces": len(mesh.faces),
    }
    out["substrat_minne"] = dict(sf.stats,
                                  dense_MB=meta.nx * meta.ny * meta.nz * 4 / 1e6,
                                  gles_tile_MB=sf.stats["active_voxels"] * 4 / 1e6,
                                  gles_over_dense_frac=sf.stats["active_voxels"] / sf.stats["dense_voxels"])

    # ---- (i) corrupted block indexing -> the conservation gate must fail ------------------------
    sf_bad = FK.klassificera_och_evaluera_testfalt(meta, HX, HY, HZ, R, corrupt_index=True)
    dense_bad = FK.rekonstruera_tat(sf_bad)
    vol_voxel_bad = float(np.sum(dense_bad < 0.0)) * PITCH ** 3
    check_bad = FK.konservering_check(vol_voxel_bad, VOL_ANALYTIC, tol_frac=0.03)
    out["fallbevis_i_korrupt_index"] = {
        "vol_voxel_mm3": vol_voxel_bad, "vol_analytic_mm3": VOL_ANALYTIC, "check": check_bad,
        "GRIND_FALLER_SOM_FORVANTAT": bool(not check_bad["GRON"]),
    }

    # ---- (ii) float32 accumulation without fixed point -> the determinism gate must fail --------
    det = FK.volym_via_gpu_reduktion(sf, n_korningar=6)
    out["fallbevis_ii_determinism"] = det
    out["fallbevis_ii_determinism"]["GRIND_FALLER_F32_SOM_FORVANTAT"] = bool(
        not det["f32_unsafe"]["bit_identiska"])
    out["fallbevis_ii_determinism"]["GRIND_HALLER_I64_SOM_FORVANTAT"] = bool(
        det["i64_fixpunkt_safe"]["bit_identiska"])

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("\nwrote", REPORT)


if __name__ == "__main__":
    main()
