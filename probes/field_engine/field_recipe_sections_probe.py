"""Measure topology-optimised field sections before designing a loft recipe.

CPU only. Uses the frozen optimiser and primitive fitter; no Warp initialization.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from scipy import ndimage
from skimage import measure
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lastfalt_v1_topopt as TO
import field_to_recipe_v1 as BASE


def field():
    previous = TO.OUT_DIR
    try:
        with tempfile.TemporaryDirectory() as directory:
            TO.OUT_DIR = directory
            with contextlib.redirect_stdout(sys.stderr):
                result = TO.run(n_iter=BASE.TOPOPT_ITER, tag="sections_probe")
            rho = np.load(Path(directory) / "rho_field_sections_probe.npy")
    finally:
        TO.OUT_DIR = previous
    pitch = float(min(result["dx_dy_dz_mm"]))
    zoom = np.asarray(result["dx_dy_dz_mm"]) / pitch
    f = ndimage.zoom(rho.astype(np.float64)-.5, zoom, order=1, mode="nearest")
    f = np.pad(f, 1, mode="constant", constant_values=-.5)
    return f, pitch, result


def main():
    f, pitch, result = field()
    v, t, _, _ = measure.marching_cubes(f.astype(np.float32), level=0, spacing=(pitch,)*3)
    box = TO.synthetic_plate_and_loads()[0]["bbox_mm"]
    origin = np.array([box["xmin"], box["ymin"], box["zmin"]]) - pitch
    v = v.astype(np.float64) + origin
    fit = BASE.passa_primitiver(v.astype(np.float64), t.astype(np.int64), BASE.PLAN_TOL_PITCH*pitch)
    import trimesh
    mesh = trimesh.Trimesh(v, t, process=False)
    sections = []
    for z in range(1, f.shape[2]-1):
        contours = measure.find_contours(f[:, :, z], 0)
        closed = [c for c in contours if np.array_equal(c[0], c[-1])]
        areas = [abs(float(np.sum(c[:-1, 0]*c[1:, 1]-c[1:, 0]*c[:-1, 1])))*pitch*pitch/2 for c in closed]
        sections.append({"z_index": z, "loops": len(contours), "closed_loops": len(closed),
                         "vertices": [len(c)-1 for c in closed], "absolute_loop_areas": areas})
    volume = abs(float(mesh.volume))
    out = {"status": "SYNTHETIC-ONLY", "rho_shape": result["grid"], "field_shape": list(f.shape),
           "pitch": pitch, "origin": origin.tolist(), "volume": volume, "area": float(mesh.area),
           "rasterisation_bound": BASE.rasteringsbund(pitch, float(mesh.area), volume),
           "primitive_coverage": fit["tackning_area_frac"], "freeform_area": fit["fri_form_area_mm2"],
           "sections": sections, "field_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
           "gates": {"closed_sections": all(r["loops"] == r["closed_loops"] for r in sections),
                     "positive_volume": volume > 0, "watertight": bool(mesh.is_watertight),
                     "partial_coverage": 0 < fit["tackning_area_frac"] < 1}}
    out["pass"] = all(out["gates"].values())
    dest = _probe_root / "src/field_engine/artifacts"
    dest.mkdir(exist_ok=True)
    np.save(dest / "field_recipe_sections_probe.npy", f)
    (dest / "field_recipe_sections_probe.json").write_text(json.dumps(out, indent=2, sort_keys=True)+"\n")
    print(json.dumps(out, sort_keys=True))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
