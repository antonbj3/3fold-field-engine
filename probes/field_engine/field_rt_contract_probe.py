"""Measure the frozen winding/EDT contract before designing an RT replacement."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline

ROOT = Path(__file__).resolve().parents[2]


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def measure():
    rows, arrays = [], {}
    first = trimesh.creation.box(extents=[4, 4, 4])
    overlap = first.copy()
    overlap.apply_translation([1, 0.5, 0.25])
    nested = trimesh.creation.box(extents=[2, 2, 2])
    shifted = first.copy()
    shifted.apply_translation([0.13, -0.21, 0.37])
    reversed_box = first.copy()
    reversed_box.faces = reversed_box.faces[:, ::-1]
    cases = {'box': first, 'offset': shifted, 'reversed': reversed_box,
             'overlap': trimesh.util.concatenate([first, overlap]),
             'nested': trimesh.util.concatenate([first, nested])}
    for name, mesh in cases.items():
        pitch, origin = 0.25, np.array([-4., -4., -4.])
        args = (mesh.vertices, mesh.faces, pitch, origin)
        gmin, shape, _, winding, distance = baseline.surface_raster_and_flood(*args)
        pgmin, pshape, _, parity, _ = baseline.surface_raster_and_flood(*args, metod='raypar_paritet')
        assert np.array_equal(gmin, pgmin) and np.array_equal(shape, pshape)
        local_origin = origin + gmin * pitch
        col, z, sign = baseline._kolumntraffar(mesh.vertices, mesh.faces, pitch, local_origin, shape[0], shape[1])
        row = dict(case=name, shape=list(map(int, shape)), triangles=len(mesh.faces),
                   ray_columns=int(shape[0]*shape[1]), hits=len(z),
                   occupied=int(winding.sum()), parity_occupied=int(parity.sum()),
                   parity_mismatches=int(np.count_nonzero(winding != parity)),
                   winding_sha256=digest(winding), parity_sha256=digest(parity), distance_sha256=digest(distance))
        if name in ('box', 'offset', 'reversed'):
            indices = np.indices(shape).reshape(3, -1).T
            points = local_origin + indices * pitch
            centre = np.asarray(mesh.bounds).mean(axis=0)
            q = np.abs(points-centre)-2
            analytic = np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(np.max(q, axis=1), 0)
            row['edt_vs_analytic_max_abs'] = float(np.max(np.abs(distance.ravel()-analytic)))
            # A fixed interior point: vertical ray distance versus nearest face distance.
            point = centre + np.array([1.75, 0., 0.])
            row['interior_vertical_hit_distance'] = float(centre[2]+2-point[2])
            row['interior_nearest_face_distance'] = 0.25
        arrays[name+'_winding'] = winding
        arrays[name+'_parity'] = parity
        arrays[name+'_distance'] = distance
        arrays[name+'_hits'] = np.column_stack([col, z, sign])
        row['hits_sha256'] = digest(arrays[name+'_hits'])
        rows.append(row)
    return rows, arrays


def main():
    first, a = measure()
    second, b = measure()
    gates = {'full_arrays_bit_identical': all(np.array_equal(a[k], b[k]) for k in a),
             'metric_tables_identical': first == second,
             'all_distances_finite': all(np.isfinite(v).all() for k, v in a.items() if k.endswith('_distance')),
             'global_orientation_invariant': np.array_equal(a['box_winding'], a['reversed_winding']) and np.array_equal(a['box_distance'], a['reversed_distance'])}
    report = {'rows': first, 'gates': gates, 'sample_offset': baseline.VOXEL_PROVPUNKT,
              'jitter_fraction': baseline.JITTER_FRAC, 'timing_measured': False,
              'scope': 'Frozen CPU contract measurement; no RT backend or performance certificate.'}
    folder = ROOT/'artifacts'
    folder.mkdir(exist_ok=True)
    (folder/'field_rt_contract_probe.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(folder/'field_rt_contract_probe_arrays.npz', **a)
    print(json.dumps(report), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
