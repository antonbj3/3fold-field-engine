"""Complete mesh-stage/adaptive-field timings and exact queries for S/M/L."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import time
import weakref
import numpy as np
from scipy import ndimage
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline
from mesh_field_cuda_columns_v1 import surface_raster_and_flood_cuda
from mesh_field_native_mask_v1 import surface_raster_and_flood_rt
from mesh_adaptive_field_v1 import mesh_to_adaptive

ROOT = Path(__file__).resolve().parents[2]


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    mesh = trimesh.load(os.environ['FIELD_PLATE_STL'], process=False)
    mesh.merge_vertices()
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    common = dict(edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'], column_library=os.environ['COLUMN_MASK_LIBRARY'])
    stages = dict(cpu=(baseline.surface_raster_and_flood, {}),
                  cuda=(surface_raster_and_flood_cuda, dict(common, columns_library=os.environ['CUDA_COLUMNS_LIBRARY'])),
                  rt=(surface_raster_and_flood_rt, dict(common, winding_library=os.environ['RT_COLUMNS_LIBRARY'], ptx=os.environ['RT_COLUMNS_PTX'])))
    rows, arrays = [], {}
    order = [(name, pitch) for pitch in (2., 1., .5) for name in stages]
    for leg in range(2):
        for name, pitch in (order if leg == 0 else list(reversed(order))):
            origin = vertices.min(0) - 3*pitch
            reference = baseline.surface_raster_and_flood(vertices, faces, pitch, origin)
            indices = np.indices(reference[1]).reshape(3, -1).T
            anchor = origin + reference[0]*pitch
            rng = np.random.default_rng(417)
            points = anchor + rng.uniform(-1, np.array(reference[1]) + 1, (4097, 3))*pitch
            expected = ndimage.map_coordinates(reference[4], ((points-anchor)/pitch).T, order=1, mode='nearest')
            stage, kwargs = stages[name]
            measured = {}
            pointers = []
            def observe_stage(*args, **kw):
                start = time.perf_counter_ns()
                result = stage(*args, **kw)
                measured['stage_ms'] = (time.perf_counter_ns()-start)*1e-6
                measured['stage_array_bytes'] = sum(np.asarray(result[i]).nbytes for i in (0, 2, 3, 4))
                pointers.extend(weakref.ref(result[i]) for i in (2, 3, 4))
                return result
            start = time.perf_counter_ns()
            field = mesh_to_adaptive(observe_stage, vertices, faces, pitch, origin, **kwargs)
            total_ms = (time.perf_counter_ns()-start)*1e-6
            source_released = all(pointer() is None for pointer in pointers)
            start = time.perf_counter_ns()
            query = field.query(points)
            query_ms = (time.perf_counter_ns()-start)*1e-6
            grid = field.samples.samples_at(indices).reshape(field.shape)
            rows.append(dict(backend=name, pitch=pitch, leg=leg, total_ms=total_ms,
                             packing_ms=total_ms-measured['stage_ms'], query_4097_ms=query_ms, **measured,
                             retained_bytes=field.storage_bytes, dense_distance_bytes=reference[4].nbytes,
                             retained_ratio=field.storage_bytes/reference[4].nbytes,
                             source_released=source_released,
                             grid_exact=grid.tobytes() == reference[4].tobytes(),
                             query_exact=query.tobytes() == expected.tobytes(),
                             offset_exact=np.array_equal(field.gmin, reference[0]),
                             hashes=dict(grid=digest(grid), query=digest(query), nodes=digest(field.samples.nodes), payload=digest(field.samples.payload))))
            arrays[f'{name}_{pitch}_{leg}_queries'] = query
            print(json.dumps(rows[-1]), flush=True)
    gates = dict(complete=len(rows) == 18, grid_exact=all(r['grid_exact'] and r['offset_exact'] for r in rows),
                 query_exact=all(r['query_exact'] for r in rows),
                 source_released=all(r['source_released'] for r in rows),
                 repeat=all(len({json.dumps(r['hashes'], sort_keys=True) for r in rows if r['pitch']==pitch})==1 for pitch in (2.,1.,.5)),
                 retained_smaller=all(r['retained_ratio']<1 for r in rows))
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', rows=rows, gates=gates,
                  scope='Full stage plus lossless packing in one process, reversed orders. First backend setup included where it occurs; later calls warm. Retained host arrays, not peak RSS/device context.')
    (ROOT/'reports/mesh_adaptive.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/mesh_adaptive_arrays.npz', **arrays)
    print(json.dumps(dict(gates=gates)))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
