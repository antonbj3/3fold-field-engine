"""Full persistent-EDT mesh service with the original unchanged 2 ms L gate."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline
from mesh_field_service_persistent_v1 import MeshFieldService

ROOT = Path(__file__).resolve().parents[2]


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns()-start)*1e-6


def main():
    mesh = trimesh.load(os.environ['FIELD_PLATE_STL'], process=False)
    mesh.merge_vertices()
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    libraries = dict(pair_library=os.environ['NATIVE_EDT_PAIR_LIBRARY'], winding_library=os.environ['RT_COLUMNS_LIBRARY'], ptx=os.environ['RT_COLUMNS_PTX'],
                     edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'], column_library=os.environ['COLUMN_MASK_LIBRARY'])
    rows, captures, controls = [], {}, []
    for leg in range(2):
        for pitch in (2., 1., .5):
            origin = vertices.min(0)-3*pitch
            expected = baseline.surface_raster_and_flood(vertices, faces, pitch, origin)
            begin = time.perf_counter_ns()
            service = MeshFieldService(**libraries)
            try:
                service.prepare(vertices, faces, pitch, origin)
                (first, worker), query_ms = timed(service.query)
                first_ms = (time.perf_counter_ns()-begin)*1e-6
                outputs, warm, worker_times = [first], [], [worker]
                for _ in range(4):
                    (result, worker), elapsed = timed(service.query)
                    outputs.append(result)
                    warm.append(elapsed)
                    worker_times.append(worker)
                hashes, exact = [], True
                for repeat, result in enumerate(outputs):
                    items = [np.asarray(x) for x in result]
                    exact &= all(a.dtype == np.asarray(b).dtype and a.shape == np.asarray(b).shape and a.tobytes() == np.asarray(b).tobytes() for a,b in zip(items, expected))
                    hashes.append([hashlib.sha256(a.tobytes()).hexdigest() for a in items])
                    for index, a in enumerate(items):
                        captures[f'{pitch}_{leg}_{repeat}_{index}'] = a
                rows.append(dict(pitch=pitch, leg=leg, first_query_ms=first_ms, first_ipc_ms=query_ms,
                                 warm_query_ms=warm, worker_ms=worker_times, exact=bool(exact), hashes=hashes))
                # Replacement changes geometry while keeping the process alive.
                moved = vertices + np.array([pitch, 0., 0.])
                service.prepare(moved, faces, pitch, origin)
                replacement, _ = service.query()
                reference = baseline.surface_raster_and_flood(moved, faces, pitch, origin)
                controls.append(all(np.asarray(a).tobytes() == np.asarray(b).tobytes() for a,b in zip(replacement, reference)))
                # Failed replacement must not return stale geometry.
                try:
                    service.prepare(vertices, faces, -1, origin)
                except RuntimeError:
                    try:
                        service.query()
                    except RuntimeError:
                        controls.append(True)
                    else:
                        controls.append(False)
                else:
                    controls.append(False)
            finally:
                service.close()
            controls.append(not service._process.is_alive())
            try:
                service.query()
            except RuntimeError:
                controls.append(True)
            else:
                controls.append(False)
    gates = dict(coverage=len(rows)==6, exact=all(r['exact'] for r in rows),
                 repeats=all(len({json.dumps(h) for r in rows if r['pitch']==p for h in r['hashes']})==1 for p in (2.,1.,.5)),
                 lifecycle=len(controls)==24 and all(controls),
                 warm_L_2ms=all(t<=2. for r in rows if r['pitch']==.5 for t in r['warm_query_ms']))
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates, rows=rows,
                  scope='Six spawned workers, two fresh workers per size. First includes spawn/import/prepare/full output IPC; warm includes full output IPC, recomputation every query, no output cache. L means plate pitch 0.5.')
    (ROOT/'reports/field_service_persistent.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_service_persistent_arrays.npz', **captures)
    print(json.dumps(report), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
