"""Graph-replayed full fields, exact CPU references and unchanged 2 ms service gate."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
from field_paths import field_path as _field_path
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import trimesh
from mesh_field_shared_service_v1 import MeshFieldService
from field_mesh_native_mask_stage_probe import cases, expected_open, baseline
from field_mesh_fused_probe import same

ROOT = Path(__file__).resolve().parents[2]


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) * 1e-6


def main():
    libraries = dict(fused_library=os.environ['MESH_FUSED_GRAPH_LIBRARY'],
                     winding_library=os.environ['RT_COLUMNS_LIBRARY'], ptx=os.environ['RT_COLUMNS_PTX'],
                     edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'], column_library=os.environ['COLUMN_MASK_LIBRARY'])
    rows, captures, controls = [], {}, []
    fixtures = cases()
    for leg in range(2):
        for name in (list(fixtures) if leg == 0 else list(fixtures)[::-1]):
            mesh, pitch, origin, closed = fixtures[name]
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            expected = (baseline.surface_raster_and_flood if closed else expected_open)(vertices, faces, pitch, origin)
            begin = time.perf_counter_ns()
            with MeshFieldService(**libraries) as service:
                service.prepare(vertices, faces, pitch, origin)
                setup_ms = (time.perf_counter_ns() - begin) * 1e-6
                timings, workers, copies, hashes, exact = [], [], [], [], True
                for repeat in range(5):
                    lease, elapsed = timed(service.query)
                    with lease:
                        snapshot, copy_ms = timed(lambda: tuple(np.array(x, copy=True) for x in lease.result))
                        exact &= same(snapshot, expected)
                        timings.append(elapsed); workers.append(lease.worker_ms); copies.append(copy_ms)
                        hashes.append([hashlib.sha256(a.tobytes()).hexdigest() for a in snapshot])
                        for index, a in enumerate(snapshot):
                            captures[f'{name}_{leg}_{repeat}_{index}'] = a
                rows.append(dict(case=name, leg=leg, setup_ms=setup_ms, pinned=service.metadata['pinned'],
                                 query_ms=timings, worker_ms=workers, owned_copy_ms=copies,
                                 exact=bool(exact), hashes=hashes))
                controls.append(service.metadata['pinned'] == closed)
                moved = vertices + np.array([pitch, 0., 0.])
                service.prepare(moved, faces, pitch, origin)
                with service.query() as replacement:
                    ref = (baseline.surface_raster_and_flood if closed else expected_open)(moved, faces, pitch, origin)
                    controls.append(same(replacement.result, ref))
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
            controls.append(not service._process.is_alive())
    for count in (32, 33):
        mesh = trimesh.util.concatenate([trimesh.creation.box(extents=[4, 4, 4]).apply_translation([0, 0, 10*i]) for i in range(count)])
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        origin = vertices.min(0) - 3
        with MeshFieldService(**libraries) as service:
            service.prepare(vertices, faces, 1., origin)
            if count == 32:
                with service.query() as lease:
                    controls.append(same(lease.result, baseline.surface_raster_and_flood(vertices, faces, 1., origin)))
                    for i, a in enumerate(lease.result):
                        captures[f'capacity_{i}'] = np.array(a, copy=True)
            else:
                for _ in range(2):
                    try:
                        service.query()
                    except RuntimeError:
                        controls.append(True)
                    else:
                        controls.append(False)
    gates = dict(coverage=len(rows) == 14,
                 exact=all(r['exact'] for r in rows),
                 repeats=all(len({json.dumps(h) for r in rows if r['case'] == name for h in r['hashes']}) == 1 for name in fixtures),
                 lifecycle_capacity=len(controls) == 59 and all(controls),
                 warm_L_2ms=all(t <= 2. for r in rows if r['case'] == 'plate_0.5' for t in r['query_ms'][1:]))
    sources = ('mesh_field_shared_service_v1.py', 'mesh_fused_into_v1.py', 'mesh_fused_shared_v1/field.cu', 'mesh_fused_graph_v1/field.cu',
               'mesh_fused_native_v1/field.cu', 'field_graph_service_probe.py')
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates,
                  rows=rows, controls=controls,
                  source_sha256={p: hashlib.sha256((_field_path(ROOT, p)).read_bytes()).hexdigest() for p in sources},
                  scope='Fourteen fresh spawned workers in two fixture orders. Setup includes spawn/import/prepare/graph capture/attach/pin. Every query recomputes full fields into shared memory and returns borrowed read-only views. First query separate; last four are warm. Explicit copy cost is excluded from borrowed latency and reported separately; snapshots retained for every query. No owned-output service acceptance claim.')
    (ROOT/'reports/field_graph_service.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT/'reports/field_graph_service_arrays.npz', **captures)
    print(json.dumps(report), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
