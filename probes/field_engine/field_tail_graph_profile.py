"""GPU event attribution with full-field CPU and repeat controls, no speed gate."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from mesh_fused_into_v1 import PreparedMeshField
from mesh_field_shared_service_v1 import layout
from field_mesh_native_mask_stage_probe import cases, expected_open, baseline
from field_mesh_fused_probe import same

ROOT = Path(__file__).resolve().parents[2]


def main():
    libraries = dict(fused_library=os.environ['MESH_TAIL_GRAPH_PROFILE_LIBRARY'],
                     winding_library=os.environ['RT_COLUMNS_LIBRARY'], ptx=os.environ['RT_COLUMNS_PTX'],
                     edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'], column_library=os.environ['COLUMN_MASK_LIBRARY'])
    rows, captures = [], {}
    fixtures = cases()
    for name, (mesh, pitch, origin, closed) in fixtures.items():
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        expected = (baseline.surface_raster_and_flood if closed else expected_open)(vertices, faces, pitch, origin)
        with PreparedMeshField(vertices, faces, pitch, origin, **libraries) as field:
            n, offset, total = layout(field.shape)
            region = memoryview(bytearray(total))
            solid = np.ndarray(field.shape, bool, buffer=region)
            surface = np.ndarray(field.shape, bool, buffer=region, offset=n)
            distance = np.ndarray(field.shape, np.float32, buffer=region, offset=offset)
            field.pin_outputs(region)
            field.evaluate_into(solid, surface, distance)
            for leg in range(2):
                start = time.perf_counter_ns()
                field.evaluate_into(solid, surface, distance)
                elapsed = (time.perf_counter_ns() - start) * 1e-6
                phases = None
                if closed:
                    native = field._rt._lib.fused_profile
                    native.argtypes = [ct.c_void_p, ct.c_void_p]; native.restype = ct.c_int
                    output = np.empty(4, np.float32)
                    status = native(field._rt._handle, output.ctypes.data)
                    if status:
                        raise RuntimeError(f'GPU profile status {status}')
                    phases = dict(zip(('trace', 'mask_surface', 'paired_edt', 'readback'), map(float, output)))
                actual = (field._gmin.copy(), field.shape, surface.copy(), solid.copy(), distance.copy())
                rows.append(dict(case=name, leg=leg, wall_ms=elapsed, gpu_ms=phases, exact=same(actual, expected),
                                 hashes=[hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for a in actual]))
                for i, a in enumerate(actual):
                    captures[f'{name}_{leg}_{i}'] = np.asarray(a)
            field.unpin_outputs()
    gates = dict(coverage=len(rows) == 14, exact=all(r['exact'] for r in rows),
                 repeats=all(len({json.dumps(r['hashes']) for r in rows if r['case'] == name}) == 1 for name in fixtures),
                 accounting=sum(r['gpu_ms'] is not None for r in rows) == 10 and all(
                     all(np.isfinite(t) and t >= 0 for t in r['gpu_ms'].values()) and sum(r['gpu_ms'].values()) <= r['wall_ms']
                     for r in rows if r['gpu_ms'] is not None))
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates, rows=rows,
                  source_sha256=hashlib.sha256((ROOT/'probes/field_engine/mesh_tail_graph_profile_v1/field.cu').read_bytes()).hexdigest(),
                  scope='Observer-only CUDA events; no extra host fences, event retrieval after complete evaluation. GPU intervals are nested within wall time and include queue gaps. Prepared pinned outputs; no service IPC or owned snapshot copy in timing. Open fallback has no GPU phase attribution. No cross-job causal speed or latency acceptance claim.')
    (ROOT/'reports/field_tail_graph_profile.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT/'reports/field_tail_graph_profile_arrays.npz', **captures)
    print(json.dumps(report), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
