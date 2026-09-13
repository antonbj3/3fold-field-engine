"""Two fresh first-use CUDA scan stages, with all setup and teardown charged."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def child(leg):
    import trimesh
    import faltkarna_v1_mesh_to_sdf as baseline
    from mesh_field_cuda_columns_v1 import PreparedMeshField
    mesh = trimesh.load(os.environ['FIELD_PLATE_STL'], process=False)
    mesh.merge_vertices()
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    pitch = .5
    origin = vertices.min(0) - 3 * pitch
    begin = time.perf_counter_ns()
    expected = baseline.surface_raster_and_flood(vertices, faces, pitch, origin)
    cpu_ms = (time.perf_counter_ns() - begin) * 1e-6
    begin = time.perf_counter_ns()
    field = PreparedMeshField(vertices, faces, pitch, origin,
                              columns_library=os.environ['CUDA_COLUMNS_LIBRARY'],
                              edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'],
                              column_library=os.environ['COLUMN_MASK_LIBRARY'])
    built = time.perf_counter_ns()
    try:
        actual = field.evaluate()
        evaluated = time.perf_counter_ns()
    finally:
        field.close()
    closed = time.perf_counter_ns()
    arrays, hashes, differences = {}, {}, {}
    for i, name in ((0, 'gmin'), (2, 'surface'), (3, 'solid'), (4, 'distance')):
        arrays[name] = actual[i]
        hashes[name] = hashlib.sha256(actual[i].tobytes()).hexdigest()
        differences[name] = int(np.count_nonzero(actual[i] != expected[i]))
    row = dict(cpu_ms=cpu_ms, constructor_ms=(built-begin)*1e-6,
               evaluate_ms=(evaluated-built)*1e-6, close_ms=(closed-evaluated)*1e-6,
               total_ms=(closed-begin)*1e-6, differences=differences, hashes=hashes,
               shape_equal=actual[1] == expected[1],
               no_optix_loaded='libnvoptix' not in Path('/proc/self/maps').read_text())
    dest = ROOT/'reports'
    (dest/f'cuda_columns_cold_{leg}.json').write_text(json.dumps(row, indent=2)+'\n')
    np.savez_compressed(dest/f'cuda_columns_cold_{leg}_arrays.npz', **arrays)


def main():
    if '--child' in sys.argv:
        child(int(sys.argv[2]))
        return 0
    rows = []
    for leg in range(2):
        begin = time.perf_counter_ns()
        run = subprocess.run([sys.executable, __file__, '--child', str(leg)],
                             capture_output=True, text=True, timeout=120)
        if run.returncode:
            print(run.stderr, file=sys.stderr)
            return run.returncode
        wall = (time.perf_counter_ns() - begin)*1e-6
        row = json.loads((ROOT/f'reports/cuda_columns_cold_{leg}.json').read_text())
        row['whole_process_ms'] = wall
        rows.append(row)
    gates = dict(exact=all(r['shape_equal'] and not any(r['differences'].values()) for r in rows),
                 repeat=rows[0]['hashes'] == rows[1]['hashes'],
                 no_optix=all(r['no_optix_loaded'] for r in rows),
                 cold_faster=all(r['total_ms'] < r['cpu_ms'] for r in rows))
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
                  rows=rows, gates=gates,
                  scope='First-use L CUDA scan, CPU reference and complete setup/evaluate/close. No OptiX or RT claim.')
    (ROOT/'reports/cuda_columns_cold.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
