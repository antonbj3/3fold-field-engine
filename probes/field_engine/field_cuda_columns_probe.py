"""Full field, exact hits, capacity and lifecycle gates for the CUDA scan."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
import os
from pathlib import Path
import threading
import numpy as np
import trimesh
from field_mesh_native_mask_stage_probe import cases, expected_open, baseline, digest
from mesh_field_cuda_columns_v1 import PreparedMeshField
from cuda_columns_handle_v1 import ColumnsHandle

ROOT = Path(__file__).resolve().parents[2]


def cpu_depths(corners, point):
    """Vectorized frozen projected-hit arithmetic, including its roundoff."""
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    x, y = point
    d = (b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])
    l0 = (b[:, 0]-x)*(c[:, 1]-y) - (b[:, 1]-y)*(c[:, 0]-x)
    l1 = (c[:, 0]-x)*(a[:, 1]-y) - (c[:, 1]-y)*(a[:, 0]-x)
    l2 = (a[:, 0]-x)*(b[:, 1]-y) - (a[:, 1]-y)*(b[:, 0]-x)
    sign = np.where(d > 0, 1, -1)
    scale = max(1., float(np.max(np.abs(corners[:, :, :2]))))
    keep = (np.abs(d) > 1e-12*scale*scale) & (l0*sign >= 0) & (l1*sign >= 0) & (l2*sign >= 0)
    z = (l0[keep]*a[keep, 2] + l1[keep]*b[keep, 2] + l2[keep]*c[keep, 2]) / d[keep]
    return np.sort(z)


def main():
    arrays, rows = {}, []
    closed_rejected = 0
    for name, (mesh, pitch, origin, closed) in cases().items():
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        before = [digest(a) for a in (vertices, faces, origin)]
        expected = (baseline.surface_raster_and_flood(vertices, faces, pitch, origin) if closed
                    else expected_open(vertices, faces, pitch, origin))
        with PreparedMeshField(vertices, faces, pitch, origin,
                               columns_library=os.environ['CUDA_COLUMNS_LIBRARY'],
                               edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'],
                               column_library=os.environ['COLUMN_MASK_LIBRARY']) as field:
            legs = []
            for leg in range(2):
                actual = field.evaluate()
                hashes, differences = {}, {}
                for i, label in ((0, 'gmin'), (2, 'surface'), (3, 'solid'), (4, 'distance')):
                    hashes[label] = digest(actual[i])
                    differences[label] = int(np.count_nonzero(actual[i] != expected[i]))
                    arrays[f'{name}_{leg}_{label}'] = actual[i]
                hit_differences = []
                if closed:
                    reference_hits = baseline._kolumntraffar(vertices, faces, pitch, field._anchor, *field.shape[:2])
                    order = np.lexsort((reference_hits[2], reference_hits[1], reference_hits[0]))
                    reference_hits = tuple(a[order] for a in reference_hits)
                    hits = field._rt.query(field._xy)
                    hit_differences = [int(np.count_nonzero(a != b)) if a.shape == b.shape else -1
                                       for a, b in zip(hits, reference_hits)]
                    for label, values in zip(('columns', 'depths', 'signs'), hits):
                        hashes[label] = digest(values)
                        arrays[f'{name}_{leg}_{label}'] = values
                legs.append(dict(hashes=hashes, differences=differences,
                                 shape_equal=actual[1] == expected[1], hit_differences=hit_differences))
            method = field.method
        try:
            field.evaluate()
        except RuntimeError:
            closed_rejected += 1
        rows.append(dict(case=name, closed=closed, method=method, legs=legs,
                         inputs_unchanged=before == [digest(a) for a in (vertices, faces, origin)]))
    box = trimesh.creation.box(extents=[4, 4, 4])
    triangle = np.asarray(box.vertices[box.faces])
    xy = np.array([[.123, .234]])
    capacity_rows = []
    overflow = retries = foreign = invalid = 0
    for leg in range(2):
        for count in (32, 33):
            corners = np.concatenate([triangle + [0, 0, 10 * i] for i in range(count)])
            with ColumnsHandle(os.environ['CUDA_COLUMNS_LIBRARY'], corners, np.zeros(3), 1) as handle:
                errors = []
                def wrong_thread():
                    try:
                        handle.query(xy)
                    except RuntimeError:
                        errors.append(True)
                thread = threading.Thread(target=wrong_thread)
                thread.start()
                thread.join()
                foreign += len(errors)
                for bad in (np.empty((0, 2)), np.zeros((2, 2)), np.array([[np.nan, 0]])):
                    try:
                        handle.query(bad)
                    except ValueError:
                        invalid += 1
                if count == 32:
                    hits = handle.query(xy)
                    analytic_depths = np.array([z for i in range(32) for z in (10 * i - 2, 10 * i + 2)])
                    expected_depths = cpu_depths(corners, xy[0])
                    expected_signs = np.tile([-1, 1], 32)
                    capacity_rows.append(dict(hits=len(hits[0]), depth_exact=np.array_equal(hits[1], expected_depths),
                                              sign_exact=np.array_equal(hits[2], expected_signs),
                                              analytic_depth_differences=int(np.count_nonzero(hits[1] != analytic_depths)),
                                              analytic_max_delta=float(np.max(np.abs(hits[1] - analytic_depths))),
                                              hashes=[digest(a) for a in hits]))
                else:
                    try:
                        handle.query(xy)
                    except RuntimeError as error:
                        overflow += int('status 4' in str(error))
                    try:
                        handle.query(xy)
                    except RuntimeError as error:
                        retries += int('no retry' in str(error))
    gates = dict(full_fields=all(q['shape_equal'] and not any(q['differences'].values()) for r in rows for q in r['legs']),
                 exact_hits=all(q['hit_differences'] == [0, 0, 0] for r in rows if r['closed'] for q in r['legs']),
                 repeat=all(r['legs'][0] == r['legs'][1] for r in rows) and capacity_rows[0] == capacity_rows[1],
                 selection=all(r['method'] == ('cuda_columns' if r['closed'] else 'ordered_gwn') for r in rows),
                 inputs_unchanged=all(r['inputs_unchanged'] for r in rows),
                 close_rejection=closed_rejected == len(rows),
                 full_capacity=all(r['hits'] == 64 and r['depth_exact'] and r['sign_exact'] for r in capacity_rows),
                 overflow_latch=overflow == retries == 2, thread_refusal=foreign == 4,
                 invalid_queries=invalid == 12)
    report = dict(rows=rows, capacity_rows=capacity_rows, gates=gates,
                  status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
                  scope='CUDA software column scan; exact full-field correctness, no timing claim.')
    (ROOT / 'reports/cuda_columns.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'reports/cuda_columns_arrays.npz', **arrays)
    print(json.dumps(report))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
