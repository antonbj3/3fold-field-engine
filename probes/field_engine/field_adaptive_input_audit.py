"""Locate an adaptive verification mismatch using complete captured inputs.

This CPU audit keeps the original cross-host failure and replays each host's
actual input. It does not change the adaptive algorithm or its tolerances.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
from field_paths import field_path as _field_path
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def sorted_faces(faces):
    return faces[np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))]


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from field_adaptive_blocks_probe import observe
    reports = ROOT/'reports'
    captures, rows, comparisons = {}, [], []
    with np.load(reports/'adaptive_inputs_local.npz') as local, np.load(reports/'adaptive_inputs_modal.npz') as remote:
        schema = set(local.files) == set(remote.files) and len(local.files) == 60
        changed = {name for name in local.files if not same(local[name], remote[name])}
        expected_changed = {'mesh_0_faces', 'mesh_1_faces', 'observe_0_points', 'observe_1_points'}
        for leg in (0, 1):
            faces_a, faces_b = local[f'mesh_{leg}_faces'], remote[f'mesh_{leg}_faces']
            points_a, points_b = local[f'observe_{leg}_points'], remote[f'observe_{leg}_points']
            comparisons.append(dict(leg=leg, vertices_exact=same(local[f'mesh_{leg}_vertices'], remote[f'mesh_{leg}_vertices']),
                                    oriented_triangle_multiset_exact=same(sorted_faces(faces_a), sorted_faces(faces_b)),
                                    changed_face_rows=int(np.any(faces_a != faces_b, axis=1).sum()),
                                    changed_point_rows=int(np.any(points_a != points_b, axis=1).sum()),
                                    interior_points_exact=same(points_a[4000:], points_b[4000:])))
        for host, inputs, expected_name in [('local', local, 'adaptive_blocks_arrays.npz'), ('modal', remote, 'adaptive_blocks_modal_arrays.npz')]:
            with np.load(reports/expected_name) as expected:
                for leg in (0, 1):
                    vertices, faces = inputs[f'mesh_{leg}_vertices'], inputs[f'mesh_{leg}_faces']
                    samples, origin, pitch, points = [inputs[f'observe_{leg}_{key}'] for key in ('samples', 'origin', 'pitch', 'points')]
                    centers = vertices[faces].mean(axis=1)
                    rng = np.random.default_rng(20260912)
                    selected = centers[rng.choice(len(centers), size=min(4000, len(centers)), replace=False)]
                    interior = origin + rng.uniform(-1, np.array(samples.shape)+1, (8193, 3))*float(pitch)
                    regenerated = np.concatenate((selected, interior))
                    for repeat in (0, 1):
                        _, actual = observe(samples, origin, float(pitch), points)
                        exact = all(same(value, expected[f'leg_{leg}_{key}']) for key, value in actual.items())
                        rows.append(dict(host=host, leg=leg, repeat=repeat, regenerated_points_exact=same(points, regenerated),
                                         all_outputs_exact=exact, bytes_compared=sum(value.nbytes for value in actual.values())))
                        for key, value in actual.items():
                            captures[f'{host}_{leg}_{repeat}_{key}'] = value
    with np.load(reports/'adaptive_mesh_transplant_arrays.npz') as transplanted, np.load(reports/'adaptive_blocks_modal_arrays.npz') as expected:
        transplant_exact = set(transplanted.files)==set(expected.files) and all(same(transplanted[key], expected[key]) for key in expected.files)
        transplanted_bytes = sum(expected[key].nbytes for key in expected.files)
    transplant_report_exact = (reports/'adaptive_mesh_transplant.json').read_bytes() == (reports/'adaptive_blocks_modal.json').read_bytes()
    gates = dict(input_schema=schema, differences_limited=changed==expected_changed,
                 same_oriented_geometry=all(r['vertices_exact'] and r['oriented_triangle_multiset_exact'] for r in comparisons),
                 captured_order_mechanism=all(r['changed_face_rows']==28 and r['changed_point_rows']==23 and r['interior_points_exact'] for r in comparisons),
                 query_generation=all(r['regenerated_points_exact'] for r in rows),
                 same_input_replay=len(rows)==8 and all(r['all_outputs_exact'] for r in rows),
                 full_mesh_transplant=transplant_exact and transplant_report_exact)
    result = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates,
                  comparisons=comparisons, rows=rows, transplanted_bytes=transplanted_bytes,
                  source_sha256={name:hashlib.sha256((_field_path(ROOT, name)).read_bytes()).hexdigest() for name in ('adaptive_sample_blocks_v1.py', 'field_adaptive_blocks_probe.py', 'field_adaptive_input_audit.py')},
                  scope='CPU audit. First captured difference is mesh face-row order, not vertex positions or oriented triangle content. '
                        'Every captured point is regenerated from that order and the original RNG. Both hosts outputs replay exactly on the same local CPU. '
                        'Replacing only the generated mesh with the captured cloud mesh reproduces the entire cloud report/archive. '
                        'The specific exporter/library instruction causing the permutation is not identified; original cross-host verification remains failed.')
    (reports/'adaptive_input_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(reports/'adaptive_input_replay_arrays.npz', **captures)
    print(json.dumps(result, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
