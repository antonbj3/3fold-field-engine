"""Frozen-reference exactness and boundary probes for independent integer EDT."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from field_edt_integer_v1 import squared_distance, signed_distance
ROOT = Path(__file__).resolve().parents[2]


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def measure():
    rows, arrays = [], {}
    with np.load(ROOT/'artifacts/field_rt_changed_query_reference_arrays.npz') as saved:
        masks = {key: saved[key] for key in saved.files if key.endswith('_mask')}
    boundary = np.zeros((5, 7, 9), dtype=bool)
    boundary[0, 0, 0] = True
    masks['boundary_single_site'] = boundary
    masks['boundary_single_hole'] = ~boundary
    masks['checkerboard_ties'] = np.indices((7, 8, 9)).sum(axis=0)%2 == 0
    masks['singleton_axis'] = np.indices((1, 5, 7)).sum(axis=0)%3 == 0
    for key, mask in masks.items():
        square_errors = 0
        grid = np.indices(mask.shape, dtype=np.int64)
        for polarity, binary in enumerate((~mask, mask)):
            indices = ndimage.distance_transform_edt(binary, return_distances=False, return_indices=True)
            reference = np.sum((grid-indices.astype(np.int64))**2, axis=0)
            actual = squared_distance(binary)
            square_errors += int(np.count_nonzero(reference != actual))
            arrays[f'{key}_square_{polarity}'] = actual
        for pitch_index, pitch in enumerate((0.125, 0.3, 0.7483641718372458)):
            pair = [ndimage.distance_transform_edt(b, sampling=(pitch,)*3).astype(np.float32) for b in (~mask, mask)]
            ref = (pair[0]-pair[1]).astype(np.float32)
            ref = (ref-np.sign(ref)*np.float32(0.5*pitch)).astype(np.float32)
            actual = signed_distance(mask, pitch)
            arrays[f'{key}_signed_{pitch_index}'] = actual
            rows.append(dict(mask=key, pitch=pitch, voxels=mask.size,
                             squared_mismatches=square_errors,
                             signed_mismatches=int(np.count_nonzero(ref != actual)),
                             max_error=float(np.max(np.abs(ref-actual))),
                             sign_mismatches=int(np.count_nonzero(np.sign(ref) != np.sign(actual))),
                             finite=bool(np.isfinite(actual).all()), sha256=digest(actual)))
    invalid = [np.zeros((2, 2, 2), bool), np.ones((2, 2, 2), bool),
               np.zeros((2, 2, 2), np.float32), np.zeros((2, 2), bool),
               np.zeros((0, 2, 2), bool), np.zeros((129, 2, 2), bool),
               np.zeros((65, 65, 65), bool)]
    calls = [(mask, 0.125) for mask in invalid]
    calls += [(boundary, pitch) for pitch in (0, -1, float('nan'), float('inf'), (1, 1, 1), True, 1e13, 1e-13)]
    rejections = 0
    for mask, pitch in calls:
        try:
            signed_distance(mask, pitch)
        except ValueError:
            rejections += 1
    return rows, arrays, rejections, len(calls)


if __name__ == '__main__':
    a, aa, rejected_a, total = measure()
    b, bb, rejected_b, _ = measure()
    gates = dict(squared_exact=all(r['squared_mismatches'] == 0 for r in a),
                 signed_exact=all(r['signed_mismatches'] == 0 for r in a),
                 full_repeat=a == b and all(digest(aa[k]) == digest(bb[k]) for k in aa),
                 finite_signs=all(r['finite'] and r['sign_mismatches'] == 0 for r in a),
                 unsupported_rejected=rejected_a == rejected_b == total)
    report = dict(rows=a, gates=gates, rejected=rejected_a, invalid_total=total,
                  array_hashes={k: digest(v) for k, v in aa.items()}, timing_measured=False)
    (ROOT/'artifacts/field_edt_integer_v1.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(ROOT/'artifacts/field_edt_integer_v1_arrays.npz', **aa)
    print(json.dumps(dict(gates=gates, rows=len(a), rejected=rejected_a)))
    raise SystemExit(0 if all(gates.values()) else 1)
