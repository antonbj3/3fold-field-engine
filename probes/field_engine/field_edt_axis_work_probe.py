"""Observe separable EDT work and intermediate reachability before acceleration."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np
from field_edt_integer_v1 import squared_distance
ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def measure():
    rows = []
    with np.load(ROOT/'artifacts/field_rt_changed_query_reference_arrays.npz') as data:
        for key in data.files:
            if not key.endswith('_mask'):
                continue
            mask = data[key]
            for polarity, binary in enumerate((~mask, mask)):
                sentinel = sum((n-1)**2 for n in binary.shape)+1
                distance = np.where(binary, sentinel, 0).astype(np.int64)
                stages = []
                for axis, length in enumerate(binary.shape):
                    source = np.moveaxis(distance, axis, -1)
                    result = np.empty_like(source)
                    positions = np.arange(length, dtype=np.int64)
                    for target in range(length):
                        result[..., target] = np.min(source+(positions-target)**2, axis=-1)
                    distance = np.moveaxis(result, -1, axis)
                    finite = distance < sentinel
                    stages.append(dict(axis=axis, length=length,
                                       candidates=int(binary.size*length),
                                       unreached=int(np.count_nonzero(~finite)),
                                       max_reached_square=int(distance[finite].max()),
                                       sha256=digest(distance)))
                expected = squared_distance(binary)
                rows.append(dict(mask=key, polarity=polarity, shape=list(binary.shape),
                                 voxels=binary.size, stages=stages,
                                 candidates=sum(s['candidates'] for s in stages),
                                 expected_candidates=int(binary.size*sum(binary.shape)),
                                 input_mask_bytes=binary.nbytes,
                                 signed_float32_output_bytes=int(binary.size*4),
                                 squared_int64_output_bytes=distance.nbytes,
                                 final_mismatches=int(np.count_nonzero(distance != expected))))
    return rows


if __name__ == '__main__':
    a, b = measure(), measure()
    gates = dict(full_tables_repeat=a == b,
                 oracle_exact=all(r['final_mismatches'] == 0 for r in a),
                 final_reachable=all(r['stages'][-1]['unreached'] == 0 for r in a),
                 accounting=all(r['candidates'] == r['expected_candidates'] for r in a))
    report = dict(rows=a, gates=gates, timing_measured=False,
                  byte_scope='Explicit array payload only; no measured bus or allocator traffic.')
    (ROOT/'artifacts/field_edt_axis_work.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(gates=gates, rows=len(a))))
    raise SystemExit(0 if all(gates.values()) else 1)
