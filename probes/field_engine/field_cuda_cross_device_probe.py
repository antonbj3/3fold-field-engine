"""Audit complete saved CUDA field/hit arrays across two hardware captures."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = {'plate_2.0': True, 'plate_1.0': True, 'plate_0.5': True,
         'rotated_0.0': True, 'rotated_1000000.0': True,
         'open': False, 'reversed_open': False}
SOURCE_GATES = {'full_fields', 'exact_hits', 'repeat', 'selection', 'inputs_unchanged',
                'close_rejection', 'full_capacity', 'overflow_latch', 'thread_refusal', 'invalid_queries'}


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def audit(report, arrays):
    expected = {}
    rows = report['rows']
    coverage = len(rows) == len(CASES) and {r['case'] for r in rows} == set(CASES)
    repeats = True
    for row in rows:
        name = row['case']
        if name not in CASES or row['closed'] != CASES[name] or len(row['legs']) != 2:
            coverage = False
            continue
        keys = ['gmin', 'surface', 'solid', 'distance']
        if row['closed']:
            keys += ['columns', 'depths', 'signs']
        for leg, result in enumerate(row['legs']):
            coverage &= set(result['hashes']) == set(keys)
            for key in keys:
                expected[f'{name}_{leg}_{key}'] = result['hashes'].get(key)
        for key in keys:
            a, b = f'{name}_0_{key}', f'{name}_1_{key}'
            repeats &= a in arrays and b in arrays and digest(arrays[a]) == digest(arrays[b])
    inventory = set(expected) == set(arrays)
    integrity = inventory and all(digest(arrays[k]) == h for k, h in expected.items())
    return dict(source_gates=set(report['gates']) == SOURCE_GATES and all(report['gates'].values()),
                coverage=bool(coverage), inventory=inventory, integrity=integrity,
                repeats=bool(repeats))


def main():
    directory = ROOT / 'reports'
    cloud = json.loads((directory / 'cuda_columns.json').read_text())
    local = json.loads((directory / 'cuda_columns_local.json').read_text())
    with np.load(directory / 'cuda_columns_arrays.npz', allow_pickle=False) as source:
        ca = dict(source)
    with np.load(directory / 'cuda_columns_local_arrays.npz', allow_pickle=False) as source:
        la = dict(source)
    audits = dict(cloud=audit(cloud, ca), local=audit(local, la))
    rows = []
    for key in sorted(set(ca) | set(la)):
        if key not in ca or key not in la:
            rows.append(dict(array=key, exact=False, reason='missing'))
            continue
        a, b = ca[key], la[key]
        layout = a.shape == b.shape and a.dtype == b.dtype
        exact = layout and a.tobytes() == b.tobytes()
        row = dict(array=key, shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes, exact=exact)
        if layout:
            row.update(differences=int(np.count_nonzero(a != b)),
                       max_abs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)), initial=0)),
                       cloud_sha256=digest(a), local_sha256=digest(b))
        rows.append(row)
    # Metadata and missing-array corruption must not pass the inventory audit.
    missing = dict(la)
    missing.pop(next(iter(missing)))
    changed = json.loads(json.dumps(local))
    changed['rows'][0]['legs'][0]['hashes']['solid'] = '0' * 64
    rejected = not all(audit(local, missing).values()) and not all(audit(changed, la).values())
    gates = dict(capture_gates=all(v['source_gates'] for v in audits.values()),
                 inventories=all(v['coverage'] and v['inventory'] and v['integrity'] for v in audits.values()),
                 within_device_repeat=all(v['repeats'] for v in audits.values()),
                 cross_device_exact=bool(rows) and all(r['exact'] for r in rows),
                 corruption_rejected=bool(rejected))
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
                  cloud_hardware=cloud.get('hardware'), local_hardware=local.get('hardware'),
                  audits=audits, rows=rows, gates=gates, compared_bytes=sum(r.get('bytes', 0) for r in rows),
                  scope='Seven synthetic closed/open fixtures, complete CUDA scan fields/hits; no throughput claim.')
    (directory / 'cuda_columns_cross_device.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(gates=gates, arrays=len(rows), bytes=report['compared_bytes'])))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
