"""Dense signed-floor and all-denominator boundary proof before CUDA execution."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np
from edt_quotient_cpu_v1 import BOUND, DENSE_DENOMINATORS, corrected_floor, boundary_cases

ROOT = Path(__file__).resolve().parents[2]


def main():
    numerators = np.arange(-BOUND, BOUND + 1, dtype=np.int64)
    rows, arrays = [], {}
    for d in DENSE_DENOMINATORS:
        expected = (numerators // d).astype(np.int32)
        actual = corrected_floor(numerators, d)
        repeated = corrected_floor(numerators, d)
        rows.append(dict(denominator=d, cases=len(numerators), exact=actual.tobytes() == expected.tobytes(), repeat=actual.tobytes() == repeated.tobytes()))
        arrays[f'dense_{d}'] = actual
    n, d = boundary_cases()
    actual = corrected_floor(n, d)
    arrays.update(boundary_n=n, boundary_d=d, boundary_floor=actual)
    gates = dict(dense_coverage=sum(r['cases'] for r in rows) == 14680071,
                 dense_exact=all(r['exact'] for r in rows), repeat=all(r['repeat'] for r in rows),
                 boundary_coverage=len(np.unique(d)) == 511,
                 boundary_exact=actual.tobytes() == (n.astype(np.int64) // d).astype(np.int32).tobytes())
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates, rows=rows,
                  boundary_cases=len(n), source_sha256=hashlib.sha256((ROOT/'src/field_engine/edt_quotient_cpu_v1.py').read_bytes()).hexdigest(),
                  scope='CPU arithmetic proof only. Seven complete numerator ranges [-2^20,2^20], plus boundaries for all 511 EDT denominators. Float estimates are corrected by exact integer inequalities; no tolerance or GPU speed claim.')
    (ROOT/'reports/field_edt_quotient_cpu.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT/'reports/field_edt_quotient_cpu_arrays.npz', **arrays)
    print(json.dumps(report, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
