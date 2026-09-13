"""CPU enclosure and no-false-rejection proof, including boundary uncertainty."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np
from rt_interval_cpu_v1 import fixtures, bounds, frozen_edges, subtract

ROOT = Path(__file__).resolve().parents[2]


def main():
    triangles, queries = fixtures()
    first = bounds(triangles, queries)
    second = bounds(triangles, queries)
    lower, upper, reject = first
    edges = frozen_edges(triangles, queries)
    outside = (edges > 0).any(1) & (edges < 0).any(1)
    zero_a = np.array([1., 0., -0., 0., -0.], np.float32)
    zero_b = np.array([1., 0., -0., -0., 0.], np.float32)
    zero_expected = np.array([-0., -0., -0., 0., -0.], np.float32)
    controls = dict(directed_zero=subtract(zero_a, zero_b, False).tobytes() == zero_expected.tobytes(), coverage=len(triangles) == 120016,
                    finite=np.isfinite(lower).all() and np.isfinite(upper).all(),
                    enclosure=np.all(lower <= edges) and np.all(edges <= upper),
                    no_false_rejection=not np.any(reject & ~outside),
                    nonvacuous=np.count_nonzero(reject) >= len(reject) // 4,
                    uncertainty_falls_through=np.count_nonzero(outside & ~reject) > 0,
                    repeat=all(a.dtype == b.dtype and a.tobytes() == b.tobytes() for a, b in zip(first, second)))
    controls = {k: bool(v) for k, v in controls.items()}
    report = dict(status='VERIFIED-FRESH' if all(controls.values()) else 'OWN-GATE-FAIL', gates=controls,
                  cases=len(triangles), rejected=int(reject.sum()), uncertain_outside=int((outside & ~reject).sum()),
                  source_sha256=hashlib.sha256((ROOT/'src/field_engine/rt_interval_cpu_v1.py').read_bytes()).hexdigest(),
                  scope='Independent CPU interval arithmetic only. Float endpoints outward-bound frozen double differences, products and subtraction. Strict mixed signs imply rejection for either triangle orientation; uncertain signs fall through. No GPU or speed claim.')
    (ROOT/'reports/field_rt_interval_cpu.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT/'reports/field_rt_interval_cpu_arrays.npz', triangles=triangles, queries=queries,
                        lower=lower, upper=upper, reject=reject, edges=edges)
    print(json.dumps(report, indent=2))
    return 0 if all(controls.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
