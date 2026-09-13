"""Prepared intervals must enclose world-space double predicates exactly."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np
from rt_interval_prepared_cpu_v1 import fixtures, bounds, frozen_edges, subtract

ROOT=Path(__file__).resolve().parents[2]


def main():
    triangles, queries, anchors = fixtures()
    actual = bounds(triangles, queries, anchors)
    repeated = bounds(triangles, queries, anchors)
    geometry, query, lower, upper, reject = actual
    edges = frozen_edges(triangles, queries)
    outside = (edges > 0).any(1) & (edges < 0).any(1)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    a = np.array([1, -1, 1, -1, 1], np.float32)
    b = np.array([tiny, -tiny, -tiny, tiny, 1], np.float32)
    low = np.array([np.nextafter(np.float32(1),np.float32(-np.inf)), -1, 1, np.nextafter(np.float32(-1),np.float32(-np.inf)), -0.], np.float32)
    high = np.array([1, np.nextafter(np.float32(-1),np.float32(np.inf)), np.nextafter(np.float32(1),np.float32(np.inf)), -1, 0.], np.float32)
    gates = dict(directed_subtraction=subtract(a,b,False).tobytes()==low.tobytes() and subtract(a,b,True).tobytes()==high.tobytes(), coverage=len(triangles)==120016,
                 finite=all(np.isfinite(a).all() for a in actual),
                 enclosure=np.all(lower<=edges) and np.all(edges<=upper),
                 no_false_rejection=not np.any(reject & ~outside),
                 nonvacuous=int(reject.sum())>=len(reject)//4,
                 uncertainty_falls_through=int((outside & ~reject).sum())>0,
                 repeat=all(a.tobytes()==b.tobytes() for a,b in zip(actual,repeated)))
    gates={k:bool(v) for k,v in gates.items()}
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,
                cases=len(triangles),rejected=int(reject.sum()),uncertain_outside=int((outside & ~reject).sum()),
                source_sha256=hashlib.sha256((ROOT/'src/field_engine/rt_interval_prepared_cpu_v1.py').read_bytes()).hexdigest(),
                scope='CPU proof only. One-ulp outward float neighbors of rounded local coordinates enclose exact shifted coordinates; interval subtraction/products enclose frozen world-space double edges. Portable fixtures with local and zero anchors, no tolerance or GPU speed claim.')
    (ROOT/'reports/field_rt_prepared_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_rt_prepared_cpu_arrays.npz',triangles=triangles,queries=queries,anchors=anchors,
                        geometry=geometry,query=query,lower=lower,upper=upper,reject=reject,edges=edges)
    print(json.dumps(report,indent=2));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
