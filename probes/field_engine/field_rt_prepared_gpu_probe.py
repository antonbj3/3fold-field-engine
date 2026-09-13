"""Native preparation and CUDA predicates against complete CPU interval arrays."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
from field_paths import field_path as _field_path
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from rt_interval_prepared_cpu_v1 import fixtures, bounds, frozen_edges

ROOT=Path(__file__).resolve().parents[2]


def main():
    triangles,queries,anchors=fixtures()
    geometry,query,lower,upper,rejected=bounds(triangles,queries,anchors)
    edges=frozen_edges(triangles,queries)
    expected=(geometry,query,lower,upper,rejected.astype(np.uint8),edges)
    library=ct.CDLL(os.environ['RT_PREPARED_PROBE_LIBRARY']);native=library.prepared_probe
    native.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]+[ct.c_void_p]*6;native.restype=ct.c_int
    arrays=dict(triangles=triangles,queries=queries,anchors=anchors);rows=[]
    for leg in range(2):
        actual=tuple(np.empty_like(a) for a in expected)
        status=native(triangles.ctypes.data,queries.ctypes.data,anchors.ctypes.data,len(triangles),*(a.ctypes.data for a in actual))
        if status:raise RuntimeError(f'Prepared interval status {status}')
        differences=[int(np.count_nonzero(a.view(np.uint8)!=b.view(np.uint8))) for a,b in zip(actual,expected)]
        rows.append(dict(leg=leg,differing_bytes=differences,hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in actual],
                         no_false_rejection=bool(not np.any(actual[4].astype(bool) & ~((edges>0).any(1)&(edges<0).any(1))))))
        for i,a in enumerate(actual):arrays[f'{leg}_{i}']=a
    gates=dict(coverage=len(triangles)==120016 and len(rows)==2,exact=all(not any(r['differing_bytes']) for r in rows),
               repeat=rows[0]['hashes']==rows[1]['hashes'],no_false_rejection=all(r['no_false_rejection'] for r in rows))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,
                cases=len(triangles),rejected=int(rejected.sum()),
                input_sha256={name:hashlib.sha256(a.tobytes()).hexdigest() for name,a in (('triangles',triangles),('queries',queries),('anchors',anchors))},
                source_sha256={name:hashlib.sha256((_field_path(ROOT, "rt_columns_prepared_v1/"+name)).read_bytes()).hexdigest() for name in ('host_bounds.hpp','filter.cuh','probe.cu')},
                scope='Complete native preparation and CUDA predicate arrays compared to CPU. Portable geometry/query/anchor inputs retained. No RT traversal or service speed claim.')
    (ROOT/'reports/field_rt_prepared_gpu.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_rt_prepared_gpu_arrays.npz',**arrays)
    print(json.dumps(report),flush=True);return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
