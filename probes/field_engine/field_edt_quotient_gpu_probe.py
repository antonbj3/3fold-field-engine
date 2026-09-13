"""Complete CUDA signed-floor outputs against independent exact integer division."""

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
from edt_quotient_cpu_v1 import BOUND, DENSE_DENOMINATORS, boundary_cases

ROOT = Path(__file__).resolve().parents[2]


def main():
    library = ct.CDLL(os.environ['EDT_QUOTIENT_PROBE_LIBRARY'])
    dense = library.quotient_dense
    dense.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_void_p]; dense.restype = ct.c_int
    pairs = library.quotient_pairs
    pairs.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]; pairs.restype = ct.c_int
    numerators = np.arange(-BOUND, BOUND+1, dtype=np.int64)
    n, d = boundary_cases()
    rows, arrays = [], dict(boundary_n=n, boundary_d=d)
    for leg in range(2):
        for denominator in (DENSE_DENOMINATORS if leg == 0 else DENSE_DENOMINATORS[::-1]):
            result = np.empty(len(numerators), np.int32)
            status = dense(-BOUND, len(result), denominator, result.ctypes.data)
            if status:raise RuntimeError(f'Dense quotient status {status}')
            expected = (numerators // denominator).astype(np.int32)
            rows.append(dict(case=str(denominator), leg=leg, cases=len(result), exact=result.tobytes()==expected.tobytes(), hash=hashlib.sha256(result.tobytes()).hexdigest()))
            arrays[f'dense_{denominator}_{leg}'] = result
        result = np.empty_like(n)
        status = pairs(n.ctypes.data, d.ctypes.data, len(n), result.ctypes.data)
        if status:raise RuntimeError(f'Boundary quotient status {status}')
        rows.append(dict(case='boundaries', leg=leg, cases=len(result), exact=result.tobytes()==(n.astype(np.int64)//d).astype(np.int32).tobytes(), hash=hashlib.sha256(result.tobytes()).hexdigest()))
        arrays[f'boundaries_{leg}'] = result
    holder = np.empty(1, np.int32)
    pointer = holder.ctypes.data
    controls = [dense(0,0,2,pointer)==1, dense(0,1,1,pointer)==1, dense(0,1,1023,pointer)==1,
                dense(-BOUND-1,1,2,pointer)==1, dense(BOUND,2,2,pointer)==1, dense(0,1,2,None)==1]
    gates = dict(coverage=len(rows)==16, exact=all(r['exact'] for r in rows),
                 repeat=all(len({r['hash'] for r in rows if r['case']==name})==1 for name in {r['case'] for r in rows}),
                 invalid=all(controls) and len(controls)==6)
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates, rows=rows, controls=controls,
                  source_sha256={name:hashlib.sha256((_field_path(ROOT, "edt_pair_quotient_v1/"+name)).read_bytes()).hexdigest() for name in ('quotient.cuh','probe.cu')},
                  scope='Two complete 14680071-case dense integer-floor proofs, plus all-denominator boundaries and invalid inputs. Corrected result compared to exact integer floor; float estimate accuracy is not an acceptance criterion. No EDT/service speed claim.')
    (ROOT/'reports/field_edt_quotient_gpu.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_edt_quotient_gpu_arrays.npz',**arrays)
    print(json.dumps(report),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
