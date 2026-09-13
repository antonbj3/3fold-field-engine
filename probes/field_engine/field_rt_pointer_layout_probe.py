"""Measure physical pointer layout versus row-major serialization before correction."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import ctypes
import json
from pathlib import Path
import numpy as np
from field_rt_mesh_probe import digest
ROOT=Path(__file__).resolve().parents[2]
def measure():
    data=np.load(ROOT/'artifacts'/'field_rt_changed_query_reference_arrays.npz');rows=[]
    for key in data.files:
        if not key.endswith(('_rays','_corners')):continue
        a=data[key];physical=np.frombuffer(ctypes.string_at(a.ctypes.data,a.nbytes),dtype=a.dtype).reshape(a.shape)
        contiguous=np.ascontiguousarray(a)
        fixed=np.frombuffer(ctypes.string_at(contiguous.ctypes.data,contiguous.nbytes),dtype=a.dtype).reshape(a.shape)
        rows.append(dict(array=key,shape=list(a.shape),strides=list(a.strides),c_contiguous=bool(a.flags.c_contiguous),
                         raw_pointer_mismatches=int(np.count_nonzero(a!=physical)),contiguous_pointer_mismatches=int(np.count_nonzero(a!=fixed)),
                         logical_hash=digest(a),physical_hash=digest(physical),fixed_hash=digest(fixed)))
    return rows
if __name__=='__main__':
    a,b=measure(),measure();gates=dict(full_repeat=a==b,contiguous_exact=all(r['contiguous_pointer_mismatches']==0 for r in a))
    report=dict(rows=a,gates=gates,timing_measured=False)
    (ROOT/'artifacts'/'field_rt_pointer_layout.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));raise SystemExit(0 if all(gates.values()) else 1)
