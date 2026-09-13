"""Exact invalid-input status and untouched-buffer probe for native EDT."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def measure(library):
    rows=[]
    cases=[('null_input',(2,2,2),'null'),('null_output',(2,2,2),'output_null'),
           ('zero_x',(0,2,2),'mixed'),('negative_y',(2,-1,2),'mixed'),
           ('zero_z',(2,2,0),'mixed'),('large_x',(129,2,2),'mixed'),
           ('large_y',(2,129,2),'mixed'),('large_z',(2,2,129),'mixed'),
           ('total_bound',(65,65,65),'mixed'),('empty',(2,2,2),'empty'),
           ('full',(2,2,2),'full'),('nonboolean',(2,2,2),'bad')]
    for name,shape,kind in cases:
        source=np.zeros(8,np.uint8);source[0]=1
        if kind=='empty':source[:]=0
        if kind=='full':source[:]=1
        if kind=='bad':source[7]=2
        output=np.full(8,-123456789,np.int32)
        before_input=digest(source);before_output=digest(output)
        code=library.edt_squared(None if kind=='null' else source.ctypes.data,*shape,
                                 None if kind=='output_null' else output.ctypes.data)
        rows.append(dict(case=name,status=code,input_unchanged=digest(source)==before_input,
                         output_unchanged=digest(output)==before_output,
                         input_sha256=digest(source),output_sha256=digest(output)))
    return rows


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Requires empty CUDA visibility before process start')
    path=Path(os.environ['EDT_NATIVE_LIBRARY']);library=ct.CDLL(str(path))
    library.edt_squared.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
    library.edt_squared.restype=ct.c_int
    a,b=measure(library),measure(library)
    gates=dict(status_one=all(r['status']==1 for r in a),
               untouched=all(r['input_unchanged'] and r['output_unchanged'] for r in a),
               repeat=a==b)
    report=dict(rows=a,gates=gates,library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                timing_measured=False,valid_cuda_calls=0)
    (ROOT/'artifacts/field_edt_native_rejection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));raise SystemExit(0 if all(gates.values()) else 1)
