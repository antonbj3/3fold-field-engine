"""Exact larger-grid squared/signed checks using frozen plate masks and tie cases."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy import ndimage
from native_edt_large_v1 import NativeEDTLarge,validate
ROOT=Path(__file__).resolve().parents[2]


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    inputs=np.load(ROOT/'reports/stage_scale_mechanism_arrays.npz')
    cases={f'plate_{p}':(inputs[f'{p}_0_solid'],p) for p in (2.,1.,.5)}
    site=np.ones((512,2,1),bool);site[0,0,0]=False
    cases.update(single_site=(site,1.),checker=(np.indices((129,3,7)).sum(axis=0)%2==0,.25))
    native=NativeEDTLarge(os.environ['NATIVE_EDT_LARGE_LIBRARY']);rows=[];arrays={}
    for leg in range(2):
        for name,(mask,pitch) in cases.items():
            square_error=0;hashes={}
            for side,binary in enumerate((~mask,mask)):
                indices=ndimage.distance_transform_edt(binary,return_distances=False,return_indices=True)
                expected=np.sum((np.indices(binary.shape,dtype=np.int64)-indices.astype(np.int64))**2,axis=0).astype(np.int32)
                actual=native.squared_distance(binary)
                square_error+=int(np.count_nonzero(actual!=expected));hashes[f'squared{side}']=digest(actual)
                arrays[f'{name}_{leg}_squared{side}']=actual
            sd=native.signed_distance(mask,pitch)
            reference=ndimage.distance_transform_edt(~mask,sampling=(pitch,)*3).astype(np.float32)-ndimage.distance_transform_edt(mask,sampling=(pitch,)*3).astype(np.float32)
            reference=(reference-np.sign(reference)*np.float32(.5*pitch)).astype(np.float32)
            hashes['signed']=digest(sd);arrays[f'{name}_{leg}_signed']=sd
            rows.append(dict(leg=leg,case=name,squared_mismatches=square_error,signed_mismatches=int(np.count_nonzero(sd!=reference)),hashes=hashes))
    invalid=[np.zeros((2,2,2),bool),np.ones((2,2,2),bool),np.zeros((513,1,1),bool),np.zeros((129,129,129),bool),np.zeros((1,2),bool),np.zeros((1,1,1),np.uint8)]
    rejected=0
    for x in invalid:
        try:validate(x)
        except ValueError:rejected+=1
    gates=dict(exact_squared=all(r['squared_mismatches']==0 for r in rows),exact_signed=all(r['signed_mismatches']==0 for r in rows),
               full_repeat=all(rows[i]['hashes']==rows[i+len(cases)]['hashes'] for i in range(len(cases))),invalid_rejected=rejected==len(invalid))
    report=dict(rows=rows,gates=gates,rejected=rejected,library_sha256=hashlib.sha256(Path(os.environ['NATIVE_EDT_LARGE_LIBRARY']).read_bytes()).hexdigest(),scope='Bounded exact larger EDT; no timing claim.')
    (ROOT/'reports/edt_large.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(ROOT/'reports/edt_large_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 2


if __name__=='__main__':raise SystemExit(main())
