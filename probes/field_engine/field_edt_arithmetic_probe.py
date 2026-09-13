"""Measure squared-grid distance and rounding before a separate EDT backend."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from field_rt_mesh_probe import digest
ROOT=Path(__file__).resolve().parents[2]

def measure():
    data=np.load(ROOT/'artifacts'/'field_rt_changed_query_reference_arrays.npz');rows=[]
    for key in data.files:
        if not key.endswith('_mask'):continue
        mask=data[key];grid=np.indices(mask.shape,dtype=np.int64)
        for pitch in (0.125,0.3,0.7483641718372458):
            reference=[];double=[];single=[];max_square=0;feature_hash=[]
            for binary in (~mask,mask):
                distance,indices=ndimage.distance_transform_edt(binary,sampling=(pitch,)*3,return_indices=True)
                delta=grid-indices.astype(np.int64);square=np.sum(delta*delta,axis=0,dtype=np.int64)
                max_square=max(max_square,int(square.max()));feature_hash.append(digest(indices))
                reference.append(distance.astype(np.float32))
                double.append((np.sqrt(square.astype(np.float64))*pitch).astype(np.float32))
                single.append(np.sqrt(square.astype(np.float32))*np.float32(pitch))
            def signed(pair):
                sd=(pair[0]-pair[1]).astype(np.float32)
                return (sd-np.sign(sd)*np.float32(0.5*pitch)).astype(np.float32)
            ref=signed(reference);d=signed(double);s=signed(single)
            rows.append(dict(mask=key,pitch=pitch,voxels=mask.size,max_square=max_square,
                             double_mismatches=int(np.count_nonzero(ref!=d)),single_mismatches=int(np.count_nonzero(ref!=s)),
                             single_max_error=float(np.max(np.abs(ref-s))),double_max_error=float(np.max(np.abs(ref-d))),
                             reference_sha256=digest(ref),double_sha256=digest(d),single_sha256=digest(s),feature_hashes=feature_hash))
    return rows

if __name__=='__main__':
    a,b=measure(),measure()
    gates=dict(full_tables_hashes_repeat=a==b,double_reconstruction_exact=all(r['double_mismatches']==0 for r in a),
               finite=all(np.isfinite(r['single_max_error']) and np.isfinite(r['double_max_error']) for r in a))
    report=dict(rows=a,gates=gates,timing_measured=False,scope='Nearest-feature indices remain supplied by frozen SciPy EDT; no replacement algorithm.')
    (ROOT/'artifacts'/'field_edt_arithmetic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));raise SystemExit(0 if all(gates.values()) else 1)
