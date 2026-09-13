"""Fixed ordered GWN against independent external winding and repeat controls."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
import numpy as np
import igl
from field_open_mesh_mechanism import fixtures,digest,ROOT,baseline
from generalized_winding_ordered_v1 import generalized_winding


def main():
    rows=[];arrays={};unchanged=True
    for name,mesh in fixtures().items():
        pitch=.25;lo=np.array([-4.031,-4.043,-4.057]);gmin,shape=baseline._fonster(mesh.vertices,pitch,lo)
        q=lo+gmin*pitch+np.indices(shape).reshape(3,-1).T*pitch
        q[:,:2]+=pitch*baseline.JITTER_FRAC*np.array(baseline._GYLLENE[:2])
        v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces);before=[digest(x) for x in (v,f,q)]
        reference=np.asarray(igl.winding_number(v,f,q)).ravel()
        a=generalized_winding(v,f,q);b=generalized_winding(v,f,q)
        unchanged &= before==[digest(x) for x in (v,f,q)]
        row=dict(case=name,max_oracle_error=float(np.max(np.abs(a-reference))),
                 classification_mismatches=int(np.count_nonzero((np.abs(a)>.5)!=(np.abs(reference)>.5))),
                 full_repeat=a.tobytes()==b.tobytes(),sha256=digest(a))
        rows.append(row);arrays[name+'_winding']=a;arrays[name+'_mask']=np.abs(a)>.5
    invalid=0
    for faces,points in ((f,np.array([v[0]])),(np.array([[-1,0,1]]),q[:1])):
        try:generalized_winding(v,faces,points)
        except ValueError:invalid+=1
    gates=dict(full_repeat=all(r['full_repeat'] for r in rows),oracle_bound=all(r['max_oracle_error']<=1e-12 for r in rows),
               exact_classification=all(r['classification_mismatches']==0 for r in rows),
               reversed_classification=np.array_equal(arrays['open_mask'],arrays['reversed_open_mask']),
               invalid_rejected=invalid==2,inputs_unchanged=bool(unchanged))
    report=dict(rows=rows,gates=gates,invalid_rejections=invalid,scope='Synthetic direct CPU GWN, no arbitrary surface or RT speed claim.')
    folder=ROOT/'reports';folder.mkdir(exist_ok=True)
    (folder/'gwn_ordered.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(folder/'gwn_ordered_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 2


if __name__=='__main__':raise SystemExit(main())
