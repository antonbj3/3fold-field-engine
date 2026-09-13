"""Exact custom-primitive RT hit-list contract; no throughput claim."""

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
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def sorted_hits(c,z,s):
    order=np.lexsort((s,z,c));return c[order],z[order],s[order]


def occupancy(c,z,s,pitch,origin,shape):
    result=np.zeros(shape,bool)
    if not len(c):return result
    order=np.lexsort((z,c));c,z,s=c[order],z[order],s[order]
    counts=np.bincount(c,minlength=shape[0]*shape[1]);ends=np.cumsum(counts)
    cum=np.r_[np.int64(0),np.cumsum(s,dtype=np.int64)]
    low=float(z.min());span=max(float(z.max())-low,1e-12)
    keys=c.astype(np.float64)+.25+.5*(z-low)/span
    zs=origin[2]+(np.arange(shape[2],dtype=np.float64)+baseline.VOXEL_PROVPUNKT)*pitch
    active=np.flatnonzero(counts);cc=np.repeat(active,shape[2]);zz=np.tile(zs,len(active))
    query=cc.astype(np.float64)+np.clip(.25+.5*(zz-low)/span,0,.999)
    pos=np.searchsorted(keys,query,side='right')
    result.reshape(-1,shape[2])[active]=(cum[ends[cc]]-cum[pos]!=0).reshape(len(active),shape[2])
    return result


def native_hits(v,t,pitch,origin,shape,capacity=64):
    from rt_columns_compact_handle_v1 import ColumnsHandle
    ij=np.indices(shape[:2]).reshape(2,-1).T
    xy=origin[:2]+(ij+baseline.VOXEL_PROVPUNKT)*pitch
    xy[:,0]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[0]
    xy[:,1]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[1]
    with ColumnsHandle(os.environ['RT_COLUMNS_LIBRARY'],os.environ['RT_COLUMNS_PTX'],v[t],origin,len(xy),capacity) as handle:
        try:return 0,handle.query(xy)
        except RuntimeError as e:
            if 'status 4' in str(e):return 4,None
            raise


def main():
    root=Path(__file__).resolve().parents[2];mesh=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);mesh.merge_vertices()
    cases=[(f'plate_{p}',mesh,p) for p in (2.,1.,.5)]
    for shift in (0.,1e6):
        box=trimesh.creation.box(extents=[4,3,2]);box.apply_transform(trimesh.transformations.rotation_matrix(.31,[1,2,3]));box.apply_translation([shift+.13,shift-.17,shift+.23]);cases.append((f'box_{shift}',box,.5))
    arrays={};rows=[];overflow=True
    for name,m,pitch in cases:
        v=np.asarray(m.vertices);t=np.asarray(m.faces);lo=v.min(0)-2*pitch;gmin,shape=baseline._fonster(v,pitch,lo);origin=lo+gmin*pitch
        ref=sorted_hits(*baseline._kolumntraffar(v,t,pitch,origin,*shape[:2]));mask=baseline.solid_via_stralvindning(v,t,pitch,origin,shape)[0]
        for leg in range(2):
            status,hits=native_hits(v,t,pitch,origin,shape)
            if status:raise RuntimeError(f'query status {status}')
            solid=occupancy(*hits,pitch,origin,shape)
            differences=[int(np.count_nonzero(a!=b)) if a.shape==b.shape else -1 for a,b in zip(hits,ref)]
            rows.append(dict(case=name,leg=leg,hits=len(hits[0]),differences=differences,occupancy_differences=int(np.count_nonzero(solid!=mask)),hashes=[digest(a) for a in (*hits,solid)]))
            for label,a in zip(('columns','z','sign','solid'),(*hits,solid)):arrays[f'{name}_{leg}_{label}']=a
        overflow &= native_hits(v,t,pitch,origin,shape,1)[0]==4
    from rt_columns_compact_handle_v1 import ColumnsHandle
    with ColumnsHandle(os.environ['RT_COLUMNS_LIBRARY'],os.environ['RT_COLUMNS_PTX'],v[t],origin,1) as handle:
        empty=handle.query(np.array([[v[:,0].max()+100,v[:,1].max()+100]]))
        overflow &= all(x.size==0 for x in empty)
    gates=dict(exact_hits=all(r['differences']==[0,0,0] for r in rows),exact_occupancy=all(r['occupancy_differences']==0 for r in rows),full_repeat=all(rows[i]['hashes']==rows[i+1]['hashes'] for i in range(0,len(rows),2)),overflow_rejected=bool(overflow))
    report=dict(rows=rows,gates=gates,scope='Synthetic custom RT columns, CPU composite keys; no throughput claim.')
    (root/'reports/rt_columns_compact.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(root/'reports/rt_columns_compact_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 2

if __name__=='__main__':raise SystemExit(main())
