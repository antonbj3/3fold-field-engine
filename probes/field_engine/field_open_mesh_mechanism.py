"""Observe directional-sign and generalized-winding disagreement before design."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
import igl
import faltkarna_v1_mesh_to_sdf as baseline
ROOT=Path(__file__).resolve().parents[2]


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def fixtures():
    box=trimesh.creation.box(extents=[4,4,4])
    opened=trimesh.Trimesh(box.vertices,box.faces[box.face_normals[:,2]<.5],process=False)
    reversed_mesh=opened.copy();reversed_mesh.faces=reversed_mesh.faces[:,::-1]
    return dict(closed=box,open=opened,reversed_open=reversed_mesh)


def leg():
    rows=[];arrays={}
    for name,mesh in fixtures().items():
        pitch=.25;lo=np.array([-4.031,-4.043,-4.057])
        gmin,shape=baseline._fonster(mesh.vertices,pitch,lo)
        anchor=lo+gmin*pitch
        points=anchor+np.indices(shape).reshape(3,-1).T*pitch
        points[:,:2]+=pitch*baseline.JITTER_FRAC*np.array(baseline._GYLLENE[:2])
        w=np.asarray(igl.winding_number(np.asarray(mesh.vertices),np.asarray(mesh.faces,dtype=np.int64),points)).reshape(shape)
        generalized=np.abs(w)>.5
        signed,_=baseline.solid_via_stralvindning(mesh.vertices,mesh.faces,pitch,anchor,shape,regel='vindning')
        parity,_=baseline.solid_via_stralvindning(mesh.vertices,mesh.faces,pitch,anchor,shape,regel='paritet')
        row=dict(case=name,triangles=len(mesh.faces),watertight=bool(mesh.is_watertight),voxels=int(w.size),
                 generalized_occupied=int(generalized.sum()),signed_occupied=int(signed.sum()),parity_occupied=int(parity.sum()),
                 parity_disagreements=int(np.count_nonzero(parity!=generalized)),signed_disagreements=int(np.count_nonzero(signed!=generalized)),
                 disagreement_fraction=float(np.mean(parity!=generalized)),winding_range=[float(w.min()),float(w.max())],
                 minimum_threshold_distance=float(np.min(np.abs(np.abs(w)-.5))),
                 hashes={k:digest(a) for k,a in dict(generalized=generalized,winding=w,signed=signed,parity=parity).items()})
        rows.append(row)
        for key,a in dict(generalized=generalized,winding=w,signed=signed,parity=parity).items():arrays[name+'_'+key]=a
    return rows,arrays


if __name__=='__main__':
    a,aa=leg();b,bb=leg()
    gates=dict(full_repeat=a==b and all(aa[k].tobytes()==bb[k].tobytes() for k in aa),
               closed_agreement=a[0]['signed_disagreements']==0,
               open_disagreement=a[1]['parity_disagreements']>0,
               reversed_classification=np.array_equal(aa['open_generalized'],aa['reversed_open_generalized']))
    report=dict(rows=a,gates=gates,scope='Synthetic open-mesh sign selection, no unique-solid or RT generalized-winding claim.')
    folder=ROOT/'reports';folder.mkdir(exist_ok=True)
    (folder/'open_mesh_mechanism.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(folder/'open_mesh_mechanism_arrays.npz',**aa)
    print(json.dumps(report));raise SystemExit(0 if all(gates.values()) else 2)
