"""Full closed/open mesh stage controls, with original plate and translated grid."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import trimesh
import igl
from scipy import ndimage
import faltkarna_v1_mesh_to_sdf as baseline
from mesh_field_columns_v1 import PreparedMeshField
ROOT=Path(__file__).resolve().parents[2]


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def cases():
    plate=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);plate.merge_vertices()
    out={f'plate_{p}':(plate,p,plate.vertices.min(0)-3*p,True) for p in (2.,1.,.5)}
    rotated=trimesh.creation.box(extents=[4,3,2]);rotated.apply_transform(trimesh.transformations.rotation_matrix(.37,[1,2,3]));rotated.apply_translation([.13,-.21,.37])
    for shift in (0.,1e6):
        m=rotated.copy();m.apply_translation([shift]*3);out[f'rotated_{shift}']=(m,.25,m.vertices.min(0)-1,True)
    box=trimesh.creation.box(extents=[4,4,4]);opened=trimesh.Trimesh(box.vertices,box.faces[box.face_normals[:,2]<.5],process=False)
    reverse=opened.copy();reverse.faces=reverse.faces[:,::-1]
    out['open']=(opened,.25,np.array([-4.031,-4.043,-4.057]),False)
    out['reversed_open']=(reverse,.25,np.array([-4.031,-4.043,-4.057]),False)
    return out


def expected_open(v,f,pitch,lo):
    gmin,shape=baseline._fonster(v,pitch,lo);anchor=lo+gmin*pitch
    points=anchor+np.indices(shape).reshape(3,-1).T*pitch
    points[:,:2]+=pitch*baseline.JITTER_FRAC*np.array(baseline._GYLLENE[:2])
    solid=(np.abs(igl.winding_number(v,f,points))>.5).reshape(shape)
    sd=ndimage.distance_transform_edt(~solid,sampling=(pitch,)*3).astype(np.float32)-ndimage.distance_transform_edt(solid,sampling=(pitch,)*3).astype(np.float32)
    sd=(sd-np.sign(sd)*np.float32(.5*pitch)).astype(np.float32)
    surface=solid & ~ndimage.binary_erosion(solid,ndimage.generate_binary_structure(3,1))
    return gmin,shape,surface,solid,sd


def main():
    rows=[];arrays={};unchanged=True;closed_rejected=0
    for name,(mesh,pitch,lo,closed) in cases().items():
        v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces);before=[digest(x) for x in (v,f,lo)]
        reference=baseline.surface_raster_and_flood(v,f,pitch,lo) if closed else expected_open(v,f,pitch,lo)
        legs=[]
        with PreparedMeshField(v,f,pitch,lo,winding_library=os.environ['RT_COLUMNS_LIBRARY'],ptx=os.environ['RT_COLUMNS_PTX'],edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY']) as field:
            for leg in range(2):
                result=field.evaluate();hashes={};differences={}
                for i,key in ((0,'gmin'),(2,'surface'),(3,'solid'),(4,'distance')):
                    hashes[key]=digest(result[i]);differences[key]=int(np.count_nonzero(result[i]!=reference[i]))
                    arrays[f'{name}_{leg}_{key}']=result[i]
                legs.append(dict(hashes=hashes,differences=differences,shape_equal=result[1]==reference[1]))
            method=field.method
        try:field.evaluate()
        except RuntimeError:closed_rejected+=1
        unchanged &= before==[digest(x) for x in (v,f,lo)]
        rows.append(dict(case=name,closed=closed,method=method,shape=list(reference[1]),legs=legs))
    def exact(row,keys):return all(q['shape_equal'] and all(q['differences'][k]==0 for k in keys) for q in row['legs'])
    gates=dict(full_repeat=all(r['legs'][0]['hashes']==r['legs'][1]['hashes'] for r in rows),
               closed_exact=all(exact(r,('gmin','surface','solid','distance')) for r in rows if r['closed']),
               open_occupancy_exact=all(exact(r,('gmin','solid')) for r in rows if not r['closed']),
               open_distance_surface_exact=all(exact(r,('surface','distance')) for r in rows if not r['closed']),
               method_selection=all(r['method']==('rt_columns' if r['closed'] else 'ordered_gwn') for r in rows),
               inputs_unchanged=bool(unchanged),closed_rejection=closed_rejected==len(rows))
    report=dict(rows=rows,gates=gates,closed_rejections=closed_rejected,scope='Bounded mesh-stage correctness, not full application or timing acceptance.')
    folder=ROOT/'reports';folder.mkdir(exist_ok=True)
    (folder/'mesh_columns_stage.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(folder/'mesh_columns_stage_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 2


if __name__=='__main__':raise SystemExit(main())
