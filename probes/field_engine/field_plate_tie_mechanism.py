"""Measure frozen interpolation/key behavior at retained RT plate mismatches."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
import os
from pathlib import Path
import numpy as np
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline
ROOT=Path(__file__).resolve().parents[2]


def leg():
    mesh=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);mesh.merge_vertices()
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces)
    reference=np.load(ROOT/'reports/stage_scale_mechanism_arrays.npz');candidate=np.load(ROOT/'reports/mesh_rt_stage_arrays.npz');rows=[]
    for pitch in (2.,1.,.5):
        cpu=reference[f'{pitch}_0_solid'];gpu=candidate[f'plate_{pitch}_0_solid']
        lo=v.min(0)-3*pitch;gmin,shape=baseline._fonster(v,pitch,lo);origin=lo+gmin*pitch
        columns,z,sign=baseline._kolumntraffar(v,f,pitch,origin,*shape[:2])
        zmin=float(z.min());zmax=float(z.max());span=max(zmax-zmin,1e-12)
        for index in np.argwhere(cpu!=gpu):
            col=int(index[0]*shape[1]+index[1]);point_z=float(origin[2]+index[2]*pitch)
            selected=columns==col;zs=z[selected];tk=sign[selected]
            key=col+.25+.5*(zs-zmin)/span
            query_key=col+np.clip(.25+.5*(point_z-zmin)/span,0.,.999)
            total=int(np.sum(tk[key>query_key]));nearest=int(np.argmin(np.abs(zs-point_z)))
            rows.append(dict(pitch=pitch,index=index.tolist(),column=col,query_z=point_z,bottom_z=float(v[:,2].min()),
                             nearest_hit_z=float(zs[nearest]),hit_plane_delta=float(zs[nearest]-point_z),
                             nearest_key_delta=float(key[nearest]-query_key),key_signed_count=total,
                             cpu=bool(cpu[tuple(index)]),candidate=bool(gpu[tuple(index)]),zmin=zmin,zmax=zmax))
    return rows


if __name__=='__main__':
    a,b=leg(),leg();gates=dict(full_repeat=a==b,all_bottom=all(r['query_z']==r['bottom_z'] for r in a),
                              reconstructs_cpu=all((r['key_signed_count']!=0)==r['cpu'] for r in a))
    report=dict(rows=a,gates=gates,scope='Frozen boundary-predicate mechanism only, not a corrected candidate.')
    (ROOT/'reports/plate_tie_mechanism.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    raise SystemExit(0 if all(gates.values()) else 2)
