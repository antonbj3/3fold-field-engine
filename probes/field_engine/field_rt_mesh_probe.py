"""Measure triangulated reference fields, then compare the frozen local RT seam."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy import ndimage
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline
from field_rt_winding_probe import guard

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'artifacts'

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def fixtures():
    sphere=trimesh.creation.icosphere(subdivisions=3,radius=2)
    fine=trimesh.creation.icosphere(subdivisions=4,radius=2)
    moved=sphere.copy(); moved.apply_translation([1.,0.25,0.125])
    box=trimesh.creation.box(extents=[4,3,2])
    box=box.subdivide().subdivide().subdivide()
    box.apply_transform(trimesh.transformations.rotation_matrix(0.37,[1,2,3]))
    return dict(sphere=sphere,sphere_fine=fine,
                sphere_overlap=trimesh.util.concatenate([sphere,moved]),
                subdivided_rotated_box=box)

def reference():
    arrays,rows={},[]
    for name,mesh in fixtures().items():
        pitch=0.125; origin=np.asarray(mesh.bounds).min(axis=0)-0.5
        gmin,shape,_,solid,sd=baseline.surface_raster_and_flood(mesh.vertices,mesh.faces,pitch,origin)
        anchor=origin+gmin*pitch
        points=anchor+np.indices(shape).reshape(3,-1).T*pitch
        points[:,0]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[0]
        points[:,1]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[1]
        columns,z,sign=baseline._kolumntraffar(mesh.vertices,mesh.faces,pitch,anchor,shape[0],shape[1])
        arrays[name+'_corners']=np.asarray(mesh.vertices[mesh.faces]-anchor,dtype='<f4')
        arrays[name+'_rays']=np.asarray(points-anchor,dtype='<f4')
        arrays[name+'_solid']=solid; arrays[name+'_distance']=sd
        rows.append(dict(case=name,triangles=len(mesh.faces),shape=list(map(int,shape)),pitch=pitch,
                         voxels=int(solid.size),occupied=int(solid.sum()),columns=int(shape[0]*shape[1]),hits=len(z),
                         mask_sha256=digest(solid),distance_sha256=digest(sd),
                         hit_sha256=digest(np.column_stack([columns,z,sign]))))
    return rows,arrays

def measure_reference():
    first,a=reference(); second,b=reference()
    gates=dict(full_arrays_repeat=all(np.array_equal(a[k],b[k]) for k in a),
               full_tables_repeat=first==second,finite=all(np.isfinite(v).all() for v in a.values()))
    report=dict(rows=first,gates=gates,timing_measured=False)
    FOLDER.mkdir(exist_ok=True)
    (FOLDER/'field_rt_mesh_reference.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(FOLDER/'field_rt_mesh_reference_arrays.npz',**a)
    print(json.dumps(report)); return 0 if all(gates.values()) else 1

def measure_native():
    report=json.loads((FOLDER/'field_rt_mesh_reference.json').read_text())
    arrays=np.load(FOLDER/'field_rt_mesh_reference_arrays.npz')
    binary,program=os.environ['RT_WINDING_BINARY'],os.environ['RT_WINDING_PTX']
    rows,outputs=[],{}
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        for item in report['rows']:
            name=item['case']; corners=arrays[name+'_corners']; rays=arrays[name+'_rays']
            solid=arrays[name+'_solid']; distance=arrays[name+'_distance']; pitch=item['pitch']
            fixture=tmp/'input.bin'
            fixture.write_bytes(np.array([len(corners),len(rays)],dtype='<u4').tobytes()+corners.tobytes()+rays.tobytes())
            row=dict(case=name,legs=[])
            for leg in range(2):
                guard(); output=tmp/f'output{leg}.bin'
                result=subprocess.run([binary,program,str(fixture),str(output)],capture_output=True,timeout=30)
                guard()
                if result.returncode: raise RuntimeError(f'native process exit{result.returncode}; no retry')
                counts=np.fromfile(output,dtype='<i4').reshape(solid.shape); mask=counts!=0
                out=ndimage.distance_transform_edt(~mask,sampling=(pitch,)*3).astype(np.float32)
                inside=ndimage.distance_transform_edt(mask,sampling=(pitch,)*3).astype(np.float32)
                sd=(out-inside).astype(np.float32); sd=(sd-np.sign(sd)*np.float32(0.5*pitch)).astype(np.float32)
                row['legs'].append(dict(exit=0,counts_sha256=digest(counts),distance_sha256=digest(sd),
                                        occupancy_mismatches=int(np.count_nonzero(mask!=solid)),
                                        distance_mismatches=int(np.count_nonzero(sd!=distance)),
                                        max_distance_difference=float(np.max(np.abs(sd-distance)))))
                outputs[f'{name}_counts_{leg}']=counts; outputs[f'{name}_distance_{leg}']=sd
            rows.append(row)
    gates=dict(runtime_success=True,full_repeat=all(r['legs'][0]==r['legs'][1] for r in rows),
               exact_occupancy=all(q['occupancy_mismatches']==0 for r in rows for q in r['legs']),
               exact_distance=all(q['distance_mismatches']==0 for r in rows for q in r['legs']),
               no_observed_current_boot_fault=True)
    result=dict(rows=rows,gates=gates,timing_measured=False,binary_sha256=digest(np.fromfile(binary,dtype=np.uint8)),ptx_sha256=digest(np.fromfile(program,dtype=np.uint8)))
    (FOLDER/'field_rt_mesh_native_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(FOLDER/'field_rt_mesh_native_v1_arrays.npz',**outputs)
    print(json.dumps(result)); return 0 if all(gates.values()) else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--reference',action='store_true'); args=parser.parse_args()
    raise SystemExit(measure_reference() if args.reference else measure_native())
