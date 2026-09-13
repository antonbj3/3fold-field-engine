"""Paired field/B-rep operation costs and curved-fixture refinement accuracy."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import trimesh
from build123d import Box, Cylinder, Sphere, Pos
from skimage.measure import marching_cubes
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks
from field_csg_tiled_v1 import evaluate
from field_csg_probe import BOXES, box_distance

ROOT=Path(__file__).resolve().parents[2]


def timed(call):
    start=time.perf_counter_ns()
    value=call()
    return value,(time.perf_counter_ns()-start)*1e-6


def distances(points, curved):
    if not curved:
        return [box_distance(points,b) for b in BOXES]
    sphere=np.linalg.norm(points-np.array([10,6,6]),axis=-1)-6
    plate=box_distance(points,((0,0,0),(20,12,2)))
    q=np.stack((np.linalg.norm(points[:,:2]-[10,6],axis=-1)-1.5,np.abs(points[:,2]-6)-10),axis=-1)
    cylinder=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(axis=-1),0)
    return [sphere,plate,cylinder]


def combine(values,curved):
    if curved:
        return np.maximum(np.minimum(values[0],values[1]),-values[2])
    a,b,c,h,j=values
    return np.maximum(np.minimum(np.minimum(a,b),c),-np.minimum(h,j))


def primitives(curved):
    if curved:
        return [Pos(10,6,6)*Sphere(6),Pos(10,6,1)*Box(20,12,2),Pos(10,6,6)*Cylinder(1.5,20)]
    return [Pos(*((np.array(lo)+np.array(hi))/2))*Box(*(np.array(hi)-np.array(lo))) for lo,hi in BOXES]


def boolean(parts,curved):
    if curved:
        return (parts[0]+parts[1])-parts[2]
    return (parts[0]+parts[1]+parts[2])-(parts[3]+parts[4])


def analytic_volume(curved):
    if not curved:
        return 960.
    radius,h,hole=6.,2.,1.5
    sphere=4*np.pi*radius**3/3
    cap=np.pi*h*h*(radius-h/3)
    removed=np.pi*hole**2*radius+2*np.pi/3*(radius**3-(radius**2-hole**2)**1.5)
    return sphere+480-cap-removed


def main():
    rows,arrays=[],{}
    for curved in (False,True):
        name='sphere_plate_round_hole' if curved else 'bracket_plate_rectangular_holes'
        program=[('union',0,1),('difference',3,2)] if curved else [('union',0,1),('union',5,2),('union',3,4),('difference',6,7)]
        for pitch in (.5,.25):
            origin=np.array([-2.,-2.,-4.]); shape=tuple(np.rint(np.array([24,16,22])/pitch).astype(int)+1)
            indices=np.indices(shape).reshape(3,-1).T
            def prepare_fields():
                samples=[a.reshape(shape).astype(np.float32) for a in distances(origin+indices*pitch,curved)]
                return samples,[AdaptiveSampleBlocks(a,origin,pitch) for a in samples]
            (samples,fields),field_prepare_ms=timed(prepare_fields)
            expected=combine(samples,curved)
            parts,brep_prepare_ms=timed(lambda:primitives(curved))
            # Bound using analytic center distances and half the cell diagonal.
            cell_shape=tuple(s-1 for s in shape)
            lower_count=upper_count=0
            for start in range(0,int(np.prod(cell_shape)),32768):
                idx=np.array(np.unravel_index(np.arange(start,min(start+32768,int(np.prod(cell_shape)))),cell_shape)).T
                values=combine(distances(origin+(idx+.5)*pitch,curved),curved)
                radius=np.sqrt(3)*pitch/2
                lower_count+=int(np.count_nonzero(values < -radius))
                upper_count+=int(np.count_nonzero(values <= radius))
            lower,upper=lower_count*pitch**3,upper_count*pitch**3
            for leg in range(2):
                results,times={},{}
                for backend in (('field','brep') if leg==0 else ('brep','field')):
                    results[backend],times[backend]=timed(lambda:evaluate(fields,program) if backend=='field' else boolean(parts,curved))
                field=results['field']; brep=results['brep']
                def extract():
                    dense=field.samples_at(indices).reshape(shape)
                    v,f,_,_=marching_cubes(dense,level=0,spacing=(pitch,)*3)
                    return dense,trimesh.Trimesh(v+origin,f,process=False)
                (dense,mesh),extraction_ms=timed(extract)
                digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                volume=abs(mesh.volume); exact_volume=analytic_volume(curved)
                # Removing the hole operation must be detectable in the sample oracle.
                missing_holes=np.minimum(samples[0],samples[1]) if curved else np.minimum(np.minimum(samples[0],samples[1]),samples[2])
                missing_count=int(np.count_nonzero((missing_holes<0)!=(dense<0)))
                row=dict(fixture=name,pitch=pitch,leg=leg,operation_ms=times,
                         field_prepare_ms=field_prepare_ms,brep_prepare_ms=brep_prepare_ms,
                         field_unpack_extract_ms=extraction_ms,retained_bytes=field.storage_bytes,
                         mesh_volume=volume,brep_volume=brep.volume,analytic_volume=exact_volume,
                         relative_volume_error=abs(volume-exact_volume)/exact_volume,
                         volume_envelope=[lower,upper],watertight=bool(mesh.is_watertight),brep_valid=bool(brep.is_valid),
                         exact=dense.tobytes()==expected.tobytes(),missing_hole_changed_samples=missing_count,
                         hashes=[digest(a) for a in (dense,field.nodes,field.payload,mesh.vertices,mesh.faces)])
                rows.append(row)
                for key,a in (('field',dense),('vertices',mesh.vertices),('faces',mesh.faces)):
                    arrays[f'{name}_{pitch}_{leg}_{key}']=a
    gates=dict(coverage=len(rows)==8,exact=all(r['exact'] for r in rows),
               repeats=all(len({json.dumps(r['hashes']) for r in rows if r['fixture']==name and r['pitch']==p})==1 for name in {r['fixture'] for r in rows} for p in (.5,.25)),
               watertight=all(r['watertight'] and r['brep_valid'] for r in rows),
               analytic_brep=all(abs(r['brep_volume']-r['analytic_volume'])<1e-8 for r in rows),
               volume_1percent=all(r['relative_volume_error']<=.01 for r in rows),
               envelope=all(r['volume_envelope'][0]<=r['mesh_volume']<=r['volume_envelope'][1] and r['volume_envelope'][0]<=r['brep_volume']<=r['volume_envelope'][1] for r in rows),
               refinement=all(max(r['relative_volume_error'] for r in rows if r['fixture']==name and r['pitch']==.25)<min(r['relative_volume_error'] for r in rows if r['fixture']==name and r['pitch']==.5) for name in {r['fixture'] for r in rows}),
               missing_hole_control=all(r['missing_hole_changed_samples']>0 for r in rows))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,
                scope='Two synthetic fixtures and two pitches, two reversed CPU orders. Operations start from prepared operands on both paths. Field sampling/packing and B-rep primitive preparation reported separately. Field extraction includes unpacking; B-rep tessellation is excluded. Outputs differ in representation, no universal speed claim or isolated CPU timing.')
    (ROOT/'reports/field_csg_curved.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_csg_curved_arrays.npz',**arrays)
    print(json.dumps(report))
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
