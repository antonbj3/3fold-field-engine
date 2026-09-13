"""Paired bracket/plate/holes CSG against dense arithmetic and exact B-rep."""

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
from skimage.measure import marching_cubes
from build123d import Box, Pos
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks
from field_csg_blocks_v1 import compose

ROOT = Path(__file__).resolve().parents[2]
# Three solids and two rectangular through holes; common units are arbitrary.
BOXES = [((0,0,0),(20,6,4)), ((0,0,0),(4,6,16)),
         ((0,0,0),(20,12,2)), ((5,3,-2),(8,5,6)), ((12,3,-2),(15,5,6))]


def box_distance(points, bounds):
    lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    q = np.abs(points-(lo+hi)/2)-(hi-lo)/2
    return np.linalg.norm(np.maximum(q,0), axis=-1)+np.minimum(q.max(axis=-1),0)


def exact_field(points):
    a,b,c,h,j = [box_distance(points, bounds) for bounds in BOXES]
    return np.maximum(np.minimum(np.minimum(a,b),c),-np.minimum(h,j))


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns()-start)*1e-6


def main():
    pitch = .5
    origin = np.array([-2.,-2.,-4.])
    shape = (49,33,45)
    indices = np.indices(shape).reshape(3,-1).T
    points = origin+indices*pitch
    samples = [box_distance(points,b).reshape(shape).astype(np.float32) for b in BOXES]
    fields, input_pack_ms = timed(lambda: [AdaptiveSampleBlocks(a,origin,pitch) for a in samples])
    expected = np.maximum(np.minimum(np.minimum(samples[0],samples[1]),samples[2]),-np.minimum(samples[3],samples[4]))
    # Cell uncertainty envelope from the analytic 1-Lipschitz level set.
    cell_shape = tuple(s-1 for s in shape)
    centers = origin+(np.indices(cell_shape).reshape(3,-1).T+.5)*pitch
    center_values = exact_field(centers)
    radius = np.sqrt(3)*pitch/2
    lower = int(np.count_nonzero(center_values < -radius))*pitch**3
    upper = int(np.count_nonzero(center_values <= radius))*pitch**3
    rows, captures = [], {}
    for leg in range(2):
        def field_chain():
            bracket = compose(fields[0],fields[1])
            union = compose(bracket,fields[2])
            holes = compose(fields[3],fields[4])
            return compose(union,holes,operation='difference')
        result, field_ms = timed(field_chain)
        dense = result.samples_at(indices).reshape(shape)
        expanded = compose(result, operation='offset', distance=.25)
        offset = expanded.samples_at(indices).reshape(shape)
        def brep_chain():
            parts = [Pos(*((np.array(lo)+np.array(hi))/2))*Box(*(np.array(hi)-np.array(lo))) for lo,hi in BOXES]
            return (parts[0]+parts[1]+parts[2])-(parts[3]+parts[4])
        brep, brep_ms = timed(brep_chain)
        def extract():
            v,f,_,_ = marching_cubes(dense,level=0,spacing=(pitch,)*3)
            return trimesh.Trimesh(v+origin,f,process=False)
        mesh, extract_ms = timed(extract)
        volume = abs(mesh.volume)
        digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        rows.append(dict(leg=leg,field_csg_ms=field_ms,brep_construct_boolean_ms=brep_ms,
                         extraction_ms=extract_ms,field_volume=volume,brep_volume=brep.volume,
                         volume_error=abs(volume-brep.volume),volume_bound=upper-lower,
                         volume_envelope=[lower,upper],watertight=bool(mesh.is_watertight),
                         brep_valid=bool(brep.is_valid),exact=dense.tobytes()==expected.tobytes(),
                         offset_exact=offset.tobytes()==(expected.astype(np.float64)-.25).astype(np.float32).tobytes(),
                         retained_bytes=result.storage_bytes,hashes=[digest(a) for a in (dense,offset,result.nodes,result.payload,mesh.vertices,mesh.faces)]))
        captures[f'field_{leg}']=dense
        captures[f'offset_{leg}']=offset
        captures[f'vertices_{leg}']=mesh.vertices
        captures[f'faces_{leg}']=mesh.faces
    rejected = 0
    for call in (lambda: compose(fields[0],AdaptiveSampleBlocks(samples[1],origin+1,pitch)),
                 lambda: compose(fields[0],operation='missing'),
                 lambda: compose(fields[0],operation='difference'),
                 lambda: compose(fields[0],fields[1],operation='offset'),
                 lambda: compose(fields[0],operation='offset',distance=float('nan'))):
        try: call()
        except ValueError: rejected += 1
    gates = dict(exact=all(r['exact'] and r['offset_exact'] for r in rows),
                 repeat=rows[0]['hashes']==rows[1]['hashes'],
                 watertight=all(r['watertight'] and r['brep_valid'] for r in rows),
                 volume=all(lower<=r['field_volume']<=upper and lower<=r['brep_volume']<=upper and r['volume_error']<=r['volume_bound'] for r in rows),
                 rejects=rejected==5)
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,
                input_pack_ms=input_pack_ms,shape=shape,pitch=pitch,
                scope='Synthetic axis-aligned bracket/plate with two rectangular through holes. CSG starts from packed operands; B-rep includes primitive construction, excludes tessellation/rasterization. Input packing and mesh extraction reported separately. Offset verified as level-set arithmetic only, not exact Euclidean redistance.')
    (ROOT/'reports/field_csg.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_csg_arrays.npz',**captures)
    print(json.dumps(report))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
