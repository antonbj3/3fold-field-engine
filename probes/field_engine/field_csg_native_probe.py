"""Native CPU tile copy gates and paired full CSG at two resolutions."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks
from csg_tile_native_v1 import NativeTileReader
from field_csg_tiled_v1 import evaluate as python_evaluate
from field_csg_native_v1 import evaluate as native_evaluate
from field_csg_curved_probe import distances,timed
from field_csg_fused_probe import hashes

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--source',type=Path,default=ROOT/'src/field_engine/csg_tile_native_v1/gather.cpp')
    parser.add_argument('--output-prefix',type=Path,default=ROOT/'reports/field_csg_native')
    args=parser.parse_args()
    source=args.source.resolve()
    library=args.library or Path('/tmp/field_csg_tile_native.so')
    if args.library is None:
        subprocess.run(['c++','-std=c++17','-O3','-fno-fast-math','-shared','-fPIC',str(source),'-o',str(library)],check=True,timeout=60)
    reader=NativeTileReader(library)
    rows,arrays=[],{}
    for curved in (False,True):
        program=[('union',0,1),('difference',3,2)] if curved else [('union',0,1),('union',5,2),('union',3,4),('difference',6,7)]
        for pitch in (.5,.25):
            origin=np.array([-2.,-2.,-4.]);shape=tuple(np.rint(np.array([24,16,22])/pitch).astype(int)+1)
            indices=np.indices(shape).reshape(3,-1).T
            fields=[AdaptiveSampleBlocks(a.reshape(shape).astype(np.float32),origin,pitch) for a in distances(origin+indices*pitch,curved)]
            for leg in range(2):
                results,times={},{}
                for name in (('python','native') if leg==0 else ('native','python')):
                    results[name],times[name]=timed(lambda:python_evaluate(fields,program) if name=='python' else native_evaluate(fields,program,reader=reader))
                rows.append(dict(curved=curved,pitch=pitch,leg=leg,times_ms=times,speedup=times['python']/times['native'],
                                 exact=hashes(results['python'])==hashes(results['native']),hashes=hashes(results['native'])))
                arrays[f'{curved}_{pitch}_{leg}']=results['native'].samples_at(indices).reshape(shape)
    controls=[]; rng=np.random.default_rng(77)
    for shape in ((1,17,19),(19,1,17),(17,19,1),(19,21,17)):
        raw=rng.normal(size=shape).astype(np.float32); raw.ravel()[::2]=np.float32(-0.)
        fields=[AdaptiveSampleBlocks(raw,[0,0,0],1),AdaptiveSampleBlocks(np.full(shape,.25,np.float32),[0,0,0],1)]
        controls.append(reader.gather(fields[0],[0,0,0],shape).tobytes()==raw.tobytes())
        for program in ([('union',0,1)],[('difference',0,1)],[('offset',0,.1),('offset',2,-.1)]):
            a=python_evaluate(fields,program)
            b=native_evaluate(fields,program,reader=reader)
            c=native_evaluate(fields,program,reader=reader)
            controls.append(hashes(a)==hashes(b)==hashes(c))
    valid=AdaptiveSampleBlocks(np.ones((4,4,4),np.float32),[0,0,0],1)
    invalid=[]
    for lower,upper in (([-1,0,0],[1,1,1]),([0,0,0],[5,4,4]),([0.,0,0],[1,1,1]),([1,1,1],[1,2,2])):
        try:reader.gather(valid,lower,upper)
        except ValueError:invalid.append(True)
        else:invalid.append(False)
    for mode in ('offset','shape','overlap','gap'):
        broken=copy.copy(valid);broken.nodes=valid.nodes.copy()
        if mode=='offset':broken.nodes['offset'][0]=1000000
        if mode=='shape':broken.nodes['stored_shape'][0]=[5,4,4]
        if mode=='overlap':broken.nodes=np.concatenate((broken.nodes,broken.nodes))
        if mode=='gap':broken.nodes['shape'][0]=[1,1,1]
        try:reader.gather(broken,[0,0,0],[4,4,4])
        except ValueError:invalid.append(True)
        else:invalid.append(False)
    with np.load(ROOT/'reports/field_csg_curved_arrays.npz',allow_pickle=False) as prior:
        frozen=all(arrays[f'{c}_{p}_{l}'].tobytes()==prior[f'{"sphere_plate_round_hole" if c else "bracket_plate_rectangular_holes"}_{p}_{l}_field'].tobytes() for c in (False,True) for p in (.5,.25) for l in range(2))
    gates=dict(coverage=len(rows)==8,exact=all(r['exact'] for r in rows),
               repeats=all(len({json.dumps(r['hashes']) for r in rows if r['curved']==c and r['pitch']==p})==1 for c in (False,True) for p in (.5,.25)),
               frozen_fixture=frozen,tile_controls=len(controls)==16 and all(controls),invalid_controls=len(invalid)==8 and all(invalid),
               speedup_1_25=all(r['speedup']>=1.25 for r in rows))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,controls=controls,invalid=invalid,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope='Native CPU bit copies only, unchanged NumPy CSG and final Python packing. Eight reversed-order observations with prepared inputs, no compilation/preparation/extraction in timing. Ordinary concurrent CPU load. Full saved field equality transfers prior fixture accuracy only.')
    args.output_prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output_prefix.parent/(args.output_prefix.name+'_arrays.npz'),**arrays)
    print(json.dumps(report))
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
