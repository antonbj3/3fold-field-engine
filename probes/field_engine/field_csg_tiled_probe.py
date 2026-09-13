"""Direct leaf tile reads versus point-gather fusion, including thin axes."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import itertools
import json
from pathlib import Path
import time
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks
from field_csg_fused_v1 import evaluate as gathered
from field_csg_tiled_v1 import evaluate as tiled, gather_tile
from field_csg_fused_probe import PROGRAM, hashes, eager
from field_csg_probe import BOXES, box_distance

ROOT=Path(__file__).resolve().parents[2]


def main():
    origin=np.array([-2.,-2.,-4.]); pitch=.5; shape=(49,33,45)
    indices=np.indices(shape).reshape(3,-1).T
    fields=[AdaptiveSampleBlocks(box_distance(origin+indices*pitch,b).reshape(shape).astype(np.float32),origin,pitch) for b in BOXES]
    rows,outputs=[],{}
    for leg in range(2):
        results,times={},{}
        for name in (('gathered','tiled') if leg==0 else ('tiled','gathered')):
            begin=time.perf_counter_ns()
            results[name]=(gathered if name=='gathered' else tiled)(fields,PROGRAM)
            times[name]=(time.perf_counter_ns()-begin)*1e-6
        rows.append(dict(leg=leg,times_ms=times,speedup=times['gathered']/times['tiled'],
                         exact=hashes(results['gathered'])==hashes(results['tiled']),hashes=hashes(results['tiled'])))
        outputs[f'field_{leg}']=results['tiled'].samples_at(indices).reshape(shape)
    rng=np.random.default_rng(841)
    controls=[]
    # Thin, invariant, split and non-multiple tile dimensions; signed zeros.
    for shape in ((1,37,19),(23,1,19),(23,37,1),(19,35,21),(3,4,5)):
        raw=rng.normal(size=shape).astype(np.float32)
        raw.ravel()[::3]=np.float32(-0.)
        field=AdaptiveSampleBlocks(raw,[0,0,0],1)
        constant=AdaptiveSampleBlocks(np.full(shape,.25,dtype=np.float32),[0,0,0],1)
        rebuilt=np.empty(shape,dtype=np.float32)
        for lo in itertools.product(*(range(0,n,7) for n in shape)):
            hi=tuple(min(l+7,n) for l,n in zip(lo,shape))
            rebuilt[tuple(slice(l,h) for l,h in zip(lo,hi))]=gather_tile(field,lo,hi)
        controls.append(rebuilt.tobytes()==raw.tobytes())
        for program in ([('union',0,1)],[('difference',0,1)],
                        [('offset',0,.1),('offset',2,-.1)],
                        [('union',0,1),('difference',2,0),('offset',3,.25)]):
            expected=eager([field,constant],program)
            a,b=tiled([field,constant],program),tiled([field,constant],program)
            controls.append(hashes(a)==hashes(b)==hashes(expected))
    rejected=0
    for program in ([],[('union',0,99)],[('offset',0,float('nan'))],[('union',True,1)],
                    [('unknown',0,1)],[('offset',0,1j)],[('union',0,1,2)],
                    [('offset',0,1e39),('union',5,1)]):
        try:tiled(fields,program)
        except ValueError:rejected+=1
    shifted=AdaptiveSampleBlocks(np.ones((2,2,2),np.float32),[0,0,0],1)
    try:tiled([fields[0],shifted],[('union',0,1)])
    except ValueError:rejected+=1
    with np.load(ROOT/'reports/field_csg_arrays.npz',allow_pickle=False) as saved:
        frozen=all(a.tobytes()==saved['field_0'].tobytes() for a in outputs.values())
    gates=dict(exact=all(r['exact'] for r in rows),repeat=rows[0]['hashes']==rows[1]['hashes'],
               frozen_fixture=frozen,tile_arithmetic=len(controls)==25 and all(controls),
               invalid_controls=rejected==9,speedup_1_25=all(r['speedup']>=1.25 for r in rows))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,
                controls=controls,invalid_rejections=rejected,
                scope='Direct leaf box intersections and broadcasting in tiles of at most 8192 samples. Same packed inputs and final pack as point-gather variant. Two reversed-order CPU observations, no isolated throughput claim. Dense output remains transient.')
    (ROOT/'reports/field_csg_tiled.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_csg_tiled_arrays.npz',**outputs)
    print(json.dumps(report))
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
