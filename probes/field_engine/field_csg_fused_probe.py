"""Paired eager/fused CSG with exact full representations and fault controls."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks
from field_csg_blocks_v1 import compose
from field_csg_fused_v1 import evaluate
from field_csg_probe import BOXES, box_distance

ROOT = Path(__file__).resolve().parents[2]
PROGRAM = [('union',0,1),('union',5,2),('union',3,4),('difference',6,7)]


def eager(fields, program):
    values = list(fields)
    for op,a,b in program:
        values.append(compose(values[a],operation='offset',distance=b) if op=='offset' else compose(values[a],values[b],operation=op))
    return values[-1]


def hashes(field):
    return [hashlib.sha256(a.tobytes()).hexdigest() for a in (field.nodes,field.payload,field.origin)]


def main():
    origin = np.array([-2.,-2.,-4.]); pitch = .5; shape = (49,33,45)
    indices = np.indices(shape).reshape(3,-1).T
    samples = [box_distance(origin+indices*pitch,b).reshape(shape).astype(np.float32) for b in BOXES]
    fields = [AdaptiveSampleBlocks(a,origin,pitch) for a in samples]
    rows, outputs = [], {}
    for leg in range(2):
        results, times = {}, {}
        for name in (('eager','fused') if leg==0 else ('fused','eager')):
            start = time.perf_counter_ns()
            results[name] = (eager if name=='eager' else evaluate)(fields,PROGRAM)
            times[name] = (time.perf_counter_ns()-start)*1e-6
        full = {name:f.samples_at(indices).reshape(shape) for name,f in results.items()}
        rows.append(dict(leg=leg,times_ms=times,speedup=times['eager']/times['fused'],
                         exact=full['eager'].tobytes()==full['fused'].tobytes(),
                         packed_exact=hashes(results['eager'])==hashes(results['fused']),
                         hashes=hashes(results['fused'])))
        outputs[f'field_{leg}']=full['fused']
    # Signed zeros, random curved data, shared operands, intermediate offsets,
    # cancellation with two rounded offsets, and a chunk boundary at sample 8192.
    rng = np.random.default_rng(912)
    shape2 = (23,21,19)
    raw = [rng.normal(size=shape2).astype(np.float32) for _ in range(3)]
    raw[0].ravel()[::2]=np.float32(-0.)
    raw[1].ravel()[::2]=np.float32(0.)
    controls = [AdaptiveSampleBlocks(a,[0,0,0],1) for a in raw]
    programs = [[('union',0,1)], [('difference',0,1)],
                [('offset',0,.1),('offset',3,-.1)],
                [('union',0,1),('difference',3,2),('offset',4,.37),('union',5,3)],
                [('offset',0,1e20),('offset',3,-1e20)]]
    control_passes=[]
    for program in programs:
        expected = eager(controls,program)
        first,second = evaluate(controls,program),evaluate(controls,program)
        control_passes.append(hashes(first)==hashes(second)==hashes(expected))
    bad_programs = [[],[('union',0,99)],[('offset',0,float('nan'))],[('union',True,1)],
                    [('unknown',0,1)],[('offset',0,1j)],[('union',0,1,2)],
                    [('offset',0,1e39),('union',3,1)]]
    rejected=0
    for program in bad_programs:
        try: evaluate(controls,program)
        except ValueError: rejected+=1
    shifted = AdaptiveSampleBlocks(raw[0],[1,0,0],1)
    try: evaluate([controls[0],shifted],[('union',0,1)])
    except ValueError: rejected+=1
    saved = np.load(ROOT/'reports/field_csg_arrays.npz',allow_pickle=False)
    gates=dict(exact=all(r['exact'] and r['packed_exact'] for r in rows),
               repeats=rows[0]['hashes']==rows[1]['hashes'],
               frozen_fixture=all(a.tobytes()==saved['field_0'].tobytes() for a in outputs.values()),
               arithmetic_controls=all(control_passes),invalid_controls=rejected==9,
               speedup_1_25=all(r['speedup']>=1.25 for r in rows))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,
                arithmetic_controls=control_passes,invalid_rejections=rejected,
                scope='Same packed input fields, two reversed-order CPU comparisons. One final pack vs four eager packs; no input packing or extraction timed. Prior fixture mesh/volume evidence applies only through complete saved field equality. Ordinary concurrent desktop load, no isolated timing claim.')
    (ROOT/'reports/field_csg_fused.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_csg_fused_arrays.npz',**outputs)
    print(json.dumps(report))
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
