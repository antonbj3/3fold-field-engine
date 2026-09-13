"""Exact native/Python packing choices, queries, lifetime and paired timings."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import weakref
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks
from adaptive_pack_native_v1 import NativeAdaptiveBlocks
from field_csg_fused_probe import hashes
from field_csg_curved_probe import timed

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--library',type=Path);args=parser.parse_args()
    source=ROOT/'src/field_engine/adaptive_pack_native_v1/pack.cpp'
    library=args.library or Path('/tmp/field_adaptive_pack_native.so')
    if args.library is None:
        subprocess.run(['c++','-std=c++17','-O3','-fno-fast-math','-shared','-fPIC',str(source),'-o',str(library)],check=True,timeout=60)
    rows=[];captures={}; rng=np.random.default_rng(445)
    with np.load(ROOT/'reports/field_csg_curved_arrays.npz',allow_pickle=False) as data:
        cases=[(name,data[name]) for name in data.files if name.endswith('_0_field')]
    for name,raw in cases:
        points=rng.uniform(-1,np.array(raw.shape)+1,(4097,3))
        for leg in range(2):
            results,times={},{}
            for mode in (('python','native') if leg==0 else ('native','python')):
                results[mode],times[mode]=timed(lambda:AdaptiveSampleBlocks(raw,[0,0,0],1) if mode=='python' else NativeAdaptiveBlocks(raw,[0,0,0],1,library=library))
            a,b=results['python'],results['native']
            queries=b.query(points)
            rows.append(dict(case=name,leg=leg,times_ms=times,speedup=times['python']/times['native'],
                             exact=hashes(a)==hashes(b),query_exact=a.query(points).tobytes()==queries.tobytes(),hashes=hashes(b)))
            captures[f'{name}_{leg}']=queries
    controls=[]
    raw_cases=[rng.normal(size=(19,23,17)).astype(np.float32),np.ones((29,1,17),np.float32),
               np.ones((1,19,31),np.float32),np.ones((17,29,1),np.float32),np.zeros((17,19,23),np.float32)]
    raw_cases[-1].ravel()[::2]=np.float32(-0.)
    raw_cases.append(raw_cases[0][::2,::2,::-1])
    for raw in raw_cases:
        for leaf in (8,512,4096):
            a=AdaptiveSampleBlocks(raw,[.1,-2,3],.7,leaf_samples=leaf)
            b=NativeAdaptiveBlocks(raw,[.1,-2,3],.7,leaf_samples=leaf,library=library)
            c=NativeAdaptiveBlocks(raw,[.1,-2,3],.7,leaf_samples=leaf,library=library)
            controls.append(hashes(a)==hashes(b)==hashes(c))
    temporary=np.ones((8,9,10),np.float32); reference=weakref.ref(temporary)
    held=NativeAdaptiveBlocks(temporary,[0,0,0],1,library=library);del temporary
    lifetime=reference() is None
    bad=[(np.ones((2,2,2),np.float64),[0,0,0],1,512),
         (np.ones((2,2),np.float32),[0,0,0],1,512),
         (np.full((2,2,2),np.nan,np.float32),[0,0,0],1,512),
         (np.ones((2,2,2),np.float32),[0,0],1,512),
         (np.ones((2,2,2),np.float32),[0,0,0],0,512),
         (np.ones((2,2,2),np.float32),[0,0,0],True,512),
         (np.ones((2,2,2),np.float32),[0,0,0],1,True),
         (np.ones((2,2,2),np.float32),[0,0,0],1,4)]
    rejected=0
    for raw,origin,pitch,leaf in bad:
        try:NativeAdaptiveBlocks(raw,origin,pitch,leaf_samples=leaf,library=library)
        except ValueError:rejected+=1
    gates=dict(coverage=len(rows)==8,exact=all(r['exact'] and r['query_exact'] for r in rows),
               repeats=all(len({json.dumps(r['hashes']) for r in rows if r['case']==name})==1 for name,_ in cases),
               controls=len(controls)==18 and all(controls),lifetime=lifetime,invalid=rejected==8,
               speedup_2=all(r['speedup']>=2 for r in rows))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,controls=controls,invalid_rejections=rejected,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Same tree splitting, axis tie order, metadata layout and exact sample bits as Python. CPU packing only, compilation excluded, queries outside timing. No retained dense source; no peak-memory claim. Two reversed orders with ordinary concurrent CPU load.')
    (ROOT/'reports/field_native_pack.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_native_pack_arrays.npz',**captures)
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
