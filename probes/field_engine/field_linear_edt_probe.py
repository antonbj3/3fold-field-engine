"""Exact integer envelope versus exhaustive native EDT and independent SciPy."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy import ndimage
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline
from native_edt_large_v1 import NativeEDTLarge

ROOT=Path(__file__).resolve().parents[2]


def timed(call):
    start=time.perf_counter_ns();value=call();return value,(time.perf_counter_ns()-start)*1e-6


def main():
    old=NativeEDTLarge(os.environ['NATIVE_EDT_LARGE_LIBRARY'])
    new=NativeEDTLarge(os.environ['NATIVE_EDT_LINEAR_LIBRARY'])
    rng=np.random.default_rng(441)
    cases=[]
    for shape in ((1,1,2),(512,1,7),(1,512,7),(7,1,512),(31,27,19),(97,65,89)):
        mask=rng.random(shape)>.4;mask.flat[0]=False;mask.flat[-1]=True
        cases.append(('random_'+str(shape),mask))
    singleton=np.ones((37,41,43),bool);singleton[18,20,21]=False
    cases.extend([('single_zero',singleton),('single_one',~singleton)])
    mesh=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);mesh.merge_vertices()
    vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces)
    plate=baseline.surface_raster_and_flood(vertices,faces,.5,vertices.min(0)-1.5)[3]
    cases.append(('plate_L',plate))
    rows,arrays=[],{}
    for name,mask in cases:
        expected=[np.rint(ndimage.distance_transform_edt(b)**2).astype(np.int32) for b in (mask,~mask)]
        for leg in range(2):
            outputs,times={},{}
            for backend in (('exhaustive','envelope') if leg==0 else ('envelope','exhaustive')):
                handle=old if backend=='exhaustive' else new
                outputs[backend],times[backend]=timed(lambda:[handle.squared_distance(b) for b in (mask,~mask)])
            pair=outputs['envelope']
            exact=all(a.tobytes()==b.tobytes()==c.tobytes() for a,b,c in zip(pair,outputs['exhaustive'],expected))
            signed_exact=[]
            for pitch in (.125,.7,2.):
                a=new.signed_distance(mask,pitch);b=old.signed_distance(mask,pitch)
                signed_exact.append(a.tobytes()==b.tobytes())
                arrays[f'{name}_{leg}_signed_{pitch}']=a
            rows.append(dict(case=name,shape=mask.shape,leg=leg,pair_ms=times,exact=exact,signed_exact=all(signed_exact),
                             hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in pair]))
            for i,a in enumerate(pair):arrays[f'{name}_{leg}_squared_{i}']=a
    rejected=0
    for mask in (np.ones((2,2,2),bool),np.zeros((2,2,2),bool),np.ones((2,2,2),np.uint8),np.zeros((513,1,2),bool)):
        try:new.squared_distance(mask)
        except ValueError:rejected+=1
    large=[r for r in rows if r['case']=='plate_L']
    gates=dict(coverage=len(rows)==18,exact=all(r['exact'] and r['signed_exact'] for r in rows),
               repeat=all(len({json.dumps(r['hashes']) for r in rows if r['case']==name})==1 for name,_ in cases),
               invalid=rejected==4,plate_speedup_2=all(r['pair_ms']['exhaustive']/r['pair_ms']['envelope']>=2 for r in large))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,invalid_rejections=rejected,
                source_sha256=hashlib.sha256((ROOT/'src/field_engine/edt_native_linear_v1/edt.cu').read_bytes()).hexdigest(),
                scope='Exact integer squared EDT; independent SciPy squared values plus frozen native reference. Two reversed orders. Pair time includes host validation/allocation/upload/three axes/download/free for mask and complement; first runtime initialization charged where it occurs. Signed host arithmetic unchanged. Modal device metadata stored separately.')
    (ROOT/'reports/field_linear_edt.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_linear_edt_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
