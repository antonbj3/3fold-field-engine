"""Persistent paired GPU EDT against exact squared and signed CPU references."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import threading
import numpy as np
from scipy import ndimage
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline
from native_edt_large_v1 import NativeEDTLarge
from edt_persistent_pair_v1 import PersistentPairEDT
from field_linear_edt_probe import timed

ROOT=Path(__file__).resolve().parents[2]


def main():
    old=NativeEDTLarge(os.environ['NATIVE_EDT_LARGE_LIBRARY']);library=os.environ['NATIVE_EDT_QUOTIENT_LIBRARY']
    rng=np.random.default_rng(441);cases=[]
    for shape in ((1,1,2),(512,1,7),(1,512,7),(7,1,512),(31,27,19),(97,65,89)):
        mask=rng.random(shape)>.4;mask.flat[0]=False;mask.flat[-1]=True;cases.append(('random_'+str(shape),mask))
    singleton=np.ones((37,41,43),bool);singleton[18,20,21]=False;cases.extend([('single_zero',singleton),('single_one',~singleton)])
    mesh=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);mesh.merge_vertices()
    vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces)
    cases.append(('plate_L',baseline.surface_raster_and_flood(vertices,faces,.5,vertices.min(0)-1.5)[3]))
    rows,arrays,controls=[],{},[]
    for name,mask in cases:
        squared=np.array([np.rint(ndimage.distance_transform_edt(b)**2).astype(np.int32) for b in (~mask,mask)])
        for leg in range(2):
            reference,reference_setup_ms=timed(lambda:PersistentPairEDT(mask.shape,library=os.environ['NATIVE_EDT_PAIR_LIBRARY']))
            handle=None
            try:
                handle,setup_ms=timed(lambda:PersistentPairEDT(mask.shape,library=library))
                reference.query(mask,.5)
                actual,first_ms=timed(lambda:handle.query(mask,.5))
                times={};outputs={}
                for mode in (('reference','candidate') if leg==0 else ('candidate','reference')):
                    outputs[mode],times[mode]=timed(lambda:reference.query(mask,.5) if mode=='reference' else handle.query(mask,.5))
                exact=[]
                for pitch in (1e-12,.125,.7,2.,1e12):
                    signed,pair=handle.query(mask,pitch,with_squared=True)
                    expected=old.signed_distance(mask,pitch)
                    cpu_pair=[(np.sqrt(x.astype(np.float64))*pitch).astype(np.float32) for x in squared]
                    cpu_delta=(cpu_pair[0]-cpu_pair[1]).astype(np.float32)
                    cpu=(cpu_delta-np.sign(cpu_delta)*np.float32(.5*pitch)).astype(np.float32)
                    exact.append(pair.tobytes()==squared.tobytes() and signed.tobytes()==expected.tobytes()==cpu.tobytes())
                    arrays[f'{name}_{leg}_{pitch}']=signed
                    arrays[f'{name}_{leg}_{pitch}_squared']=pair
                changed=handle.query(~mask,.5)
                controls.append(changed.tobytes()==old.signed_distance(~mask,.5).tobytes())
                def wrong_thread():
                    try:handle.query(mask,.5)
                    except RuntimeError:controls.append(True)
                    else:controls.append(False)
                thread=threading.Thread(target=wrong_thread);thread.start();thread.join()
                rows.append(dict(case=name,leg=leg,shape=mask.shape,setup_ms=setup_ms,reference_setup_ms=reference_setup_ms,first_query_ms=first_ms,warm_ms=times,
                                 exact=all(exact) and actual.tobytes()==outputs['reference'].tobytes()==outputs['candidate'].tobytes(),
                                 hash=hashlib.sha256(actual.tobytes()).hexdigest()))
            finally:
                if handle is not None:handle.close()
                reference.close()
            handle.close()
            try:handle.query(mask,.5)
            except RuntimeError:controls.append(True)
            else:controls.append(False)
    gates=dict(coverage=len(rows)==18,exact=all(r['exact'] for r in rows),
               repeat=all(len({r['hash'] for r in rows if r['case']==name})==1 for name,_ in cases),
               lifecycle=len(controls)==54 and all(controls),
               plate_speedup_1_25=all(r['warm_ms']['reference']/r['warm_ms']['candidate']>=1.25 for r in rows if r['case']=='plate_L'))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,controls=controls,
                source_sha256=hashlib.sha256((ROOT/'src/field_engine/edt_pair_quotient_v1/edt.cu').read_bytes()).hexdigest(),
                scope='Paired frozen persistent EDT versus exact-corrected quotient EDT; both handles warmed, opposite timing orders. Warm calls include mask validation/upload/full signed-field download. Setup and first query separate. Five pitches include contract endpoints. Full masks recomputed after change, thread/close controls. No full mesh-stage or 2 ms service claim.')
    (ROOT/'reports/field_quotient_pair.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_quotient_pair_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
