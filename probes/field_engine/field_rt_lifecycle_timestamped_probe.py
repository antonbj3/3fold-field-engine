"""Repeat frozen lifecycle measurements with durable interval and lock timestamps."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import fcntl
from datetime import datetime, timezone
import numpy as np
from scipy import ndimage
import faltkarna_v1_mesh_to_sdf as baseline
from field_rt_mesh_probe import fixtures, digest
from field_rt_winding_probe import guard

ROOT=Path(__file__).resolve().parents[2]

EVENTS=[]
def event(label, case=None, leg=None):
    row=dict(label=label,case=case,leg=leg,utc=datetime.now(timezone.utc).isoformat(),monotonic_ns=time.monotonic_ns())
    EVENTS.append(row)
    with (ROOT/'artifacts'/'field_rt_lifecycle_timestamped_events.jsonl').open('a') as output:
        output.write(json.dumps(row)+'\n'); output.flush(); os.fsync(output.fileno())


def idle():
    # Only before a child context exists; never poll during a native launch.
    q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    values=[int(x.strip()) for x in q.stdout.splitlines() if x.strip()]
    if not values or max(values)>5: raise RuntimeError('idle gate: utilization exceeds5 percent')
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if q.stdout.strip(): raise RuntimeError('idle gate: other compute context present')
    return values

def edt(mask,h):
    out=ndimage.distance_transform_edt(~mask,sampling=(h,)*3).astype(np.float32)
    inside=ndimage.distance_transform_edt(mask,sampling=(h,)*3).astype(np.float32)
    sd=(out-inside).astype(np.float32)
    return (sd-np.sign(sd)*np.float32(0.5*h)).astype(np.float32)

def main():
    binary,ptx=os.environ['RT_WINDING_BINARY'],os.environ['RT_WINDING_PTX']
    meshes=fixtures(); rows=[]; outputs={}
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder)
        for leg in range(2):
            for name,mesh in meshes.items():
                guard(); utilization=idle()
                h=0.125; origin=np.asarray(mesh.bounds).min(axis=0)-0.5
                event('cpu_total_start',name,leg)
                t=time.perf_counter()
                gmin,shape,_,truth,reference=baseline.surface_raster_and_flood(mesh.vertices,mesh.faces,h,origin)
                cpu_total=(time.perf_counter()-t)*1000
                event('cpu_total_end',name,leg)
                anchor=origin+gmin*h
                event('cpu_sign_start',name,leg)
                t=time.perf_counter()
                mask,_=baseline.solid_via_stralvindning(mesh.vertices,mesh.faces,h,anchor,shape)
                cpu_sign=(time.perf_counter()-t)*1000
                event('cpu_sign_end',name,leg)
                event('cpu_edt_start',name,leg)
                t=time.perf_counter(); distance=edt(mask,h); cpu_edt=(time.perf_counter()-t)*1000
                event('cpu_edt_end',name,leg)
                assert np.array_equal(mask,truth) and np.array_equal(distance,reference)
                # Full native adapter interval includes grid setup, conversion, file I/O,
                # child startup/context/build/launch/teardown and CPU distance reconstruction.
                event('native_total_start',name,leg)
                start=time.perf_counter()
                ngmin,nshape=baseline._fonster(mesh.vertices,h,origin)
                nanchor=origin+ngmin*h
                points=nanchor+np.indices(nshape).reshape(3,-1).T*h
                points[:,0]+=h*baseline.JITTER_FRAC*baseline._GYLLENE[0]
                points[:,1]+=h*baseline.JITTER_FRAC*baseline._GYLLENE[1]
                corners=np.asarray(mesh.vertices[mesh.faces]-nanchor,dtype='<f4')
                rays=np.asarray(points-nanchor,dtype='<f4')
                infile=folder/'input.bin'; outfile=folder/'output.bin'
                infile.write_bytes(np.array([len(corners),len(rays)],dtype='<u4').tobytes()+corners.tobytes()+rays.tobytes())
                event('native_process_start',name,leg)
                launch=time.perf_counter()
                result=subprocess.run([binary,ptx,str(infile),str(outfile)],capture_output=True,timeout=30)
                process_ms=(time.perf_counter()-launch)*1000
                event('native_process_end',name,leg)
                if result.returncode: raise RuntimeError(f'native exit{result.returncode}; no retry')
                counts=np.fromfile(outfile,dtype='<i4').reshape(shape)
                candidate=counts!=0
                sd=edt(candidate,h)
                surface=candidate & ~ndimage.binary_erosion(candidate,ndimage.generate_binary_structure(3,1)) if candidate.any() else np.zeros_like(candidate)
                total=(time.perf_counter()-start)*1000
                event('native_total_end',name,leg)
                guard()
                row=dict(case=name,leg=leg,idle_utilization_percent=utilization,cpu_total_ms=cpu_total,
                         cpu_sign_ms=cpu_sign,cpu_edt_ms=cpu_edt,native_process_ms=process_ms,
                         native_adapter_total_ms=total,total_ratio=total/cpu_total,
                         occupancy_mismatches=int(np.count_nonzero(candidate!=truth)),
                         distance_mismatches=int(np.count_nonzero(sd!=reference)),
                         counts_sha256=digest(counts),distance_sha256=digest(sd),cpu_distance_sha256=digest(reference))
                rows.append(row); outputs[f'{name}_counts_{leg}']=counts; outputs[f'{name}_distance_{leg}']=sd
    gates=dict(full_repeat=all(rows[i]['counts_sha256']==rows[i+4]['counts_sha256'] and rows[i]['distance_sha256']==rows[i+4]['distance_sha256'] for i in range(4)),
               exact_occupancy=all(r['occupancy_mismatches']==0 for r in rows),exact_distance=all(r['distance_mismatches']==0 for r in rows),
               native_total_faster_both_legs=all(r['native_adapter_total_ms']<r['cpu_total_ms'] for r in rows),
               no_observed_fault=True,idle_prechecks_passed=True)
    report=dict(rows=rows,gates=gates,cache_policy='Existing compiler cache; no warmup or cache flush. Each native call starts a new process.',
                timing_repeat_identity_claim=False,binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),ptx_sha256=hashlib.sha256(Path(ptx).read_bytes()).hexdigest())
    out=ROOT/'artifacts'
    (out/'field_rt_lifecycle_timestamped_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'field_rt_lifecycle_timestamped_v1_arrays.npz',**outputs)
    print(json.dumps(report)); return 0 if all(gates.values()) else 1

def locked_main():
    destination=ROOT/'artifacts'/'field_rt_lifecycle_timestamped_events.jsonl'
    if destination.exists(): raise RuntimeError('refuse stale event log')
    coordination=Path(os.environ['RT_COORDINATION_FILE'])
    expected=os.environ['RT_COORDINATION_SHA256']
    if hashlib.sha256(coordination.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('coordination changed before acquisition; review again')
    with open('/tmp/gpu.lock','a') as lock:
        event('lock_attempt')
        deadline=time.monotonic()+5
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic()>=deadline: raise RuntimeError('shared lock unavailable')
                time.sleep(0.05)
        try:
            event('lock_acquired')
            if hashlib.sha256(coordination.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('coordination changed during acquisition; review again')
            result=main()
        finally:
            event('lock_release_start')
            fcntl.flock(lock,fcntl.LOCK_UN)
            event('lock_released')
    return result

if __name__=='__main__': raise SystemExit(locked_main())
