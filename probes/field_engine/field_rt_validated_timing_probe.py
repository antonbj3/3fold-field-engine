"""Measure fixed64 RT plus native EDT composition with runtime setup charged."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import ctypes as ct
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from field_rt_lifecycle_probe import edt,idle
from field_rt_winding_probe import guard
from field_rt_mesh_probe import digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts'

def event(label):
    with (OUT/'field_rt_validated_timing_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label,utc=datetime.now(timezone.utc).isoformat(),monotonic_ns=time.monotonic_ns()))+'\n');f.flush();os.fsync(f.fileno())

def pointer(a):return ct.c_void_p(a.ctypes.data)

def run():
    from rt_winding_handle_v1 import WindingHandle
    from field_rt_mesh_probe import fixtures
    from scipy import ndimage
    import faltkarna_v1_mesh_to_sdf as baseline
    data=np.load(OUT/'field_rt_changed_query_reference_arrays.npz')
    refs=json.loads((OUT/'field_rt_changed_query_reference.json').read_text())['rows']
    meshes=fixtures();rows=[];arrays={};repeat_ok=True;exact_ok=True
    guard();idle()
    event('adapter_setup_start');start=time.perf_counter()
    from native_edt_v1 import NativeEDT
    distance_owner=NativeEDT(os.environ['EDT_NATIVE_LIBRARY'])
    parallel_edt=distance_owner.signed_distance
    runtime_setup=(time.perf_counter()-start)*1000;event('adapter_setup_end')
    for leg in range(2):
        for ref in refs[::3]:
            name=ref['case'];mesh=meshes[name];shape=tuple(ref['shape']);h=0.125
            anchor=np.asarray(mesh.bounds).min(axis=0)-0.5
            rays=[data[f'{name}_{phase}_rays'] for phase in range(3)]
            offsets=[np.zeros(3),h*np.array([0.25,0.5,0.75]),h*np.array([-0.5,0.25,-0.25])]
            event(f'setup_start_{name}_{leg}');start=time.perf_counter()
            owner=WindingHandle(os.environ['RT_WINDING_LIBRARY'],os.environ['RT_WINDING_PTX'],data[name+'_corners'],ref['rays'])
            setup=(time.perf_counter()-start)*1000;event(f'setup_end_{name}_{leg}')
            runtime_charge=runtime_setup if leg==0 and not rows else 0.0
            cpu_times=[];query_times=[];edt_times=[];hashes=[]
            try:
                for call in range(64):
                    phase=call%3
                    event(f'cpu_start_{name}_{leg}_{call}');start=time.perf_counter()
                    mask,_=baseline.solid_via_stralvindning(mesh.vertices,mesh.faces,h,anchor+offsets[phase],shape)
                    cpu_sd=edt(mask,h)
                    cpu_surface=mask & ~ndimage.binary_erosion(mask,ndimage.generate_binary_structure(3,1))
                    cpu_times.append((time.perf_counter()-start)*1000);event(f'cpu_end_{name}_{leg}_{call}')
                    event(f'query_start_{name}_{leg}_{call}');start=time.perf_counter()
                    counts=owner.query(rays[phase]).reshape(shape)
                    query_times.append((time.perf_counter()-start)*1000);event(f'query_end_{name}_{leg}_{call}')
                    event(f'edt_start_{name}_{leg}_{call}');start=time.perf_counter()
                    inside=counts!=0;sd=parallel_edt(inside,h)
                    surface=inside & ~ndimage.binary_erosion(inside,ndimage.generate_binary_structure(3,1))
                    edt_times.append((time.perf_counter()-start)*1000);event(f'edt_end_{name}_{leg}_{call}')
                    exact_ok &= np.array_equal(inside,mask) and np.array_equal(sd,cpu_sd) and np.array_equal(sd,data[f'{name}_{phase}_distance']) and np.array_equal(surface,cpu_surface)
                    key=f'{name}_{leg}_{phase}'
                    if key+'_counts' in arrays:repeat_ok &= np.array_equal(arrays[key+'_counts'],counts) and np.array_equal(arrays[key+'_distance'],sd)
                    else:arrays[key+'_counts']=counts.copy();arrays[key+'_distance']=sd.copy()
                    hashes.append(dict(phase=phase,counts=digest(counts),distance=digest(sd)))
            finally:
                event(f'close_start_{name}_{leg}');start=time.perf_counter();owner.close()
                close=(time.perf_counter()-start)*1000;event(f'close_end_{name}_{leg}');guard()
            cpu=float(np.mean(cpu_times));query=float(np.mean(query_times));distance=float(np.mean(edt_times))
            accounted=(runtime_charge+setup+close+sum(query_times)+sum(edt_times))/64
            rows.append(dict(case=name,leg=leg,queries=64,runtime_setup_charge_ms=runtime_charge,setup_ms=setup,close_ms=close,cpu_mean_ms=cpu,wrapper_query_mean_ms=query,edt_surface_mean_ms=distance,
                             accounted_native_mean_ms=accounted,ratio=accounted/cpu,query_hashes=hashes,cpu_samples_ms=cpu_times,query_samples_ms=query_times,edt_samples_ms=edt_times))
    gates=dict(full_repeat=bool(repeat_ok) and all(rows[i]['query_hashes']==rows[i+4]['query_hashes'] for i in range(4)),exact_cpu_outputs=bool(exact_ok),
               faster_all_cases_both_legs=all(r['ratio']<1 for r in rows),finite_timings=all(np.isfinite(r[k]) and r[k]>=0 for r in rows for k in ('setup_ms','close_ms','cpu_mean_ms','wrapper_query_mean_ms','edt_surface_mean_ms')),
               no_observed_fault=True,idle_precheck=True)
    report=dict(rows=rows,gates=gates,native_edt_sha256=hashlib.sha256(Path(os.environ['EDT_NATIVE_LIBRARY']).read_bytes()).hexdigest(),timing_repeat_identity_claim=False,runtime_setup_ms=runtime_setup,scope='Prepared fixed mesh and three ray grids; accounted timings exclude audit/hash I/O and offline fixture preparation.')
    (OUT/'field_rt_validated_timing_v1.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'field_rt_validated_timing_v1_arrays.npz',**arrays)
    print(json.dumps(dict(gates=gates,rows=[{k:v for k,v in r.items() if k not in ('query_hashes','cpu_samples_ms','query_samples_ms','edt_samples_ms')} for r in rows])));return 0 if all(gates.values()) else 1

def main():
    log=OUT/'field_rt_validated_timing_events.jsonl'
    if log.exists():raise RuntimeError('refuse stale event log')
    coord=Path(os.environ['RT_COORDINATION_FILE']);expected=os.environ['RT_COORDINATION_SHA256']
    if hashlib.sha256(coord.read_bytes()).hexdigest()!=expected:raise RuntimeError('coordination changed')
    with open('/tmp/gpu.lock','a') as lock:
        event('lock_attempt');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            event('lock_acquired')
            if hashlib.sha256(coord.read_bytes()).hexdigest()!=expected:raise RuntimeError('coordination changed')
            return run()
        finally:
            event('lock_release_start');fcntl.flock(lock,fcntl.LOCK_UN);event('lock_released')

if __name__=='__main__':raise SystemExit(main())
