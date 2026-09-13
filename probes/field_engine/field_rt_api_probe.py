"""Verify changed-input persistent RT handles against frozen CPU reference arrays."""

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
    with (OUT/'field_rt_api_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label,utc=datetime.now(timezone.utc).isoformat(),monotonic_ns=time.monotonic_ns()))+'\n');f.flush();os.fsync(f.fileno())

def pointer(a):return ct.c_void_p(a.ctypes.data)

def run():
    lib=ct.CDLL(os.environ['RT_WINDING_LIBRARY'])
    lib.rt_create.argtypes=[ct.c_char_p,ct.c_void_p,ct.c_uint32,ct.c_uint32,ct.POINTER(ct.c_void_p)];lib.rt_create.restype=ct.c_int
    lib.rt_query.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint32,ct.c_void_p];lib.rt_query.restype=ct.c_int
    lib.rt_destroy.argtypes=[ct.POINTER(ct.c_void_p)];lib.rt_destroy.restype=ct.c_int
    data=np.load(OUT/'field_rt_changed_query_reference_arrays.npz')
    refs=json.loads((OUT/'field_rt_changed_query_reference.json').read_text())['rows']
    rows=[];arrays={};prefix_ok=True;invalid_ok=True;destroy_ok=True
    guard();idle()
    for leg in range(2):
        for base in refs[::3]:
            name=base['case'];corners=data[name+'_corners'];capacity=base['rays'];handle=ct.c_void_p()
            guard();event('create_start_'+name+'_'+str(leg))
            status=lib.rt_create(os.fsencode(os.environ['RT_WINDING_PTX']),pointer(corners),len(corners),capacity,ct.byref(handle))
            if status:raise RuntimeError(f'create failed status{status}; no retry')
            try:
                for call,phase in enumerate((0,1,2,0)):
                    rays=data[f'{name}_{phase}_rays'];mask=data[f'{name}_{phase}_mask'];reference=data[f'{name}_{phase}_distance']
                    counts=np.full(capacity,1234567,dtype=np.int32)
                    event(f'query_start_{name}_{leg}_{call}')
                    status=lib.rt_query(handle,pointer(rays),capacity,pointer(counts))
                    event(f'query_end_{name}_{leg}_{call}')
                    if status:raise RuntimeError(f'query failed status{status}; no retry')
                    counts=counts.reshape(mask.shape);sd=edt(counts!=0,0.125)
                    rows.append(dict(case=name,leg=leg,call=call,phase=phase,counts_sha256=digest(counts),distance_sha256=digest(sd),
                                     occupancy_mismatches=int(np.count_nonzero((counts!=0)!=mask)),distance_mismatches=int(np.count_nonzero(sd!=reference))))
                    arrays[f'{name}_{leg}_{call}_counts']=counts;arrays[f'{name}_{leg}_{call}_distance']=sd
                    prefix=np.full(capacity,1234567,dtype=np.int32)
                    status=lib.rt_query(handle,pointer(rays),13,pointer(prefix))
                    prefix_ok &= status==0 and np.array_equal(prefix[:13],counts.ravel()[:13]) and bool(np.all(prefix[13:]==1234567))
                sentinel=np.full(capacity,1234567,dtype=np.int32)
                for count in (0,capacity+1):
                    invalid_ok &= lib.rt_query(handle,pointer(rays),count,pointer(sentinel))==1 and bool(np.all(sentinel==1234567))
                bad=rays.copy();bad[0,0]=np.nan
                invalid_ok &= lib.rt_query(handle,pointer(bad),capacity,pointer(sentinel))==1 and bool(np.all(sentinel==1234567))
            finally:
                status=lib.rt_destroy(ct.byref(handle));destroy_ok &= status==0 and not handle.value
                destroy_ok &= lib.rt_destroy(ct.byref(handle))==0
                event('destroy_'+name+'_'+str(leg));guard()
    gates=dict(full_repeat=all(np.array_equal(arrays[f'{r["case"]}_0_{r["call"]}_counts'],arrays[f'{r["case"]}_1_{r["call"]}_counts']) for r in rows if r['leg']==0),
               exact_occupancy=all(r['occupancy_mismatches']==0 for r in rows),exact_distance=all(r['distance_mismatches']==0 for r in rows),
               prefix_and_suffix=bool(prefix_ok),invalid_rejected_without_write=bool(invalid_ok),destroy_idempotent=bool(destroy_ok),no_observed_fault=True)
    report=dict(rows=rows,gates=gates,timing_measured=False,library_sha256=hashlib.sha256(Path(os.environ['RT_WINDING_LIBRARY']).read_bytes()).hexdigest())
    (OUT/'field_rt_api_v1.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'field_rt_api_v1_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 1

def main():
    log=OUT/'field_rt_api_events.jsonl'
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
