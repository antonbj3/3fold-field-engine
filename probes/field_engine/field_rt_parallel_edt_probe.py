"""Certify persistent RT and parallel integer EDT coexistence with frozen outputs."""

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
from field_rt_lifecycle_probe import idle
from field_rt_winding_probe import guard
from field_rt_mesh_probe import digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts'

def event(label):
    with (OUT/'field_rt_parallel_edt_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label,utc=datetime.now(timezone.utc).isoformat(),monotonic_ns=time.monotonic_ns()))+'\n');f.flush();os.fsync(f.fileno())

def edt(mask,h):
    from field_edt_parallel_v1 import squared_distance
    squares=[squared_distance(binary) for binary in (~mask,mask)]
    pair=[(np.sqrt(square.astype(np.float64))*h).astype(np.float32) for square in squares]
    sd=(pair[0]-pair[1]).astype(np.float32)
    return (sd-np.sign(sd)*np.float32(0.5*h)).astype(np.float32)

def run():
    import threading
    from rt_winding_handle_v1 import WindingHandle
    data=np.load(OUT/'field_rt_changed_query_reference_arrays.npz')
    refs=json.loads((OUT/'field_rt_changed_query_reference.json').read_text())['rows']
    rows=[];arrays={};invalid_ok=True;thread_ok=True;closed_ok=True;prefix_ok=True
    guard();idle()
    import warp as wp
    wp.init()
    for leg in range(2):
        for base in refs[::3]:
            name=base['case'];capacity=base['rays']
            event('create_'+name+'_'+str(leg))
            with WindingHandle(os.environ['RT_WINDING_LIBRARY'],os.environ['RT_WINDING_PTX'],data[name+'_corners'],capacity) as owner:
                for call,phase in enumerate((0,1,2,0)):
                    rays=data[f'{name}_{phase}_rays'];mask=data[f'{name}_{phase}_mask'];reference=data[f'{name}_{phase}_distance']
                    counts=owner.query(rays).reshape(mask.shape);sd=edt(counts!=0,0.125)
                    rows.append(dict(case=name,leg=leg,call=call,mask_errors=int(np.count_nonzero((counts!=0)!=mask)),distance_errors=int(np.count_nonzero(sd!=reference)),counts_sha256=digest(counts)))
                    arrays[f'{name}_{leg}_{call}_counts']=counts;arrays[f'{name}_{leg}_{call}_distance']=sd
                    prefix_ok &= np.array_equal(owner.query(rays[:13]),counts.ravel()[:13])
                bad=rays.copy();bad[0,0]=np.nan
                invalid=[rays.astype(np.float64),rays[:,0],rays[:0],bad,np.zeros((capacity+1,3),np.float32)]
                for value in invalid:
                    try:owner.query(value);invalid_ok=False
                    except (TypeError,ValueError):pass
                errors=[]
                def other_thread():
                    try:owner.query(rays[:1])
                    except RuntimeError:errors.append('query')
                    try:owner.close()
                    except RuntimeError:errors.append('close')
                t=threading.Thread(target=other_thread);t.start();t.join();thread_ok &= errors==['query','close'] and not owner.closed
            owner.close();closed_ok &= owner.closed
            try:owner.query(rays);closed_ok=False
            except RuntimeError:pass
            event('closed_'+name+'_'+str(leg));guard()
    gates=dict(full_repeat=all(all(np.array_equal(arrays[f'{r["case"]}_0_{r["call"]}_{kind}'],arrays[f'{r["case"]}_1_{r["call"]}_{kind}']) for kind in ('counts','distance')) for r in rows if r['leg']==0),
               exact_mask=all(r['mask_errors']==0 for r in rows),exact_distance=all(r['distance_errors']==0 for r in rows),prefix_exact=bool(prefix_ok),
               invalid_rejected=bool(invalid_ok),foreign_thread_rejected=bool(thread_ok),closed_rejected=bool(closed_ok),no_observed_fault=True)
    report=dict(rows=rows,gates=gates,timing_measured=False)
    (OUT/'field_rt_parallel_edt_v1.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'field_rt_parallel_edt_v1_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 1

def main():
    log=OUT/'field_rt_parallel_edt_events.jsonl'
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
