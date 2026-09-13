"""Coordinated exact full-array certification for the parallel EDT candidate."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from field_rt_lifecycle_probe import idle
from field_rt_winding_probe import guard
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts'


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def event(label):
    with (OUT/'field_edt_wrapper_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label, utc=datetime.now(timezone.utc).isoformat(),
                                monotonic_ns=time.monotonic_ns()))+'\n')
        f.flush()
        os.fsync(f.fileno())


def run():
    guard()
    idle()
    import threading
    from native_edt_v1 import NativeEDT
    owner=NativeEDT(os.environ['EDT_NATIVE_LIBRARY'])
    squared_distance=owner.squared_distance
    sample=np.zeros((2,2,2),bool);sample[0,0,0]=True
    invalid_masks=[sample.astype(np.uint8),sample[0],np.zeros((0,2,2),bool),
                   np.zeros((2,2,2),bool),np.ones((2,2,2),bool),np.zeros((129,2,2),bool),
                   np.zeros((65,65,65),bool)]
    invalid_pitches=[0,-1,float('nan'),float('inf'),(1,1,1),True,1e13,1e-13,'1']
    rejected=[]
    for mask,pitch in [(m,0.125) for m in invalid_masks]+[(sample,p) for p in invalid_pitches]:
        try:owner.signed_distance(mask,pitch);rejected.append(False)
        except ValueError:rejected.append(True)
    thread_errors=[]
    def foreign():
        for method,args in [(owner.squared_distance,(sample,)),(owner.signed_distance,(sample,0.125))]:
            try:method(*args)
            except RuntimeError:thread_errors.append(True)
    worker=threading.Thread(target=foreign);worker.start();worker.join()
    data = np.load(OUT/'field_edt_integer_v1_arrays.npz')
    names = [key.removesuffix('_square_0') for key in data.files if key.endswith('_square_0')]
    arrays, rows = {}, []
    for leg in range(2):
        for name in names:
            event(f'begin_{leg}_{name}')
            mask = np.asfortranarray(data[name+'_signed_0'] < 0)
            squares = [squared_distance(binary) for binary in (~mask, mask)]
            square_errors = sum(int(np.count_nonzero(value != data[f'{name}_square_{i}']))
                                for i, value in enumerate(squares))
            for i, value in enumerate(squares):
                arrays[f'{leg}_{name}_square_{i}'] = value
            for index, pitch in enumerate((0.125, 0.3, 0.7483641718372458)):
                sd = owner.signed_distance(mask,pitch)
                arrays[f'{leg}_{name}_signed_{index}'] = sd
                reference = data[f'{name}_signed_{index}']
                rows.append(dict(leg=leg, mask=name, pitch=pitch,
                                 square_mismatches=square_errors,
                                 signed_mismatches=int(np.count_nonzero(sd != reference)),
                                 max_error=float(np.max(np.abs(sd-reference))), sha256=digest(sd)))
            event(f'end_{leg}_{name}')
        guard()
    gates = dict(squared_exact=all(r['square_mismatches'] == 0 for r in rows),
                 signed_exact=all(r['signed_mismatches'] == 0 for r in rows),
                 full_repeat=all(digest(value) == digest(arrays['1_'+key[2:]])
                                 for key, value in arrays.items() if key.startswith('0_')),
                 invalid_rejected=all(rejected), foreign_thread_rejected=len(thread_errors)==2, no_observed_fault=True)
    report = dict(rows=rows, gates=gates, timing_measured=False, invalid_cases=len(rejected), foreign_thread_cases=len(thread_errors),
                  library_sha256=hashlib.sha256(Path(os.environ['EDT_NATIVE_LIBRARY']).read_bytes()).hexdigest(),
                  array_hashes={key: digest(value) for key, value in arrays.items()})
    (OUT/'field_edt_wrapper_v1.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(OUT/'field_edt_wrapper_v1_arrays.npz', **arrays)
    print(json.dumps(dict(gates=gates, rows=len(rows))))
    return 0 if all(gates.values()) else 1


def main():
    if (OUT/'field_edt_wrapper_events.jsonl').exists():
        raise RuntimeError('Refuse stale event log')
    coord = Path(os.environ['RT_COORDINATION_FILE'])
    expected = os.environ['RT_COORDINATION_SHA256']
    if hashlib.sha256(coord.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Coordination changed')
    with open('/tmp/gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            event('lock_acquired')
            if hashlib.sha256(coord.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Coordination changed')
            return run()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            event('lock_released')


if __name__ == '__main__':
    raise SystemExit(main())
