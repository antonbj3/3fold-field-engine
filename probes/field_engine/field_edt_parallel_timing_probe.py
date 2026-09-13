"""Two fresh-process prepared-mask EDT timing legs with cold setup retained."""

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
import subprocess
import sys
import time
import numpy as np
from scipy import ndimage
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts'
LOG = OUT/'field_edt_parallel_timing_events.jsonl'


def event(label):
    with LOG.open('a') as f:
        f.write(json.dumps(dict(label=label, utc=datetime.now(timezone.utc).isoformat(),
                                monotonic_ns=time.monotonic_ns()))+'\n')
        f.flush()
        os.fsync(f.fileno())


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def signed(pair):
    sd = (pair[0]-pair[1]).astype(np.float32)
    return (sd-np.sign(sd)*np.float32(0.0625)).astype(np.float32)


def child(leg):
    data = np.load(OUT/'field_rt_changed_query_reference_arrays.npz')
    names = [key.removesuffix('_mask') for key in data.files if key.endswith('_mask')]
    event(f'{leg}_setup_begin')
    start = time.perf_counter_ns()
    import warp as wp
    from field_edt_parallel_v1 import squared_distance
    wp.init()
    setup_ms = (time.perf_counter_ns()-start)/1e6
    event(f'{leg}_setup_end')
    rows, arrays = [], {}
    for index, name in enumerate(names):
        mask = data[name+'_mask']
        event(f'{leg}_{name}_cpu_begin')
        start = time.perf_counter_ns()
        reference = signed([ndimage.distance_transform_edt(b, sampling=(0.125,)*3).astype(np.float32)
                            for b in (~mask, mask)])
        cpu_ms = (time.perf_counter_ns()-start)/1e6
        event(f'{leg}_{name}_cpu_end')
        event(f'{leg}_{name}_candidate_begin')
        start = time.perf_counter_ns()
        squares = [squared_distance(b) for b in (~mask, mask)]
        actual = signed([(np.sqrt(s.astype(np.float64))*0.125).astype(np.float32) for s in squares])
        candidate_ms = (time.perf_counter_ns()-start)/1e6
        event(f'{leg}_{name}_candidate_end')
        charged = candidate_ms+(setup_ms if index == 0 else 0)
        rows.append(dict(leg=leg, mask=name, cpu_ms=cpu_ms, candidate_call_ms=candidate_ms,
                         setup_charge_ms=setup_ms if index == 0 else 0,
                         candidate_accounted_ms=charged, ratio=charged/cpu_ms,
                         mismatches=int(np.count_nonzero(actual != reference)),
                         saved_reference_mismatches=int(np.count_nonzero(reference != data[name+'_distance'])),
                         sha256=digest(actual)))
        arrays[name] = actual
    (OUT/f'field_edt_parallel_timing_leg{leg}.json').write_text(json.dumps(dict(rows=rows, setup_ms=setup_ms), indent=2)+'\n')
    np.savez_compressed(OUT/f'field_edt_parallel_timing_leg{leg}_arrays.npz', **arrays)
    return 0


def main():
    from field_rt_lifecycle_probe import idle
    from field_rt_winding_probe import guard
    if LOG.exists():
        raise RuntimeError('Refuse stale timing log')
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
            for leg in range(2):
                guard()
                idle()
                result = subprocess.run([sys.executable, __file__, '--child', str(leg)],
                                        capture_output=True, text=True, timeout=45)
                if result.returncode:
                    raise RuntimeError(f'Child failed exit {result.returncode}; no retry')
                guard()
            a = json.loads((OUT/'field_edt_parallel_timing_leg0.json').read_text())
            b = json.loads((OUT/'field_edt_parallel_timing_leg1.json').read_text())
            rows = a['rows']+b['rows']
            gates = dict(exact_reference=all(r['mismatches'] == r['saved_reference_mismatches'] == 0 for r in rows),
                         full_repeat=[r['sha256'] for r in a['rows']] == [r['sha256'] for r in b['rows']],
                         all_accounted_calls_faster=all(r['ratio'] < 1 for r in rows),
                         idle_prechecks=True, no_observed_fault=True)
            report = dict(rows=rows, gates=gates, timing_repeat_identity_claim=False,
                          setup_policy='Fresh child runtime per leg, existing compiler cache, no warmup, setup charged to first call.')
            (OUT/'field_edt_parallel_timing_v1.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report))
            return 0 if all(gates.values()) else 2
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            event('lock_released')


if __name__ == '__main__':
    raise SystemExit(child(int(sys.argv[2])) if len(sys.argv) == 3 and sys.argv[1] == '--child' else main())
