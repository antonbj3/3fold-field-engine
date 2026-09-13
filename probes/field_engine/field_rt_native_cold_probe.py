"""Repeat the frozen native composed benchmark in two independent cold processes."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/field_rt_native_cold'


def event(label):
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label,utc=datetime.now(timezone.utc).isoformat(),monotonic_ns=time.monotonic_ns()))+'\n')
        f.flush();os.fsync(f.fileno())


def child(index):
    import field_rt_native_timing_probe as frozen
    destination=OUT/f'child{index}'
    destination.mkdir()
    # Frozen observer reads input artifacts through OUT; copy only public references.
    for name in ('field_rt_changed_query_reference_arrays.npz','field_rt_changed_query_reference.json'):
        (destination/name).symlink_to(Path('../../')/name)
    frozen.OUT=destination
    return frozen.run()


def main():
    if OUT.exists():raise RuntimeError('Refuse stale cold-run outputs')
    coord=Path(os.environ['RT_COORDINATION_FILE']);expected=os.environ['RT_COORDINATION_SHA256']
    if hashlib.sha256(coord.read_bytes()).hexdigest()!=expected:raise RuntimeError('Coordination changed')
    OUT.mkdir()
    with open('/tmp/gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            event('lock_acquired')
            if hashlib.sha256(coord.read_bytes()).hexdigest()!=expected:raise RuntimeError('Coordination changed')
            reports=[]
            for index in range(2):
                event(f'child{index}_begin')
                result=subprocess.run([sys.executable,__file__,'--child',str(index)],capture_output=True,text=True,timeout=45)
                event(f'child{index}_end')
                path=OUT/f'child{index}/field_rt_native_timing_v1.json'
                if not path.exists():raise RuntimeError(f'Child exit{result.returncode} without report; no retry')
                report=json.loads(path.read_text())
                if not report['gates']['no_observed_fault']:raise RuntimeError('Observed fault; stop')
                if result.returncode not in (0,1):raise RuntimeError(f'Child runtime exit{result.returncode}')
                reports.append(report)
            gates={key:all(r['gates'][key] for r in reports) for key in reports[0]['gates']}
            gates['cross_process_full_hash_repeat']=[r['query_hashes'] for r in reports[0]['rows']]==[r['query_hashes'] for r in reports[1]['rows']]
            summary=dict(gates=gates,first_groups=[{k:v for k,v in r['rows'][0].items() if k not in ('query_hashes','cpu_samples_ms','query_samples_ms','edt_samples_ms')} for r in reports],timing_repeat_identity_claim=False)
            (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps(summary));return 0 if all(gates.values()) else 1
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN);event('lock_released')


if __name__=='__main__':
    raise SystemExit(child(int(sys.argv[2])) if len(sys.argv)==3 and sys.argv[1]=='--child' else main())
