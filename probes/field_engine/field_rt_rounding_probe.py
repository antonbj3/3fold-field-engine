"""Measure world/local float32 conversion before a local-frame adapter design."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[2]

def measure():
    mesh=trimesh.creation.box(extents=[4,3,2])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(0.37,[1,2,3]))
    mesh.apply_translation([0.13,-0.21,0.37])
    rows=[]
    for shift in (0.,100.,10000.,1000000.):
        vertices=np.asarray(mesh.vertices)+shift
        anchor=vertices.min(axis=0)-1
        local=vertices-anchor
        world_round=vertices.astype(np.float32).astype(np.float64)
        local_round=local.astype(np.float32).astype(np.float64)
        rows.append(dict(translation=shift,world_max_error=float(np.max(np.abs(world_round-vertices))),
                         local_max_error=float(np.max(np.abs(local_round-local))),
                         world_hash=hashlib.sha256(world_round.tobytes()).hexdigest(),
                         local_hash=hashlib.sha256(local_round.tobytes()).hexdigest()))
    return rows

def main():
    first,second=measure(),measure()
    gates=dict(full_repeat=first==second,finite=all(np.isfinite(r['world_max_error']) and np.isfinite(r['local_max_error']) for r in first))
    report=dict(rows=first,gates=gates,timing_measured=False)
    (ROOT/'artifacts'/'field_rt_rounding.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if all(gates.values()) else 1

if __name__=='__main__': raise SystemExit(main())
