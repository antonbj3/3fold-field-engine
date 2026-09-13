"""CPU proof that snapshot service preserves owned output semantics."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
from pathlib import Path
import numpy as np
from mesh_field_owned_service_v1 import MeshFieldService
from field_shared_service_probe import FakeStage


def main():
    controls = {}
    with MeshFieldService(factory=FakeStage) as service:
        service.prepare([2], [], 1, [1, 2, 3])
        first, _ = service.query()
        original = tuple(np.array(x, copy=True) for x in first)
        second, _ = service.query()
        controls['fresh_query'] = bool(np.all(first[4] == 3) and np.all(second[4] == 4))
        controls['no_outstanding_lease'] = service._active is None
        controls['owned_writable'] = all(a.flags.owndata and a.flags.writeable for a in first[2:])
        controls['tuple_shape'] = isinstance(first[1], tuple)
        first[4][:] = 31
        controls['independent_writes'] = bool(np.all(second[4] == 4))
        service.prepare([8], [], 1, [0, 0, 0])
        replacement, _ = service.query()
        controls['replacement'] = bool(np.all(replacement[4] == 9) and np.all(second[4] == 4))
        controls['original_masks_unchanged'] = all(np.asarray(a).tobytes() == np.asarray(b).tobytes() for a, b in zip(first[:4], original[:4]))
    controls['after_close'] = bool(np.all(first[4] == 31) and np.all(second[4] == 4) and np.all(replacement[4] == 9))
    controls['worker_stopped'] = not service._process.is_alive()
    report = dict(status='VERIFIED-FRESH' if all(controls.values()) else 'OWN-GATE-FAIL',
                  scope='CPU fake stage; owned snapshot semantics only, no GPU timing claim',
                  controls=controls, passed=sum(controls.values()), total=len(controls))
    target = Path(__file__).resolve().parents[2] / 'reports/field_owned_service_cpu.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if all(controls.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
