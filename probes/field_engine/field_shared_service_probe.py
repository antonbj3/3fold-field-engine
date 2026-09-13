"""CPU-only shared-service ownership, recomputation and mapping lifetime proof."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import gc
import ctypes as ct
from types import SimpleNamespace
import json
from pathlib import Path
import threading
import weakref
from multiprocessing.shared_memory import SharedMemory
import numpy as np
from mesh_field_shared_service_v1 import MeshFieldService


class FakeStage:
    def __init__(self, vertices, faces, pitch, origin):
        if pitch < 0:
            raise ValueError('Requested preparation failure')
        self.shape = (4, 3, 2)
        self._gmin = np.array(origin, np.int64)
        self.value = float(np.asarray(vertices).sum())
        self.calls = 0

    def evaluate_into(self, solid, surface, distance):
        self.calls += 1
        solid[:] = self.calls % 2 == 1
        surface[:] = not solid.flat[0]
        distance[:] = self.value + self.calls

    def pin_outputs(self, region):
        return False

    def unpin_outputs(self):
        raise AssertionError('CPU fixture must not pin')

    def close(self):
        pass


def main():
    controls = {}

    def refuses(name, call):
        try:
            call()
        except (RuntimeError, ValueError, FileNotFoundError, TimeoutError, EOFError, BrokenPipeError):
            controls[name] = True
        else:
            controls[name] = False

    for bad in (0, -1, float('nan'), float('inf'), 301):
        refuses(f'timeout_{bad}', lambda: MeshFieldService(timeout=bad, factory=FakeStage))
    service = MeshFieldService(factory=FakeStage)
    try:
        refuses('query_before_prepare', service.query)
        service.prepare([2], [], 1, [1, 2, 3])
        controls['cpu_unpinned'] = not service.metadata['pinned']
        lease = service.query()
        surface, solid, distance = lease.result[2:]
        controls['full_first_result'] = bool(solid.all() and not surface.any() and np.all(distance == 3))
        controls['readonly'] = not any(a.flags.writeable for a in (solid, surface, distance))
        refuses('readonly_cannot_enable', lambda: distance.setflags(write=True))
        refuses('busy_query', service.query)
        refuses('busy_prepare', lambda: service.prepare([8], [], 1, [0, 0, 0]))
        t = threading.Thread(target=lambda: refuses('wrong_thread_query', service.query))
        t.start(); t.join()
        t = threading.Thread(target=lambda: refuses('wrong_thread_release', lease.release))
        t.start(); t.join()
        t = threading.Thread(target=lambda: refuses('wrong_thread_close', service.close))
        t.start(); t.join()
        refuses('still_busy_after_wrong_thread', service.query)
        own_copy = distance.copy()
        lease.release(); lease.release()
        refuses('released_lease_context', lease.__enter__)
        with service.query() as second:
            controls['recomputed'] = bool(np.all(second.result[4] == 4) and not second.result[3].any())
            controls['borrowed_reused'] = bool(np.all(distance == 4) and np.all(own_copy == 3))
        old_backing = weakref.ref(service._backing)
        old_name = service._backing.shm.name
        retained = np.asarray(distance)[1:].view(np.ndarray)
        del lease, second, surface, solid, distance
        service.prepare([8], [], 1, [0, 0, 0])
        gc.collect()
        controls['array_base_retains_old_mapping'] = old_backing() is not None and bool(np.all(retained == 4))
        refuses('old_name_unlinked', lambda: SharedMemory(name=old_name))
        del retained
        gc.collect()
        controls['mapping_freed_after_last_view'] = old_backing() is None
        with service.query() as replacement:
            controls['replacement_recomputed'] = bool(np.all(replacement.result[4] == 9))
        refuses('bad_prepare', lambda: service.prepare([0], [], -1, [0, 0, 0]))
        refuses('no_stale_after_bad_prepare', service.query)
        service.prepare([11], [], 1, [0, 0, 0])
        live = service.query()
        retained = np.asarray(live.result[4])
        backing = weakref.ref(service._backing)
        service.close(); service.close()
        controls['worker_stopped'] = not service._process.is_alive()
        controls['active_view_safe_after_close'] = bool(np.all(retained == 12))
        live.release()
        del live, retained
        gc.collect()
        controls['closed_mapping_freed'] = backing() is None
        refuses('closed_query', service.query)
        refuses('closed_prepare', lambda: service.prepare([], [], 1, [0, 0, 0]))
    finally:
        service.close()
    # Abrupt worker death must not expose the previous successful field as new.
    with MeshFieldService(factory=FakeStage) as killed:
        killed.prepare([1], [], 1, [0, 0, 0])
        killed._process.terminate(); killed._process.join(2)
        refuses('worker_death', killed.query)
        controls['worker_death_closes'] = killed._closed
    # Validate the native-output seam without initializing any GPU library.
    from mesh_fused_into_v1 import PreparedMeshField
    stage = PreparedMeshField.__new__(PreparedMeshField)
    stage.shape = (2, 2, 2)
    stage._owner = threading.get_ident()
    stage._closed = stage._failed = False
    stage.method = 'rt_columns'
    native_calls = []

    def fake_query(handle, solid_ptr, surface_ptr, distance_ptr):
        native_calls.append(True)
        np.ctypeslib.as_array((ct.c_uint8 * 8).from_address(solid_ptr))[:] = 1
        np.ctypeslib.as_array((ct.c_uint8 * 8).from_address(surface_ptr))[:] = 0
        np.ctypeslib.as_array((ct.c_float * 8).from_address(distance_ptr))[:] = 2.5
        return 0

    stage._rt = SimpleNamespace(_handle=None, _lib=SimpleNamespace(fused_query=fake_query), _failed=False)
    a, b = np.empty(stage.shape, bool), np.empty(stage.shape, bool)
    d = np.empty(stage.shape, np.float32)
    refuses('output_overlap', lambda: stage.evaluate_into(a, a, d))
    refuses('output_dtype', lambda: stage.evaluate_into(a, b, d.astype(np.float64)))
    refuses('output_shape', lambda: stage.evaluate_into(a, b, d.reshape(4, 2)))
    refuses('output_stride', lambda: stage.evaluate_into(a, b, d[:, :, ::-1]))
    d.flags.writeable = False
    refuses('output_readonly', lambda: stage.evaluate_into(a, b, d))
    d.flags.writeable = True
    refuses('pin_size', lambda: stage.pin_outputs(memoryview(bytearray(1))))
    controls['invalid_outputs_never_call_native'] = not native_calls
    stage.evaluate_into(a, b, d)
    controls['output_pointer_order'] = bool(a.all() and not b.any() and np.all(d == 2.5))
    stage._rt._lib.fused_query = lambda *args: 4
    refuses('native_failure', lambda: stage.evaluate_into(a, b, d))
    controls['both_native_failure_latches'] = stage._failed and stage._rt._failed
    refuses('native_retry_refused', lambda: stage.evaluate_into(a, b, d))
    report = dict(status='VERIFIED-FRESH' if all(controls.values()) else 'OWN-GATE-FAIL',
                  scope='CPU fake stage; no GPU correctness or latency claim',
                  controls=controls, passed=sum(controls.values()), total=len(controls))
    target = Path(__file__).resolve().parents[2] / 'reports/field_shared_service_cpu.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if all(controls.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
