"""Spawned field service with explicit borrowed, read-only shared-memory results.

Trusted local IPC. Query recomputes the field. A lease must be released before
query or prepare; released arrays may change on the next query. Copy arrays for
independent snapshots. Array bases retain their mapping even after service close.
The caller coordinates hardware for the entire service lifetime.
"""
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import threading
import time
import weakref
import numpy as np


def layout(shape):
    size = int(np.prod(shape))
    distance_offset = (2 * size + 63) // 64 * 64
    return size, distance_offset, distance_offset + 4 * size


class _Backing:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.size, self.distance_offset, total = layout(shape)
        self.shm = SharedMemory(create=True, size=total)
        self.unlinked = False

    def unlink(self):
        if not self.unlinked:
            self.shm.unlink()
            self.unlinked = True

    def __del__(self):
        if hasattr(self, 'shm'):
            try:
                self.unlink()
            finally:
                self.shm.close()


class _ArrayOwner:
    def __init__(self, backing, offset, dtype):
        self.backing = backing
        # ndarray's base is this owner, including after np.asarray and slicing.
        raw = np.ndarray((backing.shm.size,), np.uint8, buffer=backing.shm.buf)
        self.__array_interface__ = dict(version=3, shape=backing.shape,
            typestr=np.dtype(dtype).str, data=(raw.ctypes.data + offset, True))


def _views(backing):
    return tuple(np.asarray(_ArrayOwner(backing, offset, dtype)) for offset, dtype in
                 ((0, np.bool_), (backing.size, np.bool_),
                  (backing.distance_offset, np.float32)))


def _worker(connection, libraries, factory):
    field = shared = arrays = None
    pinned = False

    def clear():
        nonlocal field, shared, arrays, pinned
        try:
            if pinned:
                field.unpin_outputs()
        finally:
            pinned = False
            arrays = None
            if shared is not None:
                shared.close()
                shared = None
            if field is not None:
                prior, field = field, None
                prior.close()

    try:
        if factory is None:
            from mesh_fused_into_v1 import PreparedMeshField
            factory = PreparedMeshField
        while True:
            operation, payload = connection.recv()
            if operation == 'close':
                break
            try:
                if operation == 'prepare':
                    clear()
                    field = factory(*payload, **libraries)
                    connection.send(('ok', (tuple(field.shape), field._gmin.copy())))
                elif operation == 'attach':
                    if field is None or shared is not None:
                        raise RuntimeError('Prepare exactly once before attachment')
                    shared = SharedMemory(name=payload)
                    size, offset, total = layout(field.shape)
                    if shared.size != total:
                        raise ValueError('Shared region size mismatch')
                    arrays = (np.ndarray(field.shape, np.bool_, buffer=shared.buf),
                              np.ndarray(field.shape, np.bool_, buffer=shared.buf, offset=size),
                              np.ndarray(field.shape, np.float32, buffer=shared.buf, offset=offset))
                    pinned = field.pin_outputs(shared.buf)
                    connection.send(('ok', dict(pinned=pinned)))
                elif operation == 'query':
                    if field is None or arrays is None:
                        raise RuntimeError('Prepare and attach before querying')
                    start = time.perf_counter_ns()
                    field.evaluate_into(*arrays)
                    connection.send(('ok', (time.perf_counter_ns() - start) * 1e-6))
                else:
                    raise ValueError('Unknown service operation')
            except Exception as error:
                # Any worker error invalidates its prepared state.
                try:
                    clear()
                except Exception:
                    pass
                connection.send(('error', f'{type(error).__name__}: {error}'))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        try:
            clear()
        finally:
            connection.close()


class FieldLease:
    def __init__(self, service, token, worker_ms):
        self._service = weakref.ref(service)
        self._token = token
        self._released = False
        solid, surface, distance = _views(service._backing)
        self.result = (service._gmin.copy(), service._backing.shape, surface, solid, distance)
        self.worker_ms = worker_ms

    def release(self):
        service = self._service()
        if service is not None:
            if threading.get_ident() != service._owner:
                raise RuntimeError('Lease belongs to another thread')
            if service._active == self._token:
                service._active = None
        self._released = True

    def __enter__(self):
        if self._released:
            raise RuntimeError('Lease is released')
        return self

    def __exit__(self, *args):
        self.release()


class MeshFieldService:
    """Single owner, one outstanding lease; response waiting has a timeout.

    prepare includes process setup, shared-memory attachment and optional pinning.
    close stops the worker before unlinking; extant arrays retain valid mappings.
    """
    def __init__(self, *, timeout=60., factory=None, **libraries):
        if not np.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('Timeout must be in (0, 300] seconds')
        self._owner = threading.get_ident()
        self._timeout = float(timeout)
        self._closed = False
        self._backing = None
        self._active = None
        self._generation = 0
        self._ready = False
        context = mp.get_context('spawn')
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, libraries, factory))
        self._process.start()
        child.close()

    def _check(self, idle=False):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Service belongs to another thread')
        if self._closed:
            raise RuntimeError('Service is closed')
        if idle and self._active is not None:
            raise RuntimeError('Release the outstanding field lease first')

    def _call(self, operation, payload=None):
        self._check()
        try:
            self._connection.send((operation, payload))
            if not self._connection.poll(self._timeout):
                raise TimeoutError('Service response timed out')
            status, value = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError, TimeoutError):
            self.close()
            raise
        if status != 'ok':
            self._ready = False
            raise RuntimeError(value)
        return value

    def _drop_backing(self):
        if self._backing is not None:
            self._backing.unlink()
            self._backing = None

    def prepare(self, vertices, faces, pitch, origin):
        self._check(idle=True)
        self._ready = False
        shape, self._gmin = self._call('prepare', (vertices, faces, pitch, origin))
        self._drop_backing()
        try:
            self._backing = _Backing(shape)
            self.metadata = self._call('attach', self._backing.shm.name)
            self._ready = True
        except Exception:
            self.close()
            raise
        return shape

    def query(self):
        self._check(idle=True)
        if not self._ready:
            raise RuntimeError('Prepare a field before querying')
        worker_ms = self._call('query')
        self._generation += 1
        self._active = self._generation
        return FieldLease(self, self._active, worker_ms)

    def close(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Service belongs to another thread')
        if self._closed:
            return
        try:
            self._connection.send(('close', None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2)
        if self._process.is_alive():
            raise RuntimeError('Worker did not stop; mapping retained')
        self._closed = True
        self._ready = False
        self._connection.close()
        self._drop_backing()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()
