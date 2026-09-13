"""One spawned owner process keeps one prepared RT mesh alive between requests.

Trusted local Python IPC only. Every query recomputes full fields; no result cache.
The caller owns GPU coordination for the complete service lifetime.
"""
import multiprocessing as mp
import threading
import time


def _worker(connection, libraries):
    from mesh_field_native_mask_v1 import PreparedMeshField
    field = None
    try:
        while True:
            operation, payload = connection.recv()
            if operation == 'close':
                break
            try:
                if operation == 'prepare':
                    if field is not None:
                        field.close()
                        field = None
                    field = PreparedMeshField(*payload, **libraries)
                    connection.send(('ok', field.shape))
                elif operation == 'query':
                    if field is None:
                        raise RuntimeError('Prepare a mesh before querying')
                    start = time.perf_counter_ns()
                    result = field.evaluate()
                    connection.send(('ok', (result, (time.perf_counter_ns()-start)*1e-6)))
                else:
                    raise ValueError('Unknown service operation')
            except Exception as error:
                connection.send(('error', f'{type(error).__name__}: {error}'))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if field is not None:
            field.close()
        connection.close()


class MeshFieldService:
    """Synchronous thread-owned client; replacement closes the previous mesh.

    timeout bounds response waiting. A timeout or worker death closes the service
    permanently; native errors are returned without claiming successful output.
    """
    def __init__(self, *, timeout=60., **libraries):
        if not 0 < timeout <= 300:
            raise ValueError('Timeout must be in (0, 300] seconds')
        self._owner = threading.get_ident()
        self._timeout = float(timeout)
        self._closed = False
        context = mp.get_context('spawn')
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, libraries))
        self._process.start()
        child.close()

    def _check(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Service belongs to another thread')
        if self._closed:
            raise RuntimeError('Service is closed')

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
            raise RuntimeError(value)
        return value

    def prepare(self, vertices, faces, pitch, origin):
        return self._call('prepare', (vertices, faces, pitch, origin))

    def query(self):
        """Return (full stage result, worker evaluation milliseconds)."""
        return self._call('query')

    def close(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Service belongs to another thread')
        if self._closed:
            return
        self._closed = True
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
            self._process.join()
        self._connection.close()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_service_probe import main
    raise SystemExit(main())
