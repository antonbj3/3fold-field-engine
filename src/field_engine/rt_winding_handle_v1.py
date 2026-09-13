"""Explicit same-thread owner for the external signed-winding C ABI.

Inputs are finite local-coordinate float32 arrays. Caller owns hardware scheduling,
fault checks and external compiler-cache placement. This module writes no reports.
Use a with block or close(); no destructor invokes CUDA during interpreter shutdown.
"""
import ctypes as ct
import operator
import threading
import numpy as np


def _array(value,tail):
    a=np.asarray(value)
    if a.dtype!=np.dtype(np.float32):raise TypeError('float32 input required')
    if a.ndim!=len(tail)+1 or a.shape[1:]!=tail or not len(a):raise ValueError('invalid array shape')
    if not np.isfinite(a).all():raise ValueError('finite input required')
    return np.ascontiguousarray(a)


class WindingHandle:
    """Immutable mesh, fixed query capacity; synchronous query returns new int32 array."""
    def __init__(self,library,ptx,corners,capacity):
        corners=_array(corners,(3,3))
        if isinstance(capacity,(bool,np.bool_)):raise TypeError('integer capacity required')
        capacity=operator.index(capacity)
        if not 1<=capacity<=100000000 or len(corners)>10000000:raise ValueError('capacity or mesh limit exceeded')
        self._owner=threading.get_ident();self._capacity=capacity;self._handle=ct.c_void_p()
        self._lib=ct.CDLL(str(library))
        self._lib.rt_create.argtypes=[ct.c_char_p,ct.c_void_p,ct.c_uint32,ct.c_uint32,ct.POINTER(ct.c_void_p)]
        self._lib.rt_query.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint32,ct.c_void_p]
        self._lib.rt_destroy.argtypes=[ct.POINTER(ct.c_void_p)]
        for name in ('rt_create','rt_query','rt_destroy'):getattr(self._lib,name).restype=ct.c_int
        import os
        self._check(self._lib.rt_create(os.fsencode(ptx),ct.c_void_p(corners.ctypes.data),len(corners),capacity,ct.byref(self._handle)))

    @staticmethod
    def _check(status):
        if status:raise RuntimeError(f'native winding status{status}')

    def _thread(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('handle belongs to another thread')

    @property
    def closed(self):return not bool(self._handle.value)

    def query(self,origins):
        self._thread()
        if self.closed:raise RuntimeError('handle is closed')
        rays=_array(origins,(3,))
        if len(rays)>self._capacity:raise ValueError('query exceeds capacity')
        result=np.empty(len(rays),dtype=np.int32)
        self._check(self._lib.rt_query(self._handle,ct.c_void_p(rays.ctypes.data),len(rays),ct.c_void_p(result.ctypes.data)))
        return result

    def close(self):
        self._thread()
        if not self.closed:self._check(self._lib.rt_destroy(ct.byref(self._handle)))

    def __enter__(self):
        self._thread()
        if self.closed:raise RuntimeError('handle is closed')
        return self

    def __exit__(self,exc_type,exc,tb):self.close()


if __name__=='__main__':
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).with_name('field_rt_wrapper_probe.py')),run_name='__main__')
