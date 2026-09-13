"""Thread-owned persistent paired EDT with explicit close and no output cache."""
import ctypes as ct
from numbers import Real
import operator
import threading
import numpy as np
from native_edt_large_v1 import validate


class PersistentPairEDT:
    def __init__(self,shape,*,library):
        raw=tuple(shape)
        if len(raw)!=3 or any(isinstance(x,(bool,np.bool_)) for x in raw):
            raise ValueError('Three integer dimensions required')
        try:self.shape=tuple(operator.index(x) for x in raw)
        except TypeError:raise ValueError('Three integer dimensions required') from None
        if min(self.shape)<1 or max(self.shape)>512 or np.prod(self.shape)>1048576:
            raise ValueError('Shape outside bounded EDT capacity')
        self._owner=threading.get_ident();self._failed=False;self._closed=False
        self._library=ct.CDLL(str(library))
        self._library.edt_pair_create.argtypes=[ct.c_int]*3;self._library.edt_pair_create.restype=ct.c_void_p
        self._library.edt_pair_query.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_double,ct.c_void_p,ct.c_void_p];self._library.edt_pair_query.restype=ct.c_int
        self._library.edt_pair_destroy.argtypes=[ct.c_void_p];self._library.edt_pair_destroy.restype=None
        self._pointer=self._library.edt_pair_create(*self.shape)
        if not self._pointer:raise RuntimeError('Persistent EDT allocation failed')

    def _check(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('EDT belongs to another thread')
        if self._closed or self._failed:raise RuntimeError('EDT closed or failed')

    def query(self,mask,pitch,*,with_squared=False):
        self._check();validate(mask)
        if mask.shape!=self.shape:raise ValueError('Mask shape differs from prepared shape')
        if isinstance(pitch,(bool,np.bool_)) or not isinstance(pitch,Real) or not np.isfinite(pitch) or not 1e-12<=pitch<=1e12:
            raise ValueError('Bounded real pitch required')
        source=np.ascontiguousarray(mask,dtype=np.uint8)
        output=np.empty(self.shape,dtype=np.float32)
        squared=np.empty((2,)+self.shape,dtype=np.int32) if with_squared else None
        status=self._library.edt_pair_query(self._pointer,source.ctypes.data,float(pitch),output.ctypes.data,
                                           squared.ctypes.data if squared is not None else None)
        if status:
            self._failed=status!=1
            raise RuntimeError(f'Persistent EDT status {status}')
        return (output,squared) if with_squared else output

    def close(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('EDT belongs to another thread')
        if not self._closed:
            self._library.edt_pair_destroy(self._pointer);self._pointer=None;self._closed=True

    def __enter__(self):self._check();return self
    def __exit__(self,*args):self.close()


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_persistent_edt_probe import main
    raise SystemExit(main())
