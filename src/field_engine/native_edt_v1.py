"""Validating synchronous native EDT wrapper with no implicit artifact writes."""
import ctypes as ct
from numbers import Real
import threading
import numpy as np
from field_edt_integer_v1 import validate


class NativeEDT:
    """One host thread, externally coordinated CUDA runtime, bounded mixed masks."""
    def __init__(self, library):
        self._thread=threading.get_ident()
        self._failed=False
        self._library=ct.CDLL(str(library))
        self._library.edt_squared.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
        self._library.edt_squared.restype=ct.c_int

    def _check(self):
        if threading.get_ident()!=self._thread:
            raise RuntimeError('NativeEDT belongs to another thread')
        if self._failed:
            raise RuntimeError('NativeEDT runtime previously failed; no retry')

    def squared_distance(self, binary):
        self._check()
        validate(binary)
        source=np.ascontiguousarray(binary,dtype=np.uint8)
        output=np.empty(source.shape,dtype=np.int32)
        status=self._library.edt_squared(source.ctypes.data,*source.shape,output.ctypes.data)
        if status:
            self._failed=status!=1
            raise RuntimeError(f'Native EDT status {status}')
        return output

    def signed_distance(self, mask, pitch):
        self._check()
        validate(mask)
        if isinstance(pitch,(bool,np.bool_)) or not isinstance(pitch,Real):
            raise ValueError('Expected a real scalar pitch')
        pitch=float(pitch)
        if not np.isfinite(pitch) or not 1e-12<=pitch<=1e12:
            raise ValueError('Pitch outside bounded contract')
        pair=[(np.sqrt(self.squared_distance(b).astype(np.float64))*pitch).astype(np.float32)
              for b in (~mask,mask)]
        sd=(pair[0]-pair[1]).astype(np.float32)
        return (sd-np.sign(sd)*np.float32(0.5*pitch)).astype(np.float32)


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_edt_wrapper_probe import main
    raise SystemExit(main())
