"""Explicit CPU native library for bounded exact adaptive tile copies."""
import ctypes
from pathlib import Path
import sys
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks, NODE


class NativeTileReader:
    def __init__(self, library):
        if sys.byteorder != 'little':
            raise ValueError('Little-endian host required')
        self._library=ctypes.CDLL(str(Path(library).resolve()))
        self._gather=self._library.csg_gather
        self._gather.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,
                              ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self._gather.restype=ctypes.c_int

    def gather(self, field, lower, upper):
        if not isinstance(field,AdaptiveSampleBlocks):
            raise ValueError('Adaptive block field required')
        if field.nodes.dtype!=NODE or field.nodes.ndim!=1 or not field.nodes.flags.c_contiguous or field.payload.dtype!=np.float32 or field.payload.ndim!=1 or not field.payload.flags.c_contiguous:
            raise ValueError('Contiguous native node and sample arrays required')
        lo,hi=np.asarray(lower),np.asarray(upper)
        if lo.shape!=(3,) or hi.shape!=(3,) or not np.issubdtype(lo.dtype,np.integer) or not np.issubdtype(hi.dtype,np.integer):
            raise ValueError('Integral tile coordinates required')
        if np.any(lo<0) or np.any(hi<=lo) or np.any(hi>np.array(field.shape)) or np.prod(hi-lo)>8192:
            raise ValueError('Tile outside field or bounded capacity')
        lo,hi=np.ascontiguousarray(lo,dtype=np.int32),np.ascontiguousarray(hi,dtype=np.int32)
        result=np.empty(tuple(hi-lo),dtype=np.float32)
        code=self._gather(field.nodes.ctypes.data,len(field.nodes),field.payload.ctypes.data,len(field.payload),
                          lo.ctypes.data,hi.ctypes.data,result.ctypes.data)
        if code:
            raise ValueError(f'Native tile validation failed: {code}')
        return result


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_csg_native_probe import main
    raise SystemExit(main())
