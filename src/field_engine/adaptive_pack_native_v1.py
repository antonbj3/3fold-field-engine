"""Native CPU adaptive packing with the frozen query implementation."""
import ctypes
import operator
from numbers import Real
from pathlib import Path
import sys
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks, NODE


class NativeAdaptiveBlocks(AdaptiveSampleBlocks):
    def __init__(self, samples, origin, pitch, *, library, leaf_samples=512):
        samples = np.asarray(samples)
        if samples.dtype != np.float32 or samples.ndim != 3 or not samples.size:
            raise ValueError("Nonempty three-dimensional float32 samples required")
        if max(samples.shape) > 4096 or samples.size > 16777216 or not np.isfinite(samples).all():
            raise ValueError("Finite bounded field required")
        origin = np.asarray(origin, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Finite three-coordinate origin required")
        if isinstance(pitch, (bool, np.bool_)) or not isinstance(pitch, Real):
            raise ValueError("Positive finite scalar pitch required")
        try:
            pitch = float(pitch)
        except (ValueError, TypeError):
            raise ValueError("Positive finite scalar pitch required") from None
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError("Positive finite scalar pitch required")
        if isinstance(leaf_samples, (bool, np.bool_)):
            raise ValueError("Integer leaf size required")
        try:
            leaf_samples = operator.index(leaf_samples)
        except TypeError:
            raise ValueError("Integer leaf size required") from None
        if not 8 <= leaf_samples <= 4096:
            raise ValueError("Leaf size outside bounded contract")

        if sys.byteorder != 'little':
            raise ValueError('Little-endian host required')
        lib=ctypes.CDLL(str(Path(library).resolve()))
        lib.adaptive_pack.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
        lib.adaptive_pack.restype=ctypes.c_void_p
        for name in ('adaptive_nodes','adaptive_samples'):
            function=getattr(lib,name);function.argtypes=[ctypes.c_void_p];function.restype=ctypes.c_int64
        lib.adaptive_copy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64]
        lib.adaptive_copy.restype=ctypes.c_int
        lib.adaptive_free.argtypes=[ctypes.c_void_p];lib.adaptive_free.restype=None
        array=np.ascontiguousarray(samples)
        shape=np.array(array.shape,dtype=np.int32)
        pointer=lib.adaptive_pack(array.ctypes.data,shape.ctypes.data,leaf_samples)
        if not pointer:
            raise ValueError('Native packing failed')
        try:
            nodes=np.empty(lib.adaptive_nodes(pointer),dtype=NODE)
            payload=np.empty(lib.adaptive_samples(pointer),dtype=np.float32)
            if lib.adaptive_copy(pointer,nodes.ctypes.data,len(nodes),payload.ctypes.data,len(payload)):
                raise RuntimeError('Native packing copy failed')
        finally:
            lib.adaptive_free(pointer)
        self.nodes=nodes;self.payload=payload;self.origin=origin.copy();self.pitch=pitch;self.shape=array.shape
        for value in (self.nodes,self.payload,self.origin):value.flags.writeable=False


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_native_pack_probe import main
    raise SystemExit(main())
