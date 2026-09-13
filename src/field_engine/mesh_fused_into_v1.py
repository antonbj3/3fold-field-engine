"""Caller-owned output adapter; native arithmetic and independent query unchanged."""
import ctypes as ct
import numpy as np
from mesh_field_fused_v1 import PreparedMeshField as FrozenField


class PreparedMeshField(FrozenField):
    def pin_outputs(self, region):
        self._check()
        if getattr(self, '_pinned', None) is not None:
            raise RuntimeError('Outputs already pinned')
        if self.method != 'rt_columns':
            return False
        size = int(np.prod(self.shape))
        expected = (2 * size + 63) // 64 * 64 + 4 * size
        if len(region) != expected or region.readonly:
            raise ValueError('Writable exact-size region required')
        pin = self._rt._lib.fused_pin
        pin.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        pin.restype = ct.c_int
        unpin = self._rt._lib.fused_unpin
        unpin.argtypes = [ct.c_void_p, ct.c_void_p]
        unpin.restype = ct.c_int
        owner = (ct.c_uint8 * len(region)).from_buffer(region)
        status = pin(self._rt._handle, ct.addressof(owner), len(region))
        if status:
            raise RuntimeError(f'Output pin status {status}')
        self._pinned = owner
        return True

    def unpin_outputs(self):
        # Cleanup must work even when evaluation has latched a native failure.
        import threading
        if threading.get_ident() != self._owner:
            raise RuntimeError('Field belongs to another thread')
        owner = getattr(self, '_pinned', None)
        if owner is not None:
            status = self._rt._lib.fused_unpin(self._rt._handle, ct.addressof(owner))
            if status:
                raise RuntimeError(f'Output unpin status {status}')
            self._pinned = None

    def evaluate_into(self, solid, surface, distance):
        self._check()
        arrays = (solid, surface, distance)
        for array, dtype in zip(arrays, (np.bool_, np.bool_, np.float32)):
            if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != tuple(self.shape):
                raise ValueError('Matching output shape and dtype required')
            if not array.flags.c_contiguous or not array.flags.writeable:
                raise ValueError('Contiguous writable outputs required')
        spans = sorted((a.ctypes.data, a.ctypes.data + a.nbytes) for a in arrays)
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError('Output buffers must not overlap')
        if self.method == 'rt_columns':
            status = self._rt._lib.fused_query(self._rt._handle, *(a.ctypes.data for a in arrays))
            if status:
                self._failed = self._rt._failed = True
                raise RuntimeError(f'Fused field status {status}')
        else:
            result = self.evaluate()
            for target, source in zip(arrays, (result[3], result[2], result[4])):
                np.copyto(target, source)

    def close(self):
        self.unpin_outputs()
        super().close()
