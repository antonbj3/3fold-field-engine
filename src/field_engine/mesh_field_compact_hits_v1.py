"""Prepared mesh stage with native two-hit compaction and unchanged GPU kernels."""
import ctypes as ct
import numpy as np
from mesh_field_persistent_edt_v1 import PreparedMeshField as ReferenceField
from column_mask_ordered_v1 import OrderedColumnMask
from rt_hit_compact_v1 import NativeHitSorter


class _Rows:
    def __init__(self,handle,sorter):self.handle=handle;self.sorter=sorter
    def close(self):return self.handle.close()
    def query(self,xy):
        handle=self.handle;handle._check();q=np.array(xy,dtype=np.float64,order='C',copy=True)
        if q.ndim!=2 or q.shape[1]!=2 or not 1<=len(q)<=handle.capacity or not np.isfinite(q).all() or np.max(np.abs(q))>1e12:
            raise ValueError('bounded finite query XY required')
        counts=np.empty(len(q),np.int32);z=np.empty(len(q)*handle.hit_capacity,np.float64);s=np.empty(z.shape,np.int32);width=ct.c_uint32()
        status=handle._lib.columns_query_compact(handle._handle,q.ctypes.data,len(q),counts.ctypes.data,z.ctypes.data,s.ctypes.data,ct.byref(width))
        if status:
            handle._failed=True
            raise RuntimeError(f'column query status {status}')
        w=width.value
        return self.sorter.sort_rows(counts,z[:len(q)*w].reshape(len(q),w),s[:len(q)*w].reshape(len(q),w))


class PreparedMeshField(ReferenceField):
    def __init__(self,*args,sort_library,**kwargs):
        sorter=NativeHitSorter(sort_library)
        super().__init__(*args,**kwargs)
        try:
            if self._rt is not None:self._rt=_Rows(self._rt,sorter)
            self._mask=OrderedColumnMask(kwargs['column_library'])
        except Exception:
            super().close()
            raise
