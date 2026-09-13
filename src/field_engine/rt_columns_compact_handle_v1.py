"""Compact host rows beside the frozen bounded column handle."""
import ctypes as ct
import numpy as np
from rt_columns_handle_v1 import ColumnsHandle as FullColumnsHandle


class ColumnsHandle(FullColumnsHandle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        try:
            p=ct.c_void_p
            self._lib.columns_query_compact.argtypes=[p,p,ct.c_uint32,p,p,p,ct.POINTER(ct.c_uint32)]
            self._lib.columns_query_compact.restype=ct.c_int
        except Exception:
            self.close()
            raise

    def query(self,xy):
        self._check();q=np.array(xy,dtype=np.float64,order='C',copy=True)
        if q.ndim!=2 or q.shape[1]!=2 or not 1<=len(q)<=self.capacity or not np.isfinite(q).all() or np.max(np.abs(q))>1e12:raise ValueError('bounded finite query XY required')
        counts=np.empty(len(q),np.int32);z=np.empty(len(q)*self.hit_capacity,np.float64);s=np.empty(z.shape,np.int32);width=ct.c_uint32()
        status=self._lib.columns_query_compact(self._handle,q.ctypes.data,len(q),counts.ctypes.data,z.ctypes.data,s.ctypes.data,ct.byref(width))
        if status:
            self._failed=True
            raise RuntimeError(f'column query status {status}')
        w=width.value;z=z[:len(q)*w].reshape(len(q),w);s=s[:len(q)*w].reshape(len(q),w)
        valid=np.arange(w)[None,:]<counts[:,None]
        c=np.broadcast_to(np.arange(len(q))[:,None],valid.shape)[valid];z=z[valid];s=s[valid]
        order=np.lexsort((s,z,c));return c[order],z[order],s[order]
