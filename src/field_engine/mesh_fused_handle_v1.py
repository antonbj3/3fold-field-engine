"""Explicit immutable-geometry native field handle with full output snapshots."""
import ctypes as ct
from numbers import Real
import operator
import threading
import numpy as np
from faltkarna_v1_mesh_to_sdf import VOXEL_PROVPUNKT


class FusedField:
    def __init__(self,library,ptx,corners,anchor,xy,shape,pitch):
        raw=tuple(shape)
        if len(raw)!=3 or any(isinstance(x,(bool,np.bool_)) for x in raw):raise ValueError('Three integer dimensions required')
        try:self.shape=tuple(operator.index(x) for x in raw)
        except TypeError:raise ValueError('Three integer dimensions required') from None
        if min(self.shape)<1 or max(self.shape)>512 or np.prod(self.shape)>1048576 or self.shape[0]*self.shape[1]>262144:raise ValueError('Grid outside native capacity')
        if isinstance(pitch,(bool,np.bool_)) or not isinstance(pitch,Real) or not np.isfinite(pitch) or not 1e-12<=pitch<=1e12:raise ValueError('Bounded real pitch required')
        c=np.array(corners,dtype=np.float64,order='C',copy=True);a=np.array(anchor,dtype=np.float64,copy=True);q=np.array(xy,dtype=np.float64,order='C',copy=True)
        if c.ndim!=3 or c.shape[1:]!=(3,3) or not 1<=len(c)<=1000000 or not np.isfinite(c).all() or np.max(np.abs(c))>1e12:raise ValueError('Bounded finite triangles required')
        if a.shape!=(3,) or not np.isfinite(a).all() or np.max(np.abs(a))>1e12:raise ValueError('Bounded finite anchor required')
        if q.shape!=(self.shape[0]*self.shape[1],2) or not np.isfinite(q).all() or np.max(np.abs(q))>1e12:raise ValueError('Bounded matching query grid required')
        zs=np.ascontiguousarray(a[2]+(np.arange(self.shape[2],dtype=np.float64)+VOXEL_PROVPUNKT)*float(pitch))
        self._owner=threading.get_ident();self._closed=False;self._failed=False;self._handle=ct.c_void_p();self._lib=ct.CDLL(str(library))
        p=ct.c_void_p;i=ct.c_int
        self._lib.fused_create.argtypes=[ct.c_char_p,p,ct.c_uint32,p,p,i,i,i,ct.c_double,p,ct.POINTER(p)];self._lib.fused_create.restype=i
        self._lib.fused_query.argtypes=[p,p,p,p];self._lib.fused_query.restype=i
        self._lib.fused_destroy.argtypes=[ct.POINTER(p)];self._lib.fused_destroy.restype=i
        status=self._lib.fused_create(str(ptx).encode(),c.ctypes.data,len(c),a.ctypes.data,q.ctypes.data,*self.shape,float(pitch),zs.ctypes.data,ct.byref(self._handle))
        if status:raise RuntimeError(f'Fused field creation status {status}')

    def query(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('Field belongs to another thread')
        if self._closed or self._failed:raise RuntimeError('Field closed or failed')
        solid=np.empty(self.shape,np.uint8);surface=np.empty(self.shape,np.uint8);distance=np.empty(self.shape,np.float32)
        status=self._lib.fused_query(self._handle,solid.ctypes.data,surface.ctypes.data,distance.ctypes.data)
        if status:self._failed=True;raise RuntimeError(f'Fused field status {status}')
        return solid.view(np.bool_),surface.view(np.bool_),distance

    def close(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('Field belongs to another thread')
        if not self._closed:
            self._closed=True;status=self._lib.fused_destroy(ct.byref(self._handle))
            if status:raise RuntimeError(f'Fused field close status {status}')


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_mesh_fused_probe import main
    raise SystemExit(main())
