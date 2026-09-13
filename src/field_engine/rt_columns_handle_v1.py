"""Same-thread bounded custom RT columns; no implicit files or initialization."""
import ctypes as ct
import operator
import threading
import numpy as np
import faltkarna_v1_mesh_to_sdf as baseline

def occupancy(c,z,s,pitch,origin,shape):
    result=np.zeros(shape,bool)
    if not len(c):return result
    order=np.lexsort((z,c));c,z,s=c[order],z[order],s[order]
    counts=np.bincount(c,minlength=shape[0]*shape[1]);ends=np.cumsum(counts)
    cum=np.r_[np.int64(0),np.cumsum(s,dtype=np.int64)]
    low=float(z.min());span=max(float(z.max())-low,1e-12)
    keys=c.astype(np.float64)+.25+.5*(z-low)/span
    zs=origin[2]+(np.arange(shape[2],dtype=np.float64)+baseline.VOXEL_PROVPUNKT)*pitch
    active=np.flatnonzero(counts);cc=np.repeat(active,shape[2]);zz=np.tile(zs,len(active))
    query=cc.astype(np.float64)+np.clip(.25+.5*(zz-low)/span,0,.999)
    pos=np.searchsorted(keys,query,side='right')
    result.reshape(-1,shape[2])[active]=(cum[ends[cc]]-cum[pos]!=0).reshape(len(active),shape[2])
    return result


class ColumnsHandle:
    def __init__(self,library,ptx,corners,anchor,capacity,hit_capacity=64):
        self._owner=threading.get_ident();self._closed=False;self._failed=False;self._handle=ct.c_void_p()
        c=np.array(corners,dtype=np.float64,order='C',copy=True);a=np.array(anchor,dtype=np.float64,order='C',copy=True)
        if c.ndim!=3 or c.shape[1:]!=(3,3) or not 1<=len(c)<=1000000 or not np.isfinite(c).all() or np.max(np.abs(c))>1e12:raise ValueError('bounded finite corners required')
        if a.shape!=(3,) or not np.isfinite(a).all() or np.max(np.abs(a))>1e12:raise ValueError('bounded finite anchor required')
        if isinstance(capacity,(bool,np.bool_)) or isinstance(hit_capacity,(bool,np.bool_)):raise ValueError('integer capacities required')
        self.capacity=operator.index(capacity);self.hit_capacity=operator.index(hit_capacity)
        if not 1<=self.capacity<=262144 or not 1<=self.hit_capacity<=256:raise ValueError('capacity outside contract')
        self._lib=ct.CDLL(str(library));p=ct.c_void_p;u=ct.c_uint32
        self._lib.columns_create.argtypes=[ct.c_char_p,p,u,u,u,p,ct.POINTER(p)];self._lib.columns_create.restype=ct.c_int
        self._lib.columns_query.argtypes=[p,p,u,p,p,p];self._lib.columns_query.restype=ct.c_int
        self._lib.columns_destroy.argtypes=[ct.POINTER(p)];self._lib.columns_destroy.restype=ct.c_int
        status=self._lib.columns_create(str(ptx).encode(),c.ctypes.data,len(c),self.capacity,self.hit_capacity,a.ctypes.data,ct.byref(self._handle))
        if status:raise RuntimeError(f'column creation status {status}')

    def _check(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('columns belong to another thread')
        if self._closed:raise RuntimeError('columns are closed')
        if self._failed:raise RuntimeError('column query failed; no retry')

    def query(self,xy):
        self._check();q=np.array(xy,dtype=np.float64,order='C',copy=True)
        if q.ndim!=2 or q.shape[1]!=2 or not 1<=len(q)<=self.capacity or not np.isfinite(q).all() or np.max(np.abs(q))>1e12:raise ValueError('bounded finite query XY required')
        counts=np.empty(len(q),np.int32);z=np.empty((len(q),self.hit_capacity),np.float64);s=np.empty(z.shape,np.int32)
        status=self._lib.columns_query(self._handle,q.ctypes.data,len(q),counts.ctypes.data,z.ctypes.data,s.ctypes.data)
        if status:
            self._failed=True
            raise RuntimeError(f'column query status {status}')
        valid=np.arange(self.hit_capacity)[None,:]<counts[:,None]
        c=np.broadcast_to(np.arange(len(q))[:,None],valid.shape)[valid];z=z[valid];s=s[valid]
        order=np.lexsort((s,z,c));return c[order],z[order],s[order]

    def close(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('columns belong to another thread')
        if not self._closed:
            self._closed=True
            status=self._lib.columns_destroy(ct.byref(self._handle))
            if status:raise RuntimeError(f'column destroy status {status}')

    def __enter__(self):self._check();return self
    def __exit__(self,*args):self.close()
