"""Require ordered hits and preserve the frozen composite fractions and native mask."""
import ctypes as ct
import operator
import threading
import numpy as np
from faltkarna_v1_mesh_to_sdf import VOXEL_PROVPUNKT


class OrderedColumnMask:
    def __init__(self,library):
        self._owner=threading.get_ident();self._failed=False;self._lib=ct.CDLL(str(library))
        p=ct.c_void_p;i=ct.c_int
        self._lib.column_mask_surface.argtypes=[p,p,i,p,p,i,i,i,p,p];self._lib.column_mask_surface.restype=i

    def evaluate(self,c,z,s,pitch,origin,shape):
        if threading.get_ident()!=self._owner:raise RuntimeError('column mask belongs to another thread')
        if self._failed:raise RuntimeError('column mask failed; no retry')
        shape=tuple(operator.index(x) for x in shape)
        if len(shape)!=3 or min(shape)<1 or max(shape)>512 or np.prod(shape)>1048576:raise ValueError('bounded shape required')
        c=np.asarray(c);z=np.asarray(z);s=np.asarray(s);origin=np.asarray(origin)
        if c.ndim!=1 or z.shape!=c.shape or s.shape!=c.shape or len(c)>1000000:raise ValueError('matching bounded hit vectors required')
        if not np.issubdtype(c.dtype,np.integer) or not np.issubdtype(s.dtype,np.integer):raise ValueError('integer columns/signs required')
        if not np.isfinite(z).all() or np.any((s!=1)&(s!=-1)):raise ValueError('finite hits with signed orientations required')
        if len(c) and (c.min()<0 or c.max()>=shape[0]*shape[1]):raise ValueError('column outside grid')
        if origin.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(pitch) or not 1e-12<=pitch<=1e12:raise ValueError('finite grid required')
        if np.any(c[1:]<c[:-1]) or np.any((c[1:]==c[:-1]) & (z[1:]<z[:-1])):
            raise ValueError('Hits must be ordered by column then depth')
        z=z.astype(np.float64);signs=np.ascontiguousarray(s,np.int32)
        counts=np.bincount(c,minlength=shape[0]*shape[1]);offsets=np.ascontiguousarray(np.r_[0,np.cumsum(counts)],np.int32)
        low=float(z.min()) if len(z) else 0.;span=max(float(z.max())-low,1e-12) if len(z) else 1.
        keys=np.ascontiguousarray(c.astype(np.float64)+.25+.5*(z-low)/span)
        zs=origin[2]+(np.arange(shape[2],dtype=np.float64)+VOXEL_PROVPUNKT)*pitch
        fractions=np.ascontiguousarray(np.clip(.25+.5*(zs-low)/span,0,.999))
        solid=np.empty(shape,np.uint8);surface=np.empty(shape,np.uint8)
        status=self._lib.column_mask_surface(keys.ctypes.data,signs.ctypes.data,len(c),offsets.ctypes.data,fractions.ctypes.data,*shape,solid.ctypes.data,surface.ctypes.data)
        if status:
            self._failed=status!=1
            raise RuntimeError(f'column mask status {status}')
        return solid.view(np.bool_),surface.view(np.bool_)


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_rowwise_stage_probe import main
    raise SystemExit(main())
