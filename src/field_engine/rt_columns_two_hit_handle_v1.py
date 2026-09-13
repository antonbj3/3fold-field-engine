"""Exact two-hit compare/swap with a general rowwise-sort fallback."""
import ctypes as ct
import numpy as np
from rt_columns_rowwise_handle_v1 import sort_rows as reference_sort
from rt_columns_compact_handle_v1 import ColumnsHandle as FullColumnsHandle


def sort_rows(counts,z,signs):
    counts=np.asarray(counts);z=np.asarray(z);signs=np.asarray(signs)
    if counts.ndim!=1 or z.ndim!=2 or signs.shape!=z.shape or len(z)!=len(counts) or not np.issubdtype(counts.dtype,np.integer):
        raise ValueError('Matching row counts, depths and signs required')
    if np.any(counts<0) or np.any(counts>z.shape[1]):
        raise ValueError('Count exceeds row width')
    width=z.shape[1]
    if width>2:return reference_sort(counts,z,signs)
    valid=np.arange(width)[None,:]<counts[:,None]
    if width==2:
        swap=(counts==2) & ((z[:,0]>z[:,1]) | ((z[:,0]==z[:,1]) & (signs[:,0]>signs[:,1])))
        depths=z.copy();orientations=signs.copy()
        depths[:,0]=np.where(swap,z[:,1],z[:,0]);depths[:,1]=np.where(swap,z[:,0],z[:,1])
        orientations[:,0]=np.where(swap,signs[:,1],signs[:,0]);orientations[:,1]=np.where(swap,signs[:,0],signs[:,1])
    else:
        depths=z;orientations=signs
    columns=np.broadcast_to(np.arange(len(counts))[:,None],valid.shape)[valid]
    return columns,depths[valid],orientations[valid]


class ColumnsHandle(FullColumnsHandle):
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
        return sort_rows(counts,z,s)


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_two_hit_stage_probe import main
    raise SystemExit(main())
