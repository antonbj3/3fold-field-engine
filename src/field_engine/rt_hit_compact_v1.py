"""Explicit CPU two-hit compaction; general rows retain the original sorter."""
import ctypes as ct
import numpy as np
from rt_columns_two_hit_handle_v1 import sort_rows as original_sort


class NativeHitSorter:
    def __init__(self,library):
        self._library=ct.CDLL(str(library))
        self._call=self._library.compact_two_hits
        p=ct.c_void_p
        self._call.argtypes=[p,p,p,ct.c_int64,ct.c_int,p,p,p,ct.POINTER(ct.c_int64)]
        self._call.restype=ct.c_int

    def sort_rows(self,counts,z,signs):
        counts,z,signs=np.asarray(counts),np.asarray(z),np.asarray(signs)
        if counts.ndim!=1 or z.ndim!=2 or signs.shape!=z.shape or len(z)!=len(counts) or not np.issubdtype(counts.dtype,np.integer):
            raise ValueError('Matching row counts, depths and signs required')
        if z.shape[1]>2 or len(counts)>262144 or counts.dtype!=np.int32 or z.dtype!=np.float64 or signs.dtype!=np.int32:
            return original_sort(counts,z,signs)
        counts,z,signs=(np.ascontiguousarray(a) for a in (counts,z,signs))
        capacity=z.size
        columns=np.empty(capacity,np.int64);depths=np.empty(capacity,np.float64);orientations=np.empty(capacity,np.int32)
        written=ct.c_int64()
        status=self._call(counts.ctypes.data,z.ctypes.data,signs.ctypes.data,len(counts),z.shape[1],
                          columns.ctypes.data,depths.ctypes.data,orientations.ctypes.data,ct.byref(written))
        if status:raise ValueError('Native row counts or capacity invalid')
        n=written.value
        return columns[:n],depths[:n],orientations[:n]
