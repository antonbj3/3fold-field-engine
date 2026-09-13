"""Prepared closed-mesh exact CUDA scan columns or open-mesh GWN, followed by exact native EDT.

Caller supplies native libraries and owns hardware coordination. Import performs
no compute initialization or artifact writes. Preparation binds immutable geometry
and grid, not computed outputs; every evaluate recomputes signing and distances.
"""
import operator
import threading
from numbers import Real
import numpy as np
import trimesh
from scipy import ndimage
import faltkarna_v1_mesh_to_sdf as baseline
from generalized_winding_ordered_v1 import generalized_winding
from cuda_columns_handle_v1 import ColumnsHandle
from column_mask_native_v1 import NativeColumnMask
from native_edt_large_v1 import NativeEDTLarge


class PreparedMeshField:
    def __init__(self,vertices,faces,pitch,lo,*,columns_library,edt_library,column_library,global_shape=None,margin_vox_pad=2):
        v=np.array(vertices,dtype=np.float64,copy=True);t=np.array(faces,copy=True);lo=np.array(lo,dtype=np.float64,copy=True)
        if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():raise ValueError('finite vertices required')
        if t.ndim!=2 or t.shape[1]!=3 or not len(t) or not np.issubdtype(t.dtype,np.integer):raise ValueError('integral faces required')
        if t.min()<0 or t.max()>=len(v):raise ValueError('face index outside vertices')
        if lo.shape!=(3,) or not np.isfinite(lo).all():raise ValueError('finite three-coordinate origin required')
        if isinstance(pitch,(bool,np.bool_)) or not isinstance(pitch,Real) or not np.isfinite(pitch) or not 1e-12<=pitch<=1e12:raise ValueError('bounded real pitch required')
        if isinstance(margin_vox_pad,(bool,np.bool_)):raise ValueError('integer padding required')
        margin_vox_pad=operator.index(margin_vox_pad)
        if not 0<=margin_vox_pad<=512:raise ValueError('padding outside contract')
        if global_shape is not None:
            raw=tuple(global_shape)
            if len(raw)!=3 or any(isinstance(x,(bool,np.bool_)) for x in raw):raise ValueError('three integer global dimensions required')
            global_shape=tuple(operator.index(x) for x in raw)
            if min(global_shape)<1 or max(global_shape)>1000000000:raise ValueError('global grid outside index contract')
        relative=(v-lo)/float(pitch)
        if not np.isfinite(relative).all() or np.max(np.abs(relative))>1e12:raise ValueError('grid coordinates exceed bounded index contract')
        gmin,shape=baseline._fonster(v,float(pitch),lo,global_shape,margin_vox_pad)
        if min(shape)<1 or max(shape)>512 or np.prod(shape)>1048576:raise ValueError('local grid outside native EDT capacity')
        anchor=lo+gmin.astype(np.float64)*float(pitch)
        mesh=trimesh.Trimesh(v,t,process=False)
        if np.any(mesh.area_faces==0):raise ValueError('degenerate triangle')
        self.method='cuda_columns' if mesh.is_watertight and mesh.is_winding_consistent else 'ordered_gwn'
        self._anchor=anchor;self._vertices=v-anchor;self._faces=t.astype(np.int64);self._pitch=float(pitch)
        self._gmin=gmin;self.shape=shape;self._owner=threading.get_ident();self._closed=False;self._failed=False;self._rt=None
        indices=np.indices(shape).reshape(3,-1).T
        # Preserve the frozen world-grid sampling, then subtract in float64.
        points=anchor+indices*self._pitch
        points[:,0]+=self._pitch*baseline.JITTER_FRAC*baseline._GYLLENE[0]
        points[:,1]+=self._pitch*baseline.JITTER_FRAC*baseline._GYLLENE[1]
        self._points=points-anchor
        for a in (self._vertices,self._faces,self._points,self._gmin):a.flags.writeable=False
        self._edt=NativeEDTLarge(edt_library)
        self._mask=NativeColumnMask(column_library)
        if self.method=='cuda_columns':
            corners=np.ascontiguousarray(v[self._faces],dtype=np.float64)
            ij=np.indices(shape[:2]).reshape(2,-1).T
            self._xy=anchor[:2]+(ij+baseline.VOXEL_PROVPUNKT)*self._pitch
            self._xy[:,0]+=self._pitch*baseline.JITTER_FRAC*baseline._GYLLENE[0]
            self._xy[:,1]+=self._pitch*baseline.JITTER_FRAC*baseline._GYLLENE[1]
            self._xy.flags.writeable=False
            self._rt=ColumnsHandle(columns_library,corners,anchor,len(self._xy))

    def _check(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('field belongs to another thread')
        if self._closed:raise RuntimeError('field is closed')
        if self._failed:raise RuntimeError('field evaluation failed; no retry')

    def evaluate(self):
        self._check()
        try:
            if self.method=='cuda_columns':
                solid,surface=self._mask.evaluate(*self._rt.query(self._xy),self._pitch,self._anchor,self.shape)
            else:
                solid=(np.abs(generalized_winding(self._vertices,self._faces,self._points))>.5).reshape(self.shape)
                surface=solid & ~ndimage.binary_erosion(solid,ndimage.generate_binary_structure(3,1))
            distance=self._edt.signed_distance(solid,self._pitch)
            return self._gmin.copy(),self.shape,surface,solid,distance
        except (RuntimeError,ValueError):
            self._failed=True
            raise

    def close(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('field belongs to another thread')
        if not self._closed:
            if self._rt is not None:self._rt.close()
            self._closed=True

    def __enter__(self):self._check();return self
    def __exit__(self,*args):self.close()


def surface_raster_and_flood_cuda(vertices,faces,pitch,lo,*,columns_library,edt_library,column_library,global_shape=None,margin_vox_pad=2):
    """One-shot stage including preparation/close; no hidden global cache."""
    with PreparedMeshField(vertices,faces,pitch,lo,columns_library=columns_library,edt_library=edt_library,column_library=column_library,
                           global_shape=global_shape,margin_vox_pad=margin_vox_pad) as field:
        return field.evaluate()


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_cuda_columns_probe import main
    raise SystemExit(main())
