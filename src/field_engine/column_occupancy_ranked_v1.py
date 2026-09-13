"""Exact composite-key signing by hit rank; no native state or artifact writes."""
import numpy as np
from faltkarna_v1_mesh_to_sdf import VOXEL_PROVPUNKT


def occupancy(c,z,s,pitch,origin,shape):
    result=np.zeros(shape,bool)
    if not len(c):return result
    order=np.lexsort((z,c));c,z,s=c[order],z[order],s[order]
    counts=np.bincount(c,minlength=shape[0]*shape[1]);starts=np.cumsum(counts)-counts
    active=np.flatnonzero(counts);low=float(z.min());span=max(float(z.max())-low,1e-12)
    hitkeys=c.astype(np.float64)+.25+.5*(z-low)/span
    zs=origin[2]+(np.arange(shape[2],dtype=np.float64)+VOXEL_PROVPUNKT)*pitch
    query=active.astype(np.float64)[:,None]+np.clip(.25+.5*(zs-low)/span,0,.999)[None,:]
    sums=np.zeros(query.shape,np.int64)
    for rank in range(int(counts.max())):
        selected=counts[active]>rank
        positions=starts[active[selected]]+rank
        sums[selected]+=s[positions,None]*(hitkeys[positions,None]>query[selected])
    result.reshape(-1,shape[2])[active]=sums!=0
    return result
