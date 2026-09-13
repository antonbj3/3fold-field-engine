"""Direct generalized winding with fixed face order and explicit float64 inputs.

No compute context or artifact writes. Triangle surfaces need not be closed.
Points exactly on vertices are outside this numeric contract. This is a direct
CPU solid-angle sum, not an RT-core or fast hierarchical GWN algorithm.
"""
import numpy as np


def generalized_winding(vertices,faces,points):
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces);q=np.asarray(points,dtype=np.float64)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():raise ValueError('finite vertices required')
    if f.ndim!=2 or f.shape[1]!=3 or not len(f) or not np.issubdtype(f.dtype,np.integer):raise ValueError('integral triangles required')
    if f.min()<0 or f.max()>=len(v):raise ValueError('triangle index outside vertices')
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():raise ValueError('finite query points required')
    tri=v[f]
    area=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    if np.any(np.sum(area*area,axis=1)==0):raise ValueError('degenerate triangle')
    result=np.empty(len(q),np.float64)
    for start in range(0,len(q),4096):
        x=q[start:start+4096];total=np.zeros(len(x),np.float64)
        for triangle in tri:
            a,b,c=(vertex-x for vertex in triangle)
            la=np.sqrt(np.einsum('ij,ij->i',a,a));lb=np.sqrt(np.einsum('ij,ij->i',b,b));lc=np.sqrt(np.einsum('ij,ij->i',c,c))
            if np.any(la==0) or np.any(lb==0) or np.any(lc==0):raise ValueError('query coincides with a vertex')
            numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
            denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
            total+=np.arctan2(numerator,denominator)
        result[start:start+len(x)]=total/(2*np.pi)
    return result


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_gwn_ordered_probe import main
    raise SystemExit(main())
