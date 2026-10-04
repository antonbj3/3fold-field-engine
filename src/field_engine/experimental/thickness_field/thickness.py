"""Thickness between triangulated surfaces. Coordinates and answers are in mm.

Ray calls are isolated in _ray_hits: an RT backend can replace that function while
keeping the point, direction, t and validity contract. Meshes are (V,F) arrays.
"""
import numpy as np
import trimesh
from trimesh.proximity import closest_point


def _as_mesh(surface):
    if isinstance(surface, trimesh.Trimesh):
        return surface
    v, f = surface
    return trimesh.Trimesh(vertices=np.asarray(v, np.float64),
                           faces=np.asarray(f, np.int64), process=False)


def _ray_hits(mesh, origins, directions):
    """Return positive first-hit distances in mm, NaN for misses."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    engine = RayMeshIntersector(mesh)
    positions, index_ray, _ = engine.intersects_location(
        np.ascontiguousarray(origins, np.float64),
        np.ascontiguousarray(directions, np.float64), multiple_hits=False)
    answer = np.full(len(origins), np.nan)
    if len(index_ray):
        answer[index_ray] = np.linalg.norm(positions-origins[index_ray], axis=1)
    return answer


def thickness_field(surface_A, surface_B, method='normal_ray', points=None,
                    directions=None):
    """Sample surface A at face centroids unless points are provided.

    method='normal_ray' follows supplied directions, or A face normals.
    method='nearest' returns Euclidean closest-point distance to B.
    Return dict with points, t (mm), and valid mask. All inputs are mm and ray
    normals must point A→B. Valid for triangulated surfaces with a well-defined
    first ray hit or nearest point. No numerical or physical error bound is
    returned; RT device behavior is outside the verified regime.
    """
    a, b = _as_mesh(surface_A), _as_mesh(surface_B)
    p = np.asarray(a.triangles_center if points is None else points, np.float64)
    if method == 'nearest':
        target, t, _ = closest_point(b, p)
        return {'points':p, 't':t, 'valid':np.isfinite(t), 'target':target}
    if method != 'normal_ray':
        raise ValueError("method must be 'normal_ray' or 'nearest'")
    d = np.asarray(a.face_normals if directions is None else directions, np.float64)
    if len(d) != len(p):
        raise ValueError('one direction per point required')
    d = d / np.linalg.norm(d, axis=1)[:, None]
    t = _ray_hits(b, p, d)
    return {'points':p, 't':t, 'valid':np.isfinite(t)}


def moller_batch(origins, directions, triangles, dtype=np.float64):
    """Möller–Trumbore for one triangle per ray; positive first hit, NaN miss."""
    o = np.asarray(origins, dtype=dtype)
    d = np.asarray(directions, dtype=dtype)
    tri = np.asarray(triangles, dtype=dtype)
    e1, e2 = tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]
    h = np.cross(d,e2)
    det = np.sum(e1*h,axis=1)
    safe = np.where(np.abs(det)>np.finfo(dtype).eps*8,det,np.nan)
    inv = 1/safe
    s = o-tri[:,0]
    u = inv*np.sum(s*h,axis=1)
    q = np.cross(s,e1)
    v = inv*np.sum(d*q,axis=1)
    t = inv*np.sum(e2*q,axis=1)
    return np.where((u>=0)&(v>=0)&(u+v<=1)&(t>=0),t,np.nan)
