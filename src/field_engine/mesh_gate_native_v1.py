"""Optional complete-gate primitives; explicit build and no implicit file writes.

Build mesh_gate_native_v1/gate.cpp with -std=c++17 -O3 -fopenmp
-fno-fast-math -ffp-contract=off -shared -fPIC, set FIELD_MESH_GATE_LIBRARY.
Trimesh keeps vertex merging, volume, closest-point arithmetic and tie rules.
Topology is exact integer counting. Independent ray parity keeps the original
thresholds; close floating comparisons are rerun by the reference.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import numpy as np

_LOADED = {}


def lib():
    name = os.environ.get('FIELD_MESH_GATE_LIBRARY')
    if not name:
        raise RuntimeError('FIELD_MESH_GATE_LIBRARY is not set; explicitly build mesh_gate_native_v1/gate.cpp')
    path = Path(name).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = str(path)
    if key not in _LOADED:
        L = ctypes.CDLL(key)
        P, I = ctypes.c_void_p, ctypes.c_int64
        L.meshgate_edges.argtypes = [P, I, I, P]
        L.meshgate_edges.restype = ctypes.c_int
        L.meshgate_parity.argtypes = [P, I, P, I, P, I, ctypes.c_int, P, P]
        L.meshgate_parity.restype = ctypes.c_int
        L.meshgate_tree_new.argtypes = [P, I, P, I]
        L.meshgate_tree_new.restype = P
        L.meshgate_tree_free.argtypes = [P]
        L.meshgate_tree_free.restype = None
        L.meshgate_tree_query.argtypes = [P, P, I, P, P, I]
        L.meshgate_tree_query.restype = ctypes.c_int
        _LOADED[key] = digest, L
    old, L = _LOADED[key]
    if old != digest:
        raise RuntimeError('Restart the process after replacing the native mesh-gate library')
    return L, digest


def arrays(V, T):
    v, t = np.asarray(V), np.asarray(T)
    if v.ndim != 2 or v.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 3:
        raise RuntimeError('vertices and faces must have shape (n, 3)')
    if not np.issubdtype(t.dtype, np.integer) or not np.isfinite(v).all():
        raise RuntimeError('finite vertices and integer faces required')
    if not (0 < len(v) < 2**31 and len(t) <= 10_000_000):
        raise RuntimeError('unsupported mesh size')
    if t.size and (t.min() < 0 or t.max() >= len(v)):
        raise RuntimeError('face index outside vertex array')
    return np.ascontiguousarray(v, np.float64), np.ascontiguousarray(t, np.int64)


def edges(L, t, nv):
    bad = np.zeros(1, np.int64)
    rc = L.meshgate_edges(t.ctypes.data, nv, len(t), bad.ctypes.data)
    if rc:
        raise RuntimeError(f'native edge count refused call: {rc}')
    return bool(len(t) and bad[0] == 0), int(bad[0])


def volume(m):
    """Trimesh's volume integral, same operations/reduction; no unused moments."""
    tri = np.asarray(m.triangles)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 1]
    cross_x = e1[:, 1]*e2[:, 2] - e1[:, 2]*e2[:, 1]
    f1_x = tri[:, 0, 0] + tri[:, 1, 0] + tri[:, 2, 0]
    return float((cross_x*f1_x).sum()/6.0)


def facts(V, T):
    import trimesh
    v, t = arrays(V, T)
    if len(t) == 0:
        raise RuntimeError("empty-face semantics depend on trimesh version; use the reference")
    L, digest = lib()
    raw_w, raw_bad = edges(L, t, len(v))
    m = trimesh.Trimesh(v, t, process=False)
    m.merge_vertices()
    mv, mt = arrays(m.vertices, m.faces)
    watertight, bad = edges(L, mt, len(mv))
    return m, watertight, bad, volume(m), raw_w, raw_bad, digest


def parity(V, T, P):
    import faltkarna_v1_mesh_to_sdf as M
    v, t = arrays(V, T)
    p = np.ascontiguousarray(P, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or len(p) > 4096:
        raise RuntimeError('unsupported probe points')
    L, _ = lib()
    out = np.empty(len(p), np.uint8)
    uncertain = np.empty(len(p), np.uint8)
    rc = L.meshgate_parity(v.ctypes.data, len(v), t.ctypes.data, len(t), p.ctypes.data,
                          len(p), 4, out.ctypes.data, uncertain.ctypes.data)
    if rc:
        raise RuntimeError(f'native parity refused call: {rc}')
    out = out.view(bool)
    mask = uncertain.view(bool)
    if mask.any():
        out[mask] = M.ray_parity_probe(v, t, p[mask])
    return out


class _Tree:
    def __init__(self, m):
        self.L, _ = lib()
        v, t = arrays(m.vertices, m.faces)
        self.handle = self.L.meshgate_tree_new(v.ctypes.data, len(v), t.ctypes.data, len(t))
        if not self.handle:
            raise RuntimeError('native proximity tree refused mesh')

    def close(self):
        if self.handle:
            self.L.meshgate_tree_free(self.handle)
            self.handle = None

    def intersection(self, bounds):
        b = np.asarray(bounds, np.float64).reshape(1, 6)
        return self.intersection_v(b[:, :3], b[:, 3:])[0].tolist()

    def intersection_v(self, lo, hi):
        boxes = np.ascontiguousarray(np.column_stack((lo, hi)), np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 6:
            raise RuntimeError('proximity query boxes must have shape (n, 6)')
        off = np.empty(len(boxes)+1, np.int64)
        rc = self.L.meshgate_tree_query(self.handle, boxes.ctypes.data, len(boxes), off.ctypes.data, None, 0)
        if rc or off[-1] > 64_000_000:
            raise RuntimeError(f'native proximity candidate count refused: {rc}')
        ids = np.empty(int(off[-1]), np.int64)
        rc = self.L.meshgate_tree_query(self.handle, boxes.ctypes.data, len(boxes), off.ctypes.data,
                                       ids.ctypes.data, len(ids))
        if rc:
            raise RuntimeError(f'native proximity query refused: {rc}')
        return ids, np.diff(off)


class _MeshView:
    def __init__(self, m, tree):
        self._m = m
        self.triangles_tree = tree

    def __getattr__(self, name):
        return getattr(self._m, name)


def distances(m, P, pitch):
    import trimesh
    import rtree  # Preserve the reference depth fallback if its optional dependency is absent.
    tree = _Tree(m)
    try:
        _, dist, _ = trimesh.proximity.closest_point(_MeshView(m, tree), P)
    finally:
        tree.close()
    # Trimesh may pick the second closest face when squared distances differ
    # by less than tol.merge. Candidate iteration order can therefore affect a
    # depth threshold only in this band; preserve the reference's rtree order there.
    scale = np.maximum(dist**2, pitch**2)
    near = np.abs(dist**2-pitch**2) <= trimesh.constants.tol.merge + 64*np.finfo(float).eps*scale
    if near.any():
        # Keep the complete original query order, including batch rtree ordering.
        _, reference_distance, _ = trimesh.proximity.closest_point(m, P)
        dist[near] = reference_distance[near]
    return dist
