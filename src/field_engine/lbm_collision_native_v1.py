"""Prepared D3Q19 CPU step with ordered native collision and indexed streaming.

Compile lbm_collision_native_v1/collision.cpp with -O3 -fno-fast-math
-ffp-contract=off -shared -fPIC, then set FIELD_LBM_COLLISION_LIBRARY.
The context owns fixed geometry/forcing and accepts contiguous float64 states.
The original lbm_domare_v1 implementation remains the reference and default.
"""
import copy
import ctypes
import hashlib
import os
from pathlib import Path
import numpy as np
import lbm_domare_v1 as reference


_LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LOADED_LIBRARIES = {}


def _bound_library(library=None):
    """Keep the loaded binary and its identity together; refuse replacement in-process."""
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _LOADED_SOURCE_SHA256:
        raise RuntimeError('Reload the native LBM module after a source change')
    path = Path(library or os.environ['FIELD_LBM_COLLISION_LIBRARY']).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = str(path)
    if key not in _LOADED_LIBRARIES:
        handle = ctypes.CDLL(key)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Native LBM library changed while loading')
        _LOADED_LIBRARIES[key] = (digest, handle)
    loaded_digest, handle = _LOADED_LIBRARIES[key]
    if digest != loaded_digest:
        raise RuntimeError('Restart the process after replacing a native LBM library')
    return handle, dict(source_sha256=_LOADED_SOURCE_SHA256, library_sha256=loaded_digest)


def implementation_binding(library=None):
    return _bound_library(library)[1]


class PreparedStep:
    def __init__(self, shape, solid, tau, a_field, bc_planes, library=None):
        self.shape = tuple(shape)
        if len(self.shape) != 3 or any(n <= 0 for n in self.shape):
            raise ValueError('Three positive grid dimensions required')
        self.tau = float(tau)
        if not np.isfinite(self.tau) or self.tau == 0:
            raise ValueError('Finite nonzero relaxation time required')
        self.pref = 1 - 1 / (2 * self.tau)
        self.solid = np.array(solid, dtype=bool, order='C', copy=True)
        self.force = np.array(a_field, dtype=np.float64, order='C', copy=True)
        if self.solid.shape != self.shape or self.force.shape != self.shape + (3,):
            raise ValueError('Geometry and forcing must match the grid')
        self.boundaries = copy.deepcopy(bc_planes)
        self.ea = self.force @ reference.E.T
        self.weights = reference.W.copy()
        cells = np.arange(np.prod(self.shape), dtype=np.intp).reshape(self.shape)
        self.stream = np.empty(self.shape + (reference.QN,), dtype=np.intp)
        for q in range(reference.QN):
            self.stream[..., q] = np.roll(cells, reference.Ei[q], axis=(0, 1, 2)) * reference.QN + q
        # Solid cells always reflect their own post-collision populations. Fold this
        # fixed selection into the gather map once, avoiding a second copy each step.
        self.stream[self.solid] = cells[self.solid][:, None] * reference.QN + reference.OPP
        self._library, self.implementation = _bound_library(library)
        self._collision = self._library.field_lbm_collision
        self._collision.argtypes = [ctypes.c_int64, ctypes.c_double, ctypes.c_double] + [ctypes.POINTER(ctypes.c_double)] * 8
        self._collision.restype = ctypes.c_int
        self.stencil = np.array(reference.E, dtype=np.float64, order='C', copy=True)
        self._moments = getattr(self._library, 'field_lbm_moments', None)
        if self._moments is not None:
            self._moments.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 9
            self._moments.restype = ctypes.c_int

    def _fields(self, f):
        if self._moments is not None:
            rho_safe = np.empty(self.shape)
            u = np.empty(self.shape + (3,))
            cu = np.empty(self.shape + (reference.QN,))
            usq = np.empty(self.shape + (1,))
            ua = np.empty_like(usq)
            arrays = [f, self.solid, self.force, self.stencil, rho_safe, u, cu, usq, ua]
            code = self._moments(rho_safe.size, *[a.ctypes.data for a in arrays])
            if code:
                raise RuntimeError('Native moments refused inputs: ' + str(code))
            return rho_safe, u, cu, usq, ua
        # Older collision-only libraries retain the original numerical path.
        rho = f.sum(-1)
        rho_safe = np.where(rho > 1e-9, rho, 1.0)
        u = (f @ reference.E + 0.5 * rho_safe[..., None] * self.force) / rho_safe[..., None]
        u[self.solid] = 0.0
        cu = u @ reference.E.T
        usq = (u ** 2).sum(-1, keepdims=True)
        ua = (u * self.force).sum(-1, keepdims=True)
        return rho_safe, u, cu, usq, ua

    def __call__(self, f):
        if f.shape != self.shape + (reference.QN,) or f.dtype != np.float64 or not f.flags.c_contiguous:
            raise ValueError('Contiguous float64 distribution matching the prepared grid required')
        rho_safe, u, cu, usq, ua = self._fields(f)
        fstar = np.empty_like(f)
        args = [f, rho_safe, cu, usq, self.ea, ua, self.weights, fstar]
        pointers = [a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for a in args]
        code = self._collision(rho_safe.size, self.tau, self.pref, *pointers)
        if code:
            raise RuntimeError('Native collision refused inputs: ' + str(code))
        fnew = np.take(fstar, self.stream)
        for bc in self.boundaries:
            idx, adj, rho_t = bc['idx'], bc['adj'], bc['rho']
            u_adj = u[adj]
            fnew[idx] = reference.feq3d(np.full(u_adj.shape[:-1], rho_t), u_adj)
        return fnew


def run_lbm(shp, solid, tau, a_field, bc_planes, steps, sample_every=200, tol=1e-7):
    step = PreparedStep(shp, solid, tau, a_field, bc_planes)
    f = reference.feq3d(np.ones(shp), np.zeros(shp + (3,)))
    last = 0.0
    s = 0
    for s in range(steps):
        f = step(f)
        if s % sample_every == 0 and s > 0:
            rho = f.sum(-1)
            um = float(np.abs((f @ reference.E)[..., 0] / np.where(rho > 1e-9, rho, 1.0)).max())
            if abs(um - last) < tol * max(um, 1e-30):
                break
            last = um
    return f, s
