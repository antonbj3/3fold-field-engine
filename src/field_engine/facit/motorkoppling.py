"""Couplings between the answer keys and the engine's own operators.

Each function runs an UNCHANGED engine operator on a body from the answer key and returns the
engine's number next to the exact one.  The only code here that is not the engine's is the
boundary closure, and it is named:

  lattice_conductance      engine graph Laplacian (opt/laplaceflode_v1.bygg_gitterlaplacian, unit
                           conductance) + Dirichlet electrodes half a cell outside the first node
  lattice_dirichlet_poisson  same Laplacian + u = 0 closure: 'stair' (at the outside neighbour's
                           centre) or 'cut' (at the exact crossing theta*h, symmetric closure of
                           Gibou et al. 2002, needs the body's signed distance or implicit function)
  hex8_torsion             engine Q1 hex8 core (lastfalt_v1_fem: hex8_ke, assemble_K_cpu, solve) on
                           a voxel shaft, torque about the voxel section's centroid, twist rate by
                           least squares in the middle fifth
  landreman_lattice        psi - delta of the Landreman torus sampled on the engine's lattice
                           convention (voxel i is the point lo + i*h)

Units follow the caller: the lattice functions work in cell units unless h is given.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import klassiska as K

_FE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_FE, os.path.join(_FE, "opt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _engine_laplacian(free):
    import laplaceflode_v1 as LF
    n, nodkarta, _u, _v, kond, L = LF.bygg_gitterlaplacian(~free, np.ones(free.shape), 1.0, p_kond=0.0)
    if not np.all(kond == 1.0):
        raise AssertionError("p_kond = 0 must give unit conductance")
    return n, nodkarta, L.tocsr()


def _solve(A, b):
    if A.shape[0] <= 300_000:
        return spla.spsolve(A.tocsc(), b)
    x, info = spla.cg(A, b, rtol=1e-12, maxiter=20000, M=sp.diags(1.0 / A.diagonal()))
    if info != 0:
        raise RuntimeError(f"cg info {info}")
    return x


def lattice_conductance(free, low_electrode, high_electrode):
    """Conductance between two electrode planes in units of sigma*h (multiply by h for sigma = 1).

    free: bool occupancy of conducting cells.  low/high_electrode: bool masks of the cells whose
    face lies on the electrode plane (potential 0 and 1)."""
    n, nk, L = _engine_laplacian(free)
    lo_n, hi_n = nk[low_electrode & free], nk[high_electrode & free]
    d = np.zeros(n)
    b = np.zeros(n)
    np.add.at(d, lo_n, 2.0)
    np.add.at(d, hi_n, 2.0)
    np.add.at(b, hi_n, 2.0)
    phi = _solve(L + sp.diags(d), b)
    return float(np.sum(2.0 * (1.0 - phi[hi_n])))


def lattice_dirichlet_poisson(inside, crossing, lo, shape, h, rhs=1.0, closure="stair"):
    """-lap u = rhs on cell centres where inside(P) is True, u = 0 outside.

    inside(P) -> bool for points P (..., dim); crossing(a, b) -> theta in (0, 1] of the boundary
    along a -> b (only used by closure 'cut').  Returns (integral of u, u, free mask)."""
    dim = len(shape)
    ax = [lo[i] + (np.arange(shape[i]) + 0.5) * h for i in range(dim)]
    P = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
    free = inside(P)
    n, nk, L = _engine_laplacian(free)
    d = np.zeros(n)
    for axis in range(dim):
        for sgn in (1, -1):
            nb = np.roll(free, -sgn, axis=axis).copy()
            edge = [slice(None)] * dim
            edge[axis] = -1 if sgn == 1 else 0
            nb[tuple(edge)] = False
            m = free & ~nb
            if closure == "stair":
                np.add.at(d, nk[m], 1.0)
            elif closure == "cut":
                step = np.zeros(dim)
                step[axis] = sgn * h
                th = np.clip(crossing(P[m], P[m] + step), 1e-3, 1.0)
                np.add.at(d, nk[m], 1.0 / th)
            else:
                raise ValueError(closure)
    u = _solve((L + sp.diags(d)) / (h * h), np.full(n, float(rhs)))
    return float(u.sum() * h ** dim), u, free


def bisect_crossing(sdf, iters=60):
    """theta where sdf changes sign along a -> b, given sdf(a) < 0 <= sdf(b)."""
    def f(a, b):
        t0, t1 = np.zeros(len(a)), np.ones(len(a))
        for _ in range(iters):
            tm = 0.5 * (t0 + t1)
            ins = sdf(a + tm[:, None] * (b - a)) < 0
            t0, t1 = np.where(ins, tm, t0), np.where(ins, t1, tm)
        return 0.5 * (t0 + t1)
    return f


def prandtl_torsion_lattice(sdf, J_exact, n_per_unit, ext=0.75, phase=(0.37, 0.61), closure="stair"):
    """J = 2 int phi, -lap phi = 2 on the lattice.  Returns (J, relative error)."""
    h = 1.0 / n_per_unit
    shape = (int(round(2 * ext / h)),) * 2
    lo = (-ext + 0.5 * h * phase[0], -ext + 0.5 * h * phase[1])
    I, _, _ = lattice_dirichlet_poisson(lambda P: sdf(P) < 0, bisect_crossing(sdf), lo, shape, h, 2.0, closure)
    return 2.0 * I, 2.0 * I / J_exact - 1.0


def hex8_torsion(inside2d, J_exact, d, E=1.0, nu=0.3, torque=1.0, phase=(0.37, 0.61), length_factor=4):
    """Engine hex8 twist rate of a voxel shaft against T / (G J).  Returns a dict."""
    import lastfalt_v1_fem as FEM
    half = int(math.ceil(0.75 * d)) + 1
    nxy, nz = 2 * half, length_factor * d
    xc = np.arange(nxy) - half + phase[0]
    yc = np.arange(nxy) - half + phase[1]
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    sec = inside2d(np.stack([X, Y], -1))
    ii, jj = np.nonzero(sec)
    elem_all, _ = FEM.build_grid(nxy, nxy, nz)
    sel = np.zeros((nxy, nxy, nz), bool)
    sel[ii, jj, :] = True
    en = elem_all[sel.ravel()]
    used, inv = np.unique(en, return_inverse=True)
    en_c = inv.reshape(en.shape)
    n_nodes = len(used)
    K_ = FEM.assemble_K_cpu(en_c, np.ones(len(en_c)), FEM.hex8_ke(1.0, 1.0, 1.0, E=1.0, nu=nu),
                            p=1.0, E0=E, n_nodes=n_nodes)
    per = (nxy + 1) * (nz + 1)
    gi, gj, gk = used // per, (used % per) // (nz + 1), used % (nz + 1)
    x = gi - half + phase[0] - 0.5 - float(np.mean(xc[ii]))
    y = gj - half + phase[1] - 0.5 - float(np.mean(yc[jj]))
    z = gk.astype(float)
    trib = np.zeros(n_nodes)
    top = z[en_c].max(axis=1) == nz
    for c in (4, 5, 6, 7):
        np.add.at(trib, en_c[top, c], 0.25)
    r2 = x * x + y * y
    k = torque / float(np.sum(trib * r2))
    f = np.zeros(3 * n_nodes)
    f[0::3], f[1::3] = -k * trib * y, k * trib * x
    fixed_nodes = np.nonzero(z == 0)[0]
    u = FEM.solve(K_, f, np.concatenate([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2]))
    ux, uy = u[0::3], u[1::3]
    zs = np.arange(nz + 1)
    th = np.array([np.sum(x[z == q] * uy[z == q] - y[z == q] * ux[z == q]) / np.sum(r2[z == q]) for q in zs])
    m = (zs >= 1.6 * d) & (zs <= 2.4 * d)
    rate = float(np.polyfit(zs[m], th[m], 1)[0])
    exact = K.torsion_twist_rate(torque, E, nu, J_exact)
    return {"rate": rate, "rate_exact": exact, "J_eff_rel": exact / rate - 1.0,
            "section_area_cells": float(sec.sum()), "dofs": int(3 * n_nodes - 3 * len(fixed_nodes))}


def landreman_lattice(fam, h, phase=(0.0, 0.0, 0.0), pad=2):
    """psi - delta on the lattice lo + i*h (engine convention); +1 where the formulas are undefined
    or outside fam.inside_domain (for the sheared family {psi <= delta} has further components).

    Returns (field float64, lo, xs)."""
    lo, hi = fam.bbox()
    lo = lo - pad * h + np.asarray(phase) * h
    shape = tuple(int(math.ceil((hi[i] + pad * h - lo[i]) / h)) + 1 for i in range(3))
    xs = [lo[i] + h * np.arange(shape[i]) for i in range(3)]
    out = np.empty(shape)
    Y, Z = np.meshgrid(xs[1], xs[2], indexing="ij")
    for i, x in enumerate(xs[0]):
        with np.errstate(invalid="ignore", divide="ignore"):
            Pl = np.stack([np.full_like(Y, x), Y, Z], -1)
            v = fam.psi(Pl) - fam.delta
            v[~fam.inside_domain(Pl)] = 1.0
        v[~np.isfinite(v)] = 1.0
        out[i] = v
    return out, lo, xs


def lbm_plane_poiseuille(n_fluid, tau, g=1e-6, steps=200000, tol=1e-12):
    """Engine D3Q19 BGK + Guo step (lbm_domare_v1.lbm_step) between two solid planes, periodic in
    x and y, acceleration g e_x.  The exact profile is a parabola with curvature g / nu for ANY wall
    position, so fitting it returns the solver's effective wall position.  Returns a dict with the
    fitted walls (lattice units, fluid nodes at 1..n_fluid, solid nodes at 0 and n_fluid + 1) and the
    curvature ratio, which is exactly 1 for a correct viscosity and forcing."""
    import lbm_domare_v1 as LB
    nu = (tau - 0.5) / 3.0
    nz = n_fluid + 2
    solid = np.zeros((1, 1, nz), bool)
    solid[..., 0] = solid[..., -1] = True
    a = np.zeros((1, 1, nz, 3))
    a[..., 0] = g
    a[solid] = 0.0
    f = LB.feq3d(np.ones((1, 1, nz)), np.zeros((1, 1, nz, 3)))
    last = None
    for s in range(steps):
        f = LB.lbm_step(f, solid, tau, a, [])
        if s % 100 == 0:
            rho = f.sum(-1)
            u = (f @ LB.E + 0.5 * rho[..., None] * a) / rho[..., None]
            if last is not None and np.abs(u - last).max() < tol * np.abs(u).max():
                break
            last = u
    rho = f.sum(-1)
    ux = ((f @ LB.E + 0.5 * rho[..., None] * a) / rho[..., None])[0, 0, 1:-1, 0]
    z = np.arange(1, nz - 1, dtype=float)
    c = np.polyfit(z, ux, 2)
    lo_w, hi_w = np.sort(np.roots(c).real)
    return {"wall_low": float(lo_w), "wall_high": float(hi_w), "effective_width": float(hi_w - lo_w),
            "curvature_ratio": float(-2.0 * c[0] / (g / nu)), "steps": s}
