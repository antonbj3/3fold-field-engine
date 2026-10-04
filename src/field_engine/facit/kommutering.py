"""Commutation factory: new exact test cases with controlled symmetry breaking.

Principle (Landreman 2026, eq. 2.9, read as a general rule).  If an operator N is invariant under a
map g, N(g.u) = g.N(u), and u is an exact solution, then g.u is an exact solution too.  When g is
not a symmetry of u (or of the grid the engine uses) the new case breaks a symmetry that the
original one had, at no cost and with the same exact answer.  Two instances matter here:

  HIDDEN STRETCH.  Steady Euler / MHD with (B.grad)B = -D x: B -> A B(A^-1 x) keeps
  div B = 0 and gives (B.grad)B = -A D A^-1 x, a gradient iff A^T A commutes with D.  For
  D = diag(1,1,4) that is A = R diag(M, c), M in GL(2): the axisymmetric Solov'ev field becomes a
  genuinely 3D equilibrium (`landreman.IntegerFamily`).
  GRID-BREAKING ROTATION.  The Laplacian and isotropic elasticity commute with every rotation,
  the engine's voxel lattice does not.  Rotating a body leaves every exact scalar output unchanged
  (volume, J, conductance, int u), so the spread of the engine's answer over rotations is a
  representation error measured WITHOUT a reference solution, and the rotation angle is a
  continuous dial for how much the body disagrees with the lattice.
  INTERTWINING.  A linear map A that is not a symmetry still maps one operator to another:
  v(x) = u(A^-1 x) solves div(A A^T grad v) = (lap u)(A^-1 x).  An isotropic exact solution becomes
  an exact solution of an anisotropic conduction problem on the stretched domain.
"""
from __future__ import annotations

import math

import numpy as np


def commutant_basis(D, tol=1e-12):
    """Orthonormal basis (list of n x n matrices) of {A : A D = D A}."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    I = np.eye(n)
    # vec(AD - DA) = (D^T kron I - I kron D) vec(A), column-major vec
    Mop = np.kron(D.T, I) - np.kron(I, D)
    _, s, vt = np.linalg.svd(Mop)
    null = vt[np.sum(s > tol * max(1.0, s.max())):]
    return [v.reshape(n, n, order="F") for v in null]


def stretch_admissible(A, D, tol=1e-12):
    """True if B -> A B(A^-1 x) maps solutions of (B.grad)B = -D x to equilibria: A^T A D = D A^T A."""
    A = np.asarray(A, dtype=np.float64)
    G = A.T @ A
    return bool(np.abs(G @ D - D @ G).max() <= tol * max(1.0, np.abs(G).max() * np.abs(D).max()))


def stretch_field(B0, A):
    """Return x -> A B0(A^-1 x) for a vector field B0 on (..., 3) arrays."""
    A = np.asarray(A, dtype=np.float64)
    Ai = np.linalg.inv(A)

    def B(x):
        return B0(np.asarray(x) @ Ai.T) @ A.T

    return B


def solovev_B0(x):
    """The axisymmetric Solov'ev field of Landreman eq. (2.6), Cartesian, (B0.grad)B0 = -diag(1,1,4) x."""
    x = np.asarray(x)
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    s = X * X + Y * Y
    F = np.sqrt(1.0 - (1.0 - s) ** 2 - 4.0 * Z * Z)
    return np.stack([(2 * Z * X - F * Y) / s, (2 * Z * Y + F * X) / s, 1.0 - s], axis=-1)


def rotation(axis, angle):
    """Rotation matrix about a unit axis by angle (radians)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def rotation2d(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def random_rotations(count, seed=0):
    """Deterministic uniformly distributed rotations (QR of Gaussian matrices, sign fixed)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        q, r = np.linalg.qr(rng.standard_normal((3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        out.append(q)
    return out


def rotate_scalar(u, R):
    """u_R(x) = u(R^T x): exact solution for the rotated body of any rotation-invariant scalar
    operator (Laplace, Poisson with a rotated source, heat)."""
    R = np.asarray(R, dtype=np.float64)
    return lambda x: u(np.asarray(x) @ R)


def rotate_vector(w, R):
    """w_R(x) = R w(R^T x): rotated displacement for isotropic elasticity (Navier operator)."""
    R = np.asarray(R, dtype=np.float64)
    return lambda x: w(np.asarray(x) @ R) @ R.T


def intertwine_scalar(u, A):
    """v(x) = u(A^-1 x), which solves div(K grad v) = (lap u)(A^-1 x) with K = A A^T."""
    Ai = np.linalg.inv(np.asarray(A, dtype=np.float64))
    return lambda x: u(np.asarray(x) @ Ai.T)


def orbit_spread(values, exact=None):
    """Summary of an engine output over a symmetry orbit: relative spread (max - min) / |mean| and,
    if the exact value is known, the worst relative error.  The spread is a lower bound on the
    representation error that needs no reference: every orbit member has the same exact answer."""
    v = np.asarray(values, dtype=np.float64)
    out = {"n": int(v.size), "mean": float(v.mean()), "spread_rel": float((v.max() - v.min()) / abs(v.mean()))}
    if exact is not None:
        out["worst_rel_err"] = float(np.max(np.abs(v / exact - 1.0)))
        out["mean_rel_err"] = float(v.mean() / exact - 1.0)
    return out
