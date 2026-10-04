"""Exact axisymmetric contact of a rigid sphere with an elastic layer bonded to
a rigid base (frictionless top), solved as a 1-D Fredholm integral equation.

Galerkin (energy) method with a mixed basis that separates the two edge
behaviours of the axisymmetric contact problem:

    p(r) = c_s (1-(r/a)^2)^(-1/2) + sqrt(1-(r/a)^2) sum_{k>=0} d_k T_k(r/a)

The singular (flat-punch) mode must vanish for a smooth sphere; the contact
radius is therefore the root of c_s(a)=0.  This avoids the trivial root and
keeps the Galerkin matrix diagonally dominant (well conditioned).

    A_lk = 2 pi int_0^inf Kphys(xi) pbar_k(xi) pbar_l(xi) xi dxi
    b_l  = 2 pi int_0^a g(r) basis_l(r/a) r dr
"""
import numpy as np
from scipy.special import j0
from scipy.optimize import brentq
from numpy.polynomial.chebyshev import chebvander


def Psi(t, nu):
    t = np.asarray(t, float)
    q = np.exp(-2.0 * t)
    c = 3.0 - 4.0 * nu
    c0 = 16.0 * nu * nu - 24.0 * nu + 10.0
    # The two reflected boundary terms carry exp(-4*t) = q*q.
    # This follows directly from the four-mode boundary system in
    # FD_CART_HANKEL_EXACT_20260924/code/derive_kernel3.py.  The upstream
    # transcription used q in those two places, giving negative compliance
    # for nu>1/4 at long wavelengths.
    return (c * (1.0 - q * q) - 4.0 * t * q) / (c * (1.0 + q * q) + (4.0 * t * t + c0) * q)


class LayerContact:
    def __init__(self, R, t, E, nu, K=8, nxi=900, nu_quad=110, xmax=30.0):
        self.R, self.t, self.E, self.nu = R, t, E, nu
        self.K, self.nxi, self.nu_quad, self.xmax = K, nxi, nu_quad, xmax
        self.mu = E / (2.0 * (1.0 + nu))
        gx, gw = np.polynomial.legendre.leggauss(nu_quad)
        self.uq = 0.5 * (gx + 1.0)
        self.wq = 0.5 * gw
        sq = np.sqrt(np.maximum(1.0 - self.uq ** 2, 1e-300))
        Tc = chebvander(self.uq, K - 1)
        self.Basis = np.column_stack([1.0 / sq, sq[:, None] * Tc])  # nu_quad x (K+1)
        xg, xw = np.polynomial.legendre.leggauss(nxi)
        self.gxu = 0.5 * (xg + 1.0)
        self.gwu = 0.5 * xw

    def _xi(self, a):
        xmax_xi = self.xmax * max(1.0 / self.t, 1.0 / a)
        return self.gxu * xmax_xi, self.gwu * xmax_xi

    def _pbar(self, a, xi):
        B = j0(np.outer(xi * a, self.uq))
        return a * a * (B * (self.wq * self.uq)[None, :]) @ self.Basis

    def _gap(self, r):
        return self.delta - (self.R - np.sqrt(np.maximum(self.R ** 2 - r ** 2, 0.0)))

    def _solve_a(self, a, reg=0.0):
        xi, w = self._xi(a)
        PB = self._pbar(a, xi)
        # Kphys = (1-nu)/(mu*xi) Psi, and the measure is xi dxi, so
        # 2 pi int Kphys pbar_k pbar_l xi dxi = 2 pi int (1-nu)/mu Psi pbar_k pbar_l dxi
        fac = w * (1.0 - self.nu) / self.mu * Psi(xi * self.t, self.nu)
        A = 2.0 * np.pi * (PB * fac[:, None]).T @ PB
        gq = self._gap(a * self.uq)
        b = 2.0 * np.pi * a * a * (self.Basis.T @ (self.wq * gq * self.uq))
        n = A.shape[0]
        if reg > 0.0:
            A = A + reg * np.trace(A) / n * np.eye(n)
        c = np.linalg.solve(A, b)
        return c, PB, xi, w

    def solve(self, delta, a0=None, reg=1e-6):
        self.delta = delta
        R = self.R
        if a0 is None:
            a0 = np.sqrt(max(2.0 * R * delta - delta ** 2, 1e-12))

        def hc(a):
            try:
                c, _, _, _ = self._solve_a(a, reg)
            except np.linalg.LinAlgError:
                return np.nan
            return float(c[0])

        grid = a0 * np.linspace(0.25, 3.2, 44)
        vals = np.array([hc(a) for a in grid])
        cand = [k for k in range(len(grid) - 1)
                if np.isfinite(vals[k]) and np.isfinite(vals[k + 1])
                and vals[k] * vals[k + 1] <= 0]
        if cand:
            k = min(cand, key=lambda k: abs(0.5 * (grid[k] + grid[k + 1]) - a0))
            a = brentq(hc, grid[k], grid[k + 1], xtol=1e-11, rtol=1e-12)
        else:
            a = float(grid[np.nanargmin(np.abs(vals))])
        c, PB, xi, w = self._solve_a(a, reg)
        rr = np.linspace(0.0, a, 401)
        uu = rr / a
        sq = np.sqrt(np.maximum(1.0 - uu ** 2, 0.0))
        Tc = chebvander(uu, self.K - 1)
        p = c[0] / np.sqrt(np.maximum(1.0 - uu ** 2, 1e-12)) + sq * (Tc @ c[1:])
        p = np.maximum(p, 0.0)
        gq = self.Basis @ c
        force = float(2.0 * np.pi * a * a * np.sum(self.wq * gq * self.uq))
        return {"a": float(a), "r": rr, "p": p, "force": force,
                "pmax": float(p.max()), "c": c, "c_sing": float(c[0])}


# Public four-value API lives in contact_api.py; this module contains the
# corrected transfer kernel and the numerical solver only.
