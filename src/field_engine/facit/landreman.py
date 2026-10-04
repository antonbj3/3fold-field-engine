"""Exact 3D toroidal equilibria with nested invariant surfaces (Landreman, arXiv:2609.26742v2).

Two families, both with explicit Cartesian formulas in elementary functions, used here as a CLOSED
ANSWER KEY for the field engine: a smooth, genuinely three-dimensional torus whose volume, second
moments, surface points, magnetic axis and field-line topology are known exactly, with a continuous
dial eps that moves the body away from axisymmetry.

The equations solved are (curl B) x B = grad p, div B = 0, equivalently steady incompressible Euler
flow u = B with hydrodynamic pressure Q and Bernoulli function H = Q + |B|^2/2.

INTEGER-IOTA FAMILY (paper section 2).  B0 is the Solov'ev field with (B0.grad)B0 = -D x,
D = diag(1, 1, 4).  Any constant A with A^T A commuting with D gives a new solution
B(x) = A B0(A^-1 x) with (B.grad)B = -A D A^-1 x (paper eq. 2.9 for A = diag(a, b, 1)).  Here A is
diag(a, b, c) optionally followed by a rotation R, so the paper's family is c = 1, a = sqrt(1+eps),
b = sqrt(1-eps), R = I.  c != 1 (vertical elongation) and R != I (a torus tilted against the voxel
lattice) follow from the same one-line argument; they are not in the paper and are labelled as
factory output.  Field lines are r = A r0(u, v, zeta) with r0 the paper's map (2.11) at a = b = 1,
so every surface point, the axis, the constant Jacobian -abc and V(psi) = 2 pi^2 abc psi are exact.

SHEARED-IOTA FAMILY (paper section 3).  Parameters eps, S, lam > 0; formulas (3.1)-(3.3), surface
map (3.9)-(3.13), axis (3.20), rotational transform (3.24), volume (3.34).

Units: the formulas are dimensionless.  `scale` (length L) maps them to a part in mm:
x_mm = L * x.  Volumes scale L^3, second moments L^5.

Every function takes arrays of shape (..., 3) and is written with holomorphic numpy operations only
(no abs, conj, real or imag inside the field formulas), so a complex-step derivative
Im f(x + i h e_j) / h gives the Jacobian to machine precision.  `residuals` uses that.
"""
from __future__ import annotations

import math

import numpy as np

D_INT = np.diag([1.0, 1.0, 4.0])


# ------------------------------------------------------------------------------------------------
# integer-iota family
# ------------------------------------------------------------------------------------------------
class IntegerFamily:
    """B(x) = A B0(A^-1 x), A = R diag(a, b, c).

    eps sets a = sqrt(1+eps), b = sqrt(1-eps) (paper); a, b may be given directly instead.
    R is a 3x3 rotation (default identity).  delta is the outermost flux surface psi = delta.
    """

    def __init__(self, eps=0.5, delta=1.0 / 64.0, c=1.0, a=None, b=None, R=None, scale=1.0):
        if a is None or b is None:
            if not (0.0 <= eps < 1.0):
                raise ValueError("eps must lie in [0, 1)")
            a, b = math.sqrt(1.0 + eps), math.sqrt(1.0 - eps)
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.eps = (self.a ** 2 - self.b ** 2) / 2.0
        self.m = (self.a ** 2 + self.b ** 2) / 2.0
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
        if not np.allclose(self.R @ self.R.T, np.eye(3), atol=1e-12):
            raise ValueError("R must be orthogonal")
        self.scale = float(scale)
        self.S = np.diag([self.a, self.b, self.c])
        self.A = self.R @ self.S
        # label-plane centre of the psi circles and the admissible delta (paper 2.17 generalised)
        self.u_axis = -self.eps / (2.0 * self.c ** 2)
        if not abs(self.u_axis) < 0.5:
            # the axis label lies outside the label disk q < 1/2 (paper U_eps): no flux surface exists
            raise ValueError(f"no admissible delta: |u_axis| = eps / (2 c^2) = {abs(self.u_axis):.6g} "
                             "must be < 1/2")
        self.delta_max = (0.5 - abs(self.u_axis)) ** 2
        self.delta = float(delta)
        if not (0.0 < self.delta < self.delta_max):
            raise ValueError(f"delta must lie in (0, {self.delta_max:.6g})")

    # --- coordinates -----------------------------------------------------------------------------
    def _to_base(self, x):
        """Base (axisymmetric) coordinates X = S^-1 R^T x / scale."""
        x = np.asarray(x) / self.scale
        xr = x @ self.R  # R^T x for row vectors
        return xr[..., 0] / self.a, xr[..., 1] / self.b, xr[..., 2] / self.c

    def B(self, x):
        """Field at points x (..., 3).  Dimensionless amplitude (same for any scale)."""
        X, Y, Z = self._to_base(x)
        s = X * X + Y * Y
        F = np.sqrt(1.0 - (1.0 - s) ** 2 - 4.0 * Z * Z)
        b0 = np.stack([(2.0 * Z * X - F * Y) / s, (2.0 * Z * Y + F * X) / s, 1.0 - s], axis=-1)
        return (b0 * np.array([self.a, self.b, self.c])) @ self.R.T

    def Q(self, x):
        """Hydrodynamic pressure: (B.grad)B = -grad Q, Q = x.(A D A^-1)x / 2 (base units)."""
        x = np.asarray(x) / self.scale
        M = self.A @ D_INT @ np.linalg.inv(self.A)
        M = 0.5 * (M + M.T)
        return 0.5 * np.einsum("...i,ij,...j->...", x, M, x)

    def H(self, x):
        Bv = self.B(x)
        return self.Q(x) + 0.5 * np.sum(Bv * Bv, axis=-1)

    def psi(self, x):
        """Flux label, zero on the axis: psi = (u - u_axis)^2 + v^2 in field-line labels.

        For c = 1, R = I, a^2 + b^2 = 2 this is the paper's (2.4)."""
        H = self.H(x)
        return (H - self.m + self.eps ** 2 / (2.0 * self.c ** 2)) / (2.0 * self.c ** 2)

    def p(self, x, p_a=None):
        """Scalar pressure, (curl B) x B = grad p; p_a defaults to 2 c^2 delta (p = 0 on the edge)."""
        if p_a is None:
            p_a = 2.0 * self.c ** 2 * self.delta
        return p_a - 2.0 * self.c ** 2 * self.psi(x)

    def sdf_proxy(self, x):
        """psi - delta: negative inside the solid torus Omega_delta, zero exactly on its surface."""
        return self.psi(x) - self.delta

    def inside_domain(self, x):
        """True where the formulas are smooth (paper U_eps: radicand > 0)."""
        X, Y, Z = self._to_base(x)
        s = X * X + Y * Y
        return (1.0 - (1.0 - s) ** 2 - 4.0 * Z * Z) > 0.0

    # --- field-line map --------------------------------------------------------------------------
    @staticmethod
    def base_map(u, v, zeta):
        """Paper (2.11) with a = b = 1: axisymmetric field lines, labels (u, v), q = |(u,v)| < 1/2."""
        q2 = u * u + v * v
        L = np.sqrt((1.0 + np.sqrt(1.0 - 4.0 * q2)) / 2.0)
        cz, sz = np.cos(zeta), np.sin(zeta)
        x = L * cz + (u * cz + v * sz) / L
        y = L * sz + (v * cz - u * sz) / L
        z = v * np.cos(2.0 * zeta) - u * np.sin(2.0 * zeta)
        return np.stack([x, y, z], axis=-1)

    def field_line(self, u, v, zeta):
        """Exact position on the field line with labels (u, v) at parameter zeta (d r/d zeta = B)."""
        r0 = self.base_map(np.asarray(u), np.asarray(v), np.asarray(zeta))
        return self.scale * (r0 @ self.A.T)

    def surface_points(self, psi, alpha, zeta):
        """Exact points on the surface psi (paper 2.16): labels on a circle of radius sqrt(psi)."""
        rho = math.sqrt(psi)
        return self.field_line(self.u_axis + rho * np.cos(alpha), rho * np.sin(alpha), zeta)

    def axis(self, zeta):
        return self.field_line(self.u_axis + 0.0 * np.asarray(zeta), 0.0 * np.asarray(zeta), zeta)

    def surface_mesh(self, n_alpha, n_zeta, psi=None):
        """Closed triangle mesh with every vertex exactly on the surface psi (default delta).

        Outward-oriented for det(A) > 0.  Returns (V, F)."""
        psi = self.delta if psi is None else psi
        al = np.linspace(0.0, 2.0 * np.pi, n_alpha, endpoint=False)
        ze = np.linspace(0.0, 2.0 * np.pi, n_zeta, endpoint=False)
        AL, ZE = np.meshgrid(al, ze, indexing="ij")
        V = self.surface_points(psi, AL, ZE).reshape(-1, 3)
        i, j = np.meshgrid(np.arange(n_alpha), np.arange(n_zeta), indexing="ij")
        i1, j1 = (i + 1) % n_alpha, (j + 1) % n_zeta
        v00, v10 = (i * n_zeta + j).ravel(), (i1 * n_zeta + j).ravel()
        v01, v11 = (i * n_zeta + j1).ravel(), (i1 * n_zeta + j1).ravel()
        F = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
        # orientation: the map has det -abc, so (alpha, zeta) ordering gives inward normals for
        # det(A) > 0; flip to outward
        if np.linalg.det(self.A) > 0:
            F = F[:, ::-1]
        return V, F.astype(np.int64)

    # --- exact integrals -------------------------------------------------------------------------
    def jacobian_det(self):
        """det d(x,y,z)/d(u,v,zeta) = -abc * det(R) * scale^3 (paper 2.13 at c = 1)."""
        return -self.a * self.b * self.c * np.linalg.det(self.R) * self.scale ** 3

    def volume(self, psi=None):
        """V(psi) = 2 pi^2 abc psi (paper 2.20 at c = 1)."""
        psi = self.delta if psi is None else psi
        return 2.0 * math.pi ** 2 * self.a * self.b * self.c * psi * self.scale ** 3

    def toroidal_flux(self, psi=None):
        """Phi_t = pi abc psi (paper 2.21 at c = 1), in base units times scale^2."""
        psi = self.delta if psi is None else psi
        return math.pi * self.a * self.b * self.c * psi * self.scale ** 2

    def centroid(self):
        return np.zeros(3)

    def second_moment(self, psi=None):
        """Exact C_ij = integral over Omega_psi of x_i x_j dV (density 1).

        Derivation: C = |det| A (int r0 r0^T du dv dzeta) A^T.  Over one zeta period
        int x0^2 = pi (1 + 2u), int y0^2 = pi (1 - 2u), int z0^2 = pi (u^2 + v^2), cross terms
        pi*2v, 0, 0 (use L^2 + q^2/L^2 = 1, paper 2.12).  Over the label disk of radius sqrt(psi)
        centred at (u_axis, 0): int u = pi psi u_axis, int v = 0, int (u^2+v^2) = pi psi
        (u_axis^2 + psi/2).  Hence the formula below, a polynomial in (a, b, c, psi)."""
        psi = self.delta if psi is None else psi
        u0 = self.u_axis
        C0 = np.diag([math.pi ** 2 * psi * (1.0 + 2.0 * u0),
                      math.pi ** 2 * psi * (1.0 - 2.0 * u0),
                      math.pi ** 2 * psi * (u0 * u0 + psi / 2.0)])
        det = self.a * self.b * self.c
        return det * (self.A @ C0 @ self.A.T) * self.scale ** 5

    def inertia(self, psi=None):
        """Physical inertia tensor about the centroid (density 1): tr(C) I - C."""
        C = self.second_moment(psi)
        return np.trace(C) * np.eye(3) - C

    def bbox(self, psi=None, n=256):
        """A bounding box of Omega_psi from dense exact surface points, padded by 2 %."""
        psi = self.delta if psi is None else psi
        al, ze = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, 2 * np.pi, n))
        P = self.surface_points(psi, al, ze).reshape(-1, 3)
        lo, hi = P.min(0), P.max(0)
        pad = 0.02 * (hi - lo).max()
        return lo - pad, hi + pad

    def volume_avg_p_B2(self):
        """<p>_V and <|B|^2>_V for p vanishing at the edge (paper 2.27, 2.29), c = 1, R = I only."""
        if self.c != 1.0 or abs(self.m - 1.0) > 1e-14:
            raise NotImplementedError("paper normalisation only")
        return self.delta, 1.0 - self.eps ** 2 / 2.0 + self.delta


# ------------------------------------------------------------------------------------------------
# sheared-iota family, written with real pairs so that complex-step differentiation works
# ------------------------------------------------------------------------------------------------
def _cmul(p, q, r, s):
    return p * r - q * s, p * s + q * r


def _cdiv(p, q, r, s):
    d = r * r + s * s
    return (p * r + q * s) / d, (q * r - p * s) / d


def _csqrt(p, q):
    """Principal square root of p + i q, valid off the non-positive real axis.

    Two algebraically equal forms are selected by the sign of p so that neither r + p nor r - p
    cancels catastrophically.  Inside Omega_delta the argument of 1 + eps / conj(w)^2 stays below
    84 degrees for the panels of paper fig. 2 (below 96 for eq. 3.27), and the form
    sqrt((r + p) / 2) alone gives roundoff residuals there; the second form matters on the rest of the smooth domain, e.g. the second component of
    {psi <= delta} beside the excluded segment, where the argument reaches 180 degrees and the
    single form lost up to 3e-3 in the complex-step residuals.  The selection is by np.where on the
    real part, so each branch stays holomorphic and complex-step differentiation remains exact."""
    r = np.sqrt(p * p + q * q)
    with np.errstate(invalid="ignore", divide="ignore"):
        re1 = np.sqrt((r + p) / 2.0)
        im1 = q / (2.0 * re1)
        t = np.sqrt((r - p) / 2.0)
        im2 = np.where(np.real(q) >= 0, t, -t)
        re2 = q / (2.0 * im2)
    pos = np.real(p) >= 0
    return np.where(pos, re1, re2), np.where(pos, im1, im2)


def _csin(p, q):
    return np.sin(p) * np.cosh(q), np.cos(p) * np.sinh(q)


def _ccos(p, q):
    return np.cos(p) * np.cosh(q), -np.sin(p) * np.sinh(q)


class ShearedFamily:
    """Paper section 3: eps, S, lam > 0, outermost surface psi = delta = kb^2 / 2."""

    def __init__(self, eps=2.0, S=1.0, lam=3.5, kb=0.1, scale=1.0):
        self.eps, self.S, self.lam, self.kb = float(eps), float(S), float(lam), float(kb)
        if not (0 < kb < 1 and math.asin(kb) < S):
            raise ValueError("need 0 < kb < 1 and arcsin(kb) < S (paper 3.15)")
        self.delta = kb * kb / 2.0
        self.scale = float(scale)

    def _parts(self, x):
        x = np.asarray(x) / self.scale
        X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
        wb_r, wb_i = X, -Y                                    # omega-bar
        w2_r, w2_i = _cmul(wb_r, wb_i, wb_r, wb_i)             # omega-bar^2
        t_r, t_i = _cdiv(self.eps + 0.0 * X, 0.0 * X, w2_r, w2_i)
        sq_r, sq_i = _csqrt(1.0 + t_r, t_i)
        K_r, K_i = _cmul(wb_r, wb_i, sq_r, sq_i)
        wK_r, wK_i = _cmul(X, Y, K_r, K_i)
        Xi_r, Xi_i = wK_r + math.pi / 2.0 - self.S, wK_i
        e_r, e_i = np.cos(self.lam * Z), -np.sin(self.lam * Z)  # exp(-i lam z)
        return X, Y, Z, K_r, K_i, Xi_r, Xi_i, e_r, e_i

    def B(self, x):
        X, Y, Z, K_r, K_i, Xi_r, Xi_i, e_r, e_i = self._parts(x)
        s_r, s_i = _csin(Xi_r, Xi_i)
        W_r, W_i = _cdiv(-s_i, s_r, 2.0 * K_r, 2.0 * K_i)      # i sin(Xi) / (2K)
        h_r, h_i = _cmul(e_r, e_i, W_r, W_i)
        c_r, c_i = _ccos(Xi_r, Xi_i)
        ec_r, _ = _cmul(e_r, e_i, c_r, c_i)
        return np.stack([h_r, h_i, ec_r / self.lam], axis=-1)

    def psi(self, x):
        X, Y, Z, K_r, K_i, Xi_r, Xi_i, e_r, e_i = self._parts(x)
        c_r, c_i = _ccos(Xi_r, Xi_i)
        ec_r, _ = _cmul(e_r, e_i, c_r, c_i)
        return 0.5 * (np.sin(self.lam * Z) ** 2 + ec_r ** 2)

    def Q(self, x):
        """(B.grad)B = -grad Q with Q = -|W|^2/2 + sin^2(lam z)/(2 lam^2) (paper 3.7)."""
        Bv = self.B(x)
        Z = np.asarray(x)[..., 2] / self.scale
        return -0.5 * (Bv[..., 0] ** 2 + Bv[..., 1] ** 2) + np.sin(self.lam * Z) ** 2 / (2 * self.lam ** 2)

    def H(self, x):
        Bv = self.B(x)
        return self.Q(x) + 0.5 * np.sum(Bv * Bv, axis=-1)

    def p(self, x, p_a=None):
        """p = p_a - psi / lam^2 (paper 3.3, H = psi / lam^2 by 3.7); p_a defaults to delta/lam^2."""
        if p_a is None:
            p_a = self.delta / self.lam ** 2
        return p_a - self.psi(x) / self.lam ** 2

    def sdf_proxy(self, x):
        """psi - delta.  Only inside_domain(x) is Omega_delta: {psi <= delta} has further components
        (sigma - S near +-pi, other periods of lam z) that are not the torus."""
        return self.psi(x) - self.delta

    def inside_domain(self, x):
        """True on the chart of the surface map (paper 3.11-3.13): |sigma - S| < pi/2 with
        sigma = Re(omega K) (3.10), and |lam z| < pi/2.  On this chart psi <= delta is exactly
        Omega_delta, since 2 psi >= sin^2(sigma - S) and 2 psi >= sin^2(lam z)."""
        X, Y, Z, K_r, K_i, Xi_r, Xi_i, e_r, e_i = self._parts(x)
        sigma = np.real(Xi_r) - math.pi / 2.0 + self.S
        return (np.abs(sigma - self.S) < 0.5 * math.pi) & (np.abs(self.lam * np.real(Z)) < 0.5 * math.pi)

    # --- surface map (paper 3.9-3.13) ------------------------------------------------------------
    def _acbc(self, sigma):
        h = np.sqrt(4.0 * sigma * sigma + self.eps ** 2)
        return np.sqrt((h - self.eps) / 2.0), np.sqrt((h + self.eps) / 2.0), h

    def map_XYzeta(self, Xc, Yc, zeta):
        nu = 0.5 * self.eps * np.sin(2.0 * zeta)
        sigma = (self.S + np.arctan(np.tanh(nu) * Yc / np.sqrt(1.0 - Yc * Yc))
                 - np.arcsin(Xc / np.sqrt(np.cosh(nu) ** 2 - Yc * Yc)))
        ac, bc, _ = self._acbc(sigma)
        P = np.stack([ac * np.cos(zeta), bc * np.sin(zeta), -np.arcsin(Yc) / self.lam], axis=-1)
        return self.scale * P

    def surface_points(self, psi, chi, zeta):
        k = math.sqrt(2.0 * psi)
        return self.map_XYzeta(-k * np.cos(chi), k * np.sin(chi), zeta)

    def axis(self, zeta):
        ac, bc, _ = self._acbc(np.asarray(self.S))
        return self.scale * np.stack([ac * np.cos(zeta), bc * np.sin(zeta), 0.0 * zeta], axis=-1)

    def bbox(self, psi=None, n=256):
        psi = self.delta if psi is None else psi
        ch, ze = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, 2 * np.pi, n))
        P = self.surface_points(psi, ch, ze).reshape(-1, 3)
        lo, hi = P.min(0), P.max(0)
        pad = 0.02 * (hi - lo).max()
        return lo - pad, hi + pad

    def surface_mesh(self, n_chi, n_zeta, psi=None):
        psi = self.delta if psi is None else psi
        ch = np.linspace(0.0, 2.0 * np.pi, n_chi, endpoint=False)
        ze = np.linspace(0.0, 2.0 * np.pi, n_zeta, endpoint=False)
        CH, ZE = np.meshgrid(ch, ze, indexing="ij")
        V = self.surface_points(psi, CH, ZE).reshape(-1, 3)
        i, j = np.meshgrid(np.arange(n_chi), np.arange(n_zeta), indexing="ij")
        i1, j1 = (i + 1) % n_chi, (j + 1) % n_zeta
        v00, v10 = (i * n_zeta + j).ravel(), (i1 * n_zeta + j).ravel()
        v01, v11 = (i * n_zeta + j1).ravel(), (i1 * n_zeta + j1).ravel()
        F = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
        return V, F.astype(np.int64)

    # --- rotational transform (paper 3.24) and volume (paper 3.34) -------------------------------
    def iota(self, psi, n_transits=40, steps_per_transit=2000):
        """Field-line average of d chi / d zeta over n_transits, RK4 in zeta."""
        k = math.sqrt(2.0 * psi)

        def rhs(zeta, chi):
            if k == 0.0:
                Xc = Yc = 0.0
            else:
                Xc, Yc = -k * math.cos(chi), k * math.sin(chi)
            nu = 0.5 * self.eps * math.sin(2.0 * zeta)
            sigma = (self.S + math.atan(math.tanh(nu) * Yc / math.sqrt(1.0 - Yc * Yc))
                     - math.asin(Xc / math.sqrt(math.cosh(nu) ** 2 - Yc * Yc)))
            h = math.sqrt(4.0 * sigma * sigma + self.eps ** 2)
            G = (h + self.eps * math.cos(2.0 * zeta)) / 2.0
            return 2.0 * G * math.sqrt(1.0 - k * k * math.sin(chi) ** 2) / math.sqrt(math.cosh(nu) ** 2 - k * k)

        dz = 2.0 * math.pi / steps_per_transit
        chi, zeta = 0.0, 0.0
        for _ in range(n_transits * steps_per_transit):
            k1 = rhs(zeta, chi)
            k2 = rhs(zeta + dz / 2, chi + dz * k1 / 2)
            k3 = rhs(zeta + dz / 2, chi + dz * k2 / 2)
            k4 = rhs(zeta + dz, chi + dz * k3)
            chi += dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            zeta += dz
        return chi / (2.0 * math.pi * n_transits)

    def iota_axis(self, n=4096):
        """iota(0) = h(S) * mean(sech nu) (paper 3.25-3.26); trapezoid is spectral for periodic."""
        z = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return math.sqrt(4 * self.S ** 2 + self.eps ** 2) * float(np.mean(1.0 / np.cosh(0.5 * self.eps * np.sin(2 * z))))

    def volume(self, n_zeta=256, n_xi=256):
        """V(delta) = I[U] / lam (paper 3.34), Gauss-Legendre in xi, trapezoid in zeta."""
        d = math.asin(self.kb)
        # xi = d sin(t) removes the square-root endpoint singularity of U at xi = +-d, so the
        # Gauss rule converges spectrally instead of algebraically
        g, w = np.polynomial.legendre.leggauss(n_xi)
        t = 0.5 * math.pi * g
        xi = d * np.sin(t)
        wx = d * np.cos(t) * 0.5 * math.pi * w
        ze = np.linspace(0.0, 2 * np.pi, n_zeta, endpoint=False)
        XI, ZE = np.meshgrid(xi, ze, indexing="ij")
        nu = 0.5 * self.eps * np.sin(2 * ZE)
        F = np.cosh(nu) ** 2 - np.sin(XI) ** 2
        U = np.arcsin(np.sqrt(np.clip((self.kb ** 2 - np.sin(XI) ** 2) / F, 0.0, 1.0)))
        I = float(np.sum(U * wx[:, None]) * (2 * np.pi / n_zeta))
        return I / self.lam * self.scale ** 3


# ------------------------------------------------------------------------------------------------
# residuals by complex step (machine-precision derivatives, no subtractive cancellation)
# ------------------------------------------------------------------------------------------------
def complex_step_jacobian(f, x, h=1e-30):
    """J[..., i, j] = d f_i / d x_j for vector f, or gradient [..., j] for scalar f."""
    x = np.asarray(x, dtype=np.float64)
    cols = []
    for j in range(3):
        xc = x.astype(np.complex128)
        xc[..., j] += 1j * h
        cols.append(np.imag(f(xc)) / h)
    return np.stack(cols, axis=-1)


def residuals(fam, x):
    """Max-abs residuals over points x of the exact equations, each scaled by its natural size.

    Returns dict: div (div B), euler ((B.grad)B + grad Q), mhd ((curl B) x B - grad p),
    invariance (B . grad psi), and the scales |B|^2 / L used for normalisation."""
    Bv = fam.B(x)
    J = complex_step_jacobian(fam.B, x)                       # J[..., i, j] = dB_i/dx_j
    gQ = complex_step_jacobian(fam.Q, x)
    gpsi = complex_step_jacobian(fam.psi, x)
    gp = complex_step_jacobian(fam.p, x)
    div = np.trace(J, axis1=-2, axis2=-1)
    tension = np.einsum("...ij,...j->...i", J, Bv)
    curl = np.stack([J[..., 2, 1] - J[..., 1, 2], J[..., 0, 2] - J[..., 2, 0],
                     J[..., 1, 0] - J[..., 0, 1]], axis=-1)
    lorentz = np.cross(curl, Bv)
    scaleB = float(np.max(np.linalg.norm(Bv, axis=-1)))
    return {
        "n": int(np.asarray(x).reshape(-1, 3).shape[0]),
        "div": float(np.max(np.abs(div))),
        "euler": float(np.max(np.linalg.norm(tension + gQ, axis=-1))),
        "mhd": float(np.max(np.linalg.norm(lorentz - gp, axis=-1))),
        "invariance": float(np.max(np.abs(np.einsum("...i,...i->...", Bv, gpsi)))),
        "max_B": scaleB,
        "max_grad_p": float(np.max(np.linalg.norm(gp, axis=-1))),
    }


def interior_points(fam, n, seed=0, psi_max=None, max_rounds=200):
    """n uniformly random points of Omega_psi_max (default delta), by rejection from the bbox.

    Both families are restricted to fam.inside_domain: for the sheared family {psi <= delta} in the
    bbox also holds a second component (up to 1.1 x V for the panels of paper fig. 2).

    Raises RuntimeError after max_rounds rejection rounds, so an empty or near-empty domain fails
    instead of looping forever."""
    rng = np.random.default_rng(seed)
    lo, hi = fam.bbox()
    psi_max = fam.delta if psi_max is None else psi_max
    out = []
    got = 0
    rounds = 0
    while got < n:
        if rounds >= max_rounds:
            raise RuntimeError(
                f"interior_points: only {got} of {n} points after {max_rounds} rejection rounds; "
                "the domain is empty or far smaller than its bounding box")
        rounds += 1
        P = lo + (hi - lo) * rng.random((4 * n, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = np.isfinite(fam.psi(P)) & (fam.psi(P) <= psi_max) & fam.inside_domain(P)
        out.append(P[ok])
        got += int(ok.sum())
    return np.concatenate(out)[:n]
