"""Classical exact solutions against the engine's lattice Laplacian and hex8 elasticity.

Frozen from build/SOL_FALT_FACIT_20261001 (raw/e4_laplace.json, raw/e5_torsion.json).  The
point of each test is a SEPARATION: a case whose representation is exact on the lattice (aligned
block, aligned square) carries only discretisation error, and the same case rotated against the
lattice (commutation factory: the exact answer is unchanged) shows how much the representation
adds.
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(1, os.path.join(ROOT, "src", "field_engine"))

from field_engine.facit import klassiska as K            # noqa: E402
from field_engine.facit import kommutering as KO         # noqa: E402
pytest.importorskip("scipy")                              # the engine operators need it
from field_engine.facit import motorkoppling as MK       # noqa: E402


def _lap2(f, p, h=1e-3):
    return sum((f(p + h * e) + f(p - h * e) - 2 * f(p)) / h ** 2 for e in np.eye(2))


def test_prandtl_fields_solve_their_equation_and_vanish_on_the_boundary():
    p = np.array([[0.05, -0.03], [0.1, 0.02]])
    assert np.allclose(_lap2(lambda q: K.prandtl_phi_circle(q, 0.5), p), -2.0, atol=1e-6)
    assert np.allclose(_lap2(lambda q: K.prandtl_phi_ellipse(q, 0.5, 0.3), p), -2.0, atol=1e-6)
    assert np.allclose(_lap2(lambda q: K.prandtl_phi_triangle(q, 1.0), p), -2.0, atol=1e-6)
    V = K.equilateral_triangle(1.0)
    edge = 0.3 * V[0] + 0.7 * V[1]
    assert abs(K.prandtl_phi_triangle(edge[None], 1.0)[0]) < 1e-15
    # Saint-Venant series summed to 6000 odd terms in 30-digit arithmetic: 0.140577014955153716849...
    assert abs(K.torsion_constant("square", side=1.0) - 0.14057701495515372) < 1e-15


def test_torsion_constants_are_twice_the_integral_of_their_prandtl_fields():
    """J = 2 int phi, with phi the fields checked above.  phi is a polynomial of degree <= 3, so the
    Gauss rules below are exact: polar (circle, ellipse) and a collapsed square (triangle)."""
    g, w = np.polynomial.legendre.leggauss(8)
    r, wr = 0.5 * (g + 1), 0.5 * w
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for name, d, field, sx, sy in (
            ("circle", dict(R=0.7), lambda P: K.prandtl_phi_circle(P, 0.7), 0.7, 0.7),
            ("ellipse", dict(a=0.5, b=0.3), lambda P: K.prandtl_phi_ellipse(P, 0.5, 0.3), 0.5, 0.3)):
        RR, TT = np.meshgrid(r, t, indexing="ij")
        P = np.stack([sx * RR * np.cos(TT), sy * RR * np.sin(TT)], -1)
        J = 2 * np.sum(field(P) * sx * sy * RR * wr[:, None]) * (2 * np.pi / 16)
        assert abs(J / K.torsion_constant(name, **d) - 1) < 1e-13, name
    side = 1.3
    V = K.equilateral_triangle(side)
    xi, eta = np.meshgrid(r, r, indexing="ij")
    P = V[0] + xi[..., None] * (V[1] - V[0]) + (xi * eta)[..., None] * (V[2] - V[1])
    jac = 2 * K.polygon_area(V) * xi
    J = 2 * np.sum(K.prandtl_phi_triangle(P, side) * jac * np.outer(wr, wr))
    assert abs(J / K.torsion_constant("triangle", side=side) - 1) < 1e-13


def test_lame_solutions_meet_their_boundary_conditions():
    a, b, p, E, nu, h = 1.0, 2.0, 1.0, 1.0, 0.3, 1e-6
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    r = np.array([a, b])
    u = lambda s: K.lame_sphere_pressure(s, a, b, p, E, nu)
    sr = (lam + 2 * mu) * (u(r + h) - u(r - h)) / (2 * h) + 2 * lam * u(r) / r
    assert np.allclose(sr, [-p, 0.0], atol=1e-8)
    u = lambda s: K.lame_cylinder_pressure(s, a, b, p, E, nu)
    sr = (lam + 2 * mu) * (u(r + h) - u(r - h)) / (2 * h) + lam * u(r) / r
    assert np.allclose(sr, [-p, 0.0], atol=1e-8)
    u = lambda s: K.lame_cylinder_pressure(s, a, b, p, E, nu, plane="stress")
    sr = E / (1 - nu * nu) * ((u(r + h) - u(r - h)) / (2 * h) + nu * u(r) / r)
    assert np.allclose(sr, [-p, 0.0], atol=1e-8)
    # Navier radial equation u'' + k u'/r - k u/r^2 = 0 (k = 2 sphere, 1 cylinder) inside the wall
    s, d = np.array([1.2, 1.7]), 1e-4
    for k, f in ((2, lambda q: K.lame_sphere_pressure(q, a, b, p, E, nu)),
                 (1, lambda q: K.lame_cylinder_pressure(q, a, b, p, E, nu)),
                 (1, lambda q: K.lame_cylinder_pressure(q, a, b, p, E, nu, plane="stress"))):
        u2 = (f(s + d) - 2 * f(s) + f(s - d)) / d ** 2
        u1 = (f(s + d) - f(s - d)) / (2 * d)
        assert np.abs(u2 + k * u1 / s - k * f(s) / s ** 2).max() < 1e-6


def test_remaining_closed_forms():
    """Answers that no engine test reads yet, each checked by a route that does not use the formula."""
    g, w = np.polynomial.legendre.leggauss(40)
    R1, R2, H, th = 0.6, 1.4, 0.3, 0.7
    R = R1 + (R2 - R1) * 0.5 * (g + 1)
    G = H / th * np.sum(0.5 * (R2 - R1) * w / R)                  # (sigma / theta) int dA / R
    assert abs(G / K.conductance_annular_sector(R1, R2, H, th) - 1) < 1e-13
    L = np.array([1.0, 0.7, 0.4])
    u = lambda x: np.prod(np.sin(np.pi * x / L), axis=-1)
    x0, d = np.array([0.31, 0.22, 0.13]), 1e-3
    lap = sum(u(x0 + d * e) + u(x0 - d * e) - 2 * u(x0) for e in np.eye(3)) / d ** 2
    assert abs(-lap / u(x0) / K.cube_dirichlet_eigenvalue(L) - 1) < 1e-4      # O(d^2) difference
    assert K.axial_stiffness(2.0, 3.0, 4.0) == 1.5
    A = np.array([[1.3, 0.2, 0.0], [-0.1, 0.8, 0.3], [0.2, 0.0, 1.1]])
    u = lambda x: x[..., 0] ** 2 + 2 * x[..., 1] * x[..., 2] + x[..., 2] ** 3     # lap u = 2 + 6 z
    v = KO.intertwine_scalar(u, A)
    Kt, x0, d = A @ A.T, np.array([0.3, -0.2, 0.5]), 1e-3
    I3 = np.eye(3)
    hess = np.array([[(v(x0 + d * (I3[i] + I3[j])) - v(x0 + d * (I3[i] - I3[j]))
                       - v(x0 - d * (I3[i] - I3[j])) + v(x0 - d * (I3[i] + I3[j]))) / (4 * d * d)
                      for j in range(3)] for i in range(3)])
    assert abs(np.sum(Kt * hess) - (2 + 6 * (np.linalg.solve(A, x0))[2])) < 1e-6
    for q in KO.random_rotations(4, seed=5):
        assert np.allclose(q @ q.T, I3, atol=1e-14) and abs(np.linalg.det(q) - 1) < 1e-14
    sq = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    P = np.array([[0.0, 0.0], [0.1, 0.3], [1.0, 0.2], [1.0, 1.0], [-0.5, 0.1]])
    exact = np.array([-0.5, -0.2, 0.5, math.sqrt(0.5), 0.0])
    assert np.abs(K.polygon_sdf(P, sq) - exact).max() < 1e-15


def test_engine_laplacian_block_conductance_is_exact():
    """Aligned block: no representation error and a linear potential, so the lattice is exact."""
    for (nx, ny, nz) in ((10, 4, 3), (40, 13, 7)):
        free = np.ones((nx, ny, nz), bool)
        lo = np.zeros_like(free); lo[0] = True
        hi = np.zeros_like(free); hi[-1] = True
        G = MK.lattice_conductance(free, lo, hi)
        assert abs(G / K.conductance_box(nx, ny * nz) - 1) < 1e-12


def test_engine_laplacian_quarter_torus_staircase_error():
    """Curved Neumann walls on the lattice: -6.2 % at h = r/4, -3.5 % at h = r/8 (first order)."""
    R0, r = 1.0, 0.4
    ex = K.conductance_torus_sector(R0, r, math.pi / 2)
    errs = []
    for h in (0.1, 0.05):
        n = int(math.ceil((R0 + r) / h)) + 1
        nz = int(math.ceil(2 * r / h)) + 2
        xc = (np.arange(n) + 0.5) * h
        zc = (np.arange(nz) - nz / 2 + 0.5) * h
        X, Y, Z = np.meshgrid(xc, xc, zc, indexing="ij")
        free = (np.sqrt(X * X + Y * Y) - R0) ** 2 + Z * Z < r * r
        lo = np.zeros_like(free); lo[:, 0, :] = True
        hi = np.zeros_like(free); hi[0, :, :] = True
        errs.append(MK.lattice_conductance(free, lo, hi) * h / ex - 1)
    assert -0.07 < errs[0] < -0.055 and -0.04 < errs[1] < -0.03


def test_prandtl_square_rotation_separates_representation_from_discretisation():
    """Same square, same lattice, same operator: rotating it (exact J unchanged) multiplies the
    staircase error; the cut closure (boundary position from the exact SDF) removes it."""
    Jex = K.torsion_constant("square", side=1.0)
    out = {}
    for deg in (0.0, 22.5):
        V = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) @ KO.rotation2d(math.radians(deg)).T
        sdf = lambda P, V=V: K.polygon_sdf(P, V)
        out[deg] = {c: MK.prandtl_torsion_lattice(sdf, Jex, 32, closure=c)[1] for c in ("stair", "cut")}
    assert abs(out[0.0]["cut"]) < 5e-3 and abs(out[22.5]["cut"]) < 5e-3
    assert abs(out[22.5]["stair"]) > 5 * abs(out[22.5]["cut"])


def test_engine_hex8_torsion_aligned_square_is_discretisation_only():
    """Aligned voxel square (exact representation): Q1 over-stiff by +1.35 % at d = 8, and a
    translated lattice gives the same number (translation is in the commutation group)."""
    Jex = K.torsion_constant("square", side=8)
    V = np.array([[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]])
    inside = lambda P: K.polygon_sdf(P, V) < 0
    a = MK.hex8_torsion(inside, Jex, 8, phase=(0.5, 0.5))
    b = MK.hex8_torsion(inside, Jex, 8, phase=(0.37, 0.61))
    assert 0.010 < a["J_eff_rel"] < 0.017
    assert abs(a["J_eff_rel"] - b["J_eff_rel"]) < 1e-9
    Vr = V @ KO.rotation2d(math.radians(22.5)).T
    c = MK.hex8_torsion(lambda P: K.polygon_sdf(P, Vr) < 0, Jex, 8)
    assert abs(c["J_eff_rel"]) > 3 * abs(a["J_eff_rel"])


def test_ellipsoid_poisson_integral_and_rotation_invariance():
    semi = (0.5, 0.35, 0.25)
    R = KO.random_rotations(1, seed=3)[0]
    u = KO.rotate_scalar(lambda x: K.ellipsoid_poisson_u(x, semi), R)
    x0, h = np.array([[0.1, 0.05, -0.02]]), 1e-3
    lap = sum((u(x0 + h * e) + u(x0 - h * e) - 2 * u(x0)) / h ** 2 for e in np.eye(3))
    assert np.allclose(-lap, 1.0, atol=1e-6)
    assert abs(K.ellipsoid_poisson_integral((1, 1, 1)) - 4 * math.pi / 45) < 1e-15


def test_engine_lbm_plane_poiseuille_curvature_exact_and_wall_position_depends_on_tau():
    """Exact parabola between two planes.  The curvature is g/nu to 1e-9; the fitted wall of the
    default solid-node reversal is NOT on the solid node but tau-dependent: width n + 0.14 (tau 0.6),
    n + 0.55 (0.8), n + 1.38 (1.2), independent of n (raw/e6_lbm.json)."""
    widths = {}
    for tau in (0.6, 0.8, 1.2):
        r = MK.lbm_plane_poiseuille(8, tau)
        assert abs(r["curvature_ratio"] - 1.0) < 1e-9
        widths[tau] = r["effective_width"] - 8
    assert 0.10 < widths[0.6] < 0.18 and 0.50 < widths[0.8] < 0.60 and 1.33 < widths[1.2] < 1.43
