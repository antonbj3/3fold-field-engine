"""Classical exact solutions for the equations the field engine itself solves.

The engine has a graph Laplacian on occupancy (opt/laplaceflode_v1.py), a Q1 hex8 linear-elastic
core (lastfalt_v1_fem.py), volume meters (faltvolym_v1.py) and mesh mass properties
(experimental/shape_massprop/femur_massprops.py).  Each function here returns a number or a field
that one of those can be compared with, together with the units.  Lengths are in any consistent unit
(mm in the engine); E, G in force/length^2; conductivity sigma in 1/(ohm length).

Contents
  Poisson, Dirichlet, -lap u = f:
    ellipsoid_poisson_u / _integral      -lap u = 1 in an ellipsoid, u = 0 on the boundary
    cube_dirichlet_eigenvalue            first eigenvalue of -lap on a box
  Prandtl torsion, -lap phi = 2 in a cross-section, phi = 0 on its boundary, J = 2 int phi:
    prandtl_phi_circle / _ellipse / _triangle, torsion_constant(shape, ...)
  Conduction, Neumann walls, Dirichlet electrodes:
    conductance_box, conductance_torus_sector, conductance_annular_sector
  Linear elasticity (isotropic):
    torsion_twist_rate, axial_stiffness, lame_sphere_pressure, lame_cylinder_pressure
  Polygon geometry (exact for straight edges):
    regular_polygon, polygon_sdf, polygon_area
"""
from __future__ import annotations

import math

import numpy as np


# ------------------------------------------------------------------------------------------------
# Poisson
# ------------------------------------------------------------------------------------------------
def ellipsoid_poisson_u(x, semi_axes):
    """u = (1 - sum x_i^2/a_i^2) / (2 sum 1/a_i^2) solves -lap u = 1, u = 0 on the ellipsoid.

    x (..., 3) in the ellipsoid's own frame (rotate first for a rotated body)."""
    a = np.asarray(semi_axes, dtype=np.float64)
    x = np.asarray(x)
    return (1.0 - np.sum((x / a) ** 2, axis=-1)) / (2.0 * np.sum(1.0 / a ** 2))


def ellipsoid_poisson_integral(semi_axes):
    """int u dV = (8 pi abc / 15) / (2 sum 1/a_i^2).  Sphere radius R: 4 pi R^5 / 45."""
    a = np.asarray(semi_axes, dtype=np.float64)
    return (8.0 * math.pi * float(np.prod(a)) / 15.0) / (2.0 * float(np.sum(1.0 / a ** 2)))


def ellipsoid_poisson_energy(semi_axes):
    """Dirichlet energy int |grad u|^2 dV = int u dV for -lap u = 1 (Green's identity)."""
    return ellipsoid_poisson_integral(semi_axes)


def cube_dirichlet_eigenvalue(sides):
    """First Dirichlet eigenvalue of -lap on a box with side lengths sides: pi^2 sum 1/L_i^2."""
    L = np.asarray(sides, dtype=np.float64)
    return math.pi ** 2 * float(np.sum(1.0 / L ** 2))


# ------------------------------------------------------------------------------------------------
# Prandtl torsion
# ------------------------------------------------------------------------------------------------
def prandtl_phi_circle(xy, R):
    xy = np.asarray(xy)
    return (R * R - np.sum(xy * xy, axis=-1)) / 2.0


def prandtl_phi_ellipse(xy, a, b):
    xy = np.asarray(xy)
    return (a * a * b * b / (a * a + b * b)) * (1.0 - (xy[..., 0] / a) ** 2 - (xy[..., 1] / b) ** 2)


def equilateral_triangle(side, center=(0.0, 0.0), angle=0.0):
    """Vertices of an equilateral triangle with centroid at center, one vertex at polar angle
    angle + pi/2."""
    return regular_polygon(3, side / math.sqrt(3.0), center=center, angle=angle)


def prandtl_phi_triangle(xy, side, center=(0.0, 0.0), angle=0.0):
    """phi = (2 / h) l1 l2 l3 with l_i the distances to the three sides (positive inside) and
    h = sqrt(3) side / 2 the height: lap(l1 l2 l3) = 2 sum_{i<j} n_i.n_j l_k = -(l1+l2+l3) = -h."""
    V = equilateral_triangle(side, center, angle)
    h = math.sqrt(3.0) * side / 2.0
    xy = np.asarray(xy)
    prod = 1.0
    for i in range(3):
        p, q = V[i], V[(i + 1) % 3]
        t = q - p
        n = np.array([t[1], -t[0]]) / np.linalg.norm(t)   # outward for counter-clockwise vertices
        prod = prod * (-(np.einsum("...i,i->...", xy - p, n)))
    return 2.0 / h * prod


def _rect_torsion(w, t, terms=2000):
    """Saint-Venant rectangle w x t (w >= t): J = w t^3 / 3 [1 - 192 t / (pi^5 w) sum_odd tanh(n pi w / 2t) / n^5].

    The truncated tail is below 1e-16 relative at 2000 odd terms (200 terms leave 7e-12)."""
    if t > w:
        w, t = t, w
    n = np.arange(1, 2 * terms, 2, dtype=np.float64)
    s = float(np.sum(np.tanh(n * math.pi * w / (2.0 * t)) / n ** 5))
    return w * t ** 3 / 3.0 * (1.0 - 192.0 * t / (math.pi ** 5 * w) * s)


def torsion_constant(shape, **d):
    """Saint-Venant torsion constant J (length^4), shear stiffness per unit length = G J.

    shape: 'circle' (R), 'ellipse' (a, b semi-axes), 'triangle' (side), 'rectangle' (w, t),
    'square' (side).  J is invariant under rotation of the section, which the engine's lattice is not.
    """
    if shape == "circle":
        return math.pi * d["R"] ** 4 / 2.0
    if shape == "ellipse":
        a, b = d["a"], d["b"]
        return math.pi * a ** 3 * b ** 3 / (a * a + b * b)
    if shape == "triangle":
        return math.sqrt(3.0) * d["side"] ** 4 / 80.0
    if shape == "rectangle":
        return _rect_torsion(d["w"], d["t"])
    if shape == "square":
        return _rect_torsion(d["side"], d["side"])
    raise ValueError(shape)


# ------------------------------------------------------------------------------------------------
# Conduction with Neumann walls and two Dirichlet electrodes
# ------------------------------------------------------------------------------------------------
def conductance_box(length, area, sigma=1.0):
    """Two electrode faces a distance length apart on a prism of cross-section area: sigma A / L."""
    return sigma * area / length


def conductance_torus_sector(R0, r, theta, sigma=1.0):
    """Solid torus (major R0, circular minor r), electrodes on two meridional half-planes an angle
    theta apart.  The potential is the toroidal angle, harmonic and tangent to every axisymmetric
    wall, so G = (sigma / theta) int_disc dA / R = sigma 2 pi (R0 - sqrt(R0^2 - r^2)) / theta."""
    return sigma * 2.0 * math.pi * (R0 - math.sqrt(R0 * R0 - r * r)) / theta


def conductance_annular_sector(R1, R2, height, theta, sigma=1.0):
    """Rectangular cross-section R1 < R < R2, height H, angle theta: sigma H ln(R2/R1) / theta."""
    return sigma * height * math.log(R2 / R1) / theta


# ------------------------------------------------------------------------------------------------
# Elasticity
# ------------------------------------------------------------------------------------------------
def shear_modulus(E, nu):
    return E / (2.0 * (1.0 + nu))


def torsion_twist_rate(torque, E, nu, J):
    """Saint-Venant: d(theta)/dz = T / (G J), radians per length, away from the ends."""
    return torque / (shear_modulus(E, nu) * J)


def axial_stiffness(E, area, length):
    return E * area / length


def lame_sphere_pressure(r, a, b, p_in, E, nu):
    """Thick-walled spherical shell a < r < b, internal pressure p_in, free outside: radial displacement u_r(r)."""
    r = np.asarray(r, dtype=np.float64)
    k = p_in * a ** 3 / (b ** 3 - a ** 3)
    return k / E * ((1.0 - 2.0 * nu) * r + (1.0 + nu) * b ** 3 / (2.0 * r * r))


def lame_cylinder_pressure(r, a, b, p_in, E, nu, plane="strain"):
    """Thick cylinder a < r < b, internal pressure: u_r(r), plane strain (default) or plane stress."""
    r = np.asarray(r, dtype=np.float64)
    A = p_in * a * a / (b * b - a * a)
    if plane == "strain":
        return (1.0 + nu) / E * A * ((1.0 - 2.0 * nu) * r + b * b / r)
    return A / E * ((1.0 - nu) * r + (1.0 + nu) * b * b / r)


# ------------------------------------------------------------------------------------------------
# polygons: exact straight-edge geometry (the "polygon face" representation in 2D)
# ------------------------------------------------------------------------------------------------
def regular_polygon(n, circumradius, center=(0.0, 0.0), angle=0.0):
    """Counter-clockwise vertices, first vertex at polar angle angle + pi/2."""
    k = np.arange(n)
    t = angle + math.pi / 2.0 + 2.0 * math.pi * k / n
    return np.stack([center[0] + circumradius * np.cos(t), center[1] + circumradius * np.sin(t)], 1)


def polygon_area(V):
    V = np.asarray(V, dtype=np.float64)
    x, y = V[:, 0], V[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_sdf(xy, V):
    """Exact signed distance to a simple polygon (negative inside), by nearest edge and winding."""
    xy = np.asarray(xy, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    P = xy.reshape(-1, 2)
    d2 = np.full(len(P), np.inf)
    wn = np.zeros(len(P), dtype=np.int64)
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        e = b - a
        w = P - a
        t = np.clip((w @ e) / (e @ e), 0.0, 1.0)
        d2 = np.minimum(d2, np.sum((w - t[:, None] * e) ** 2, axis=1))
        cr = e[0] * w[:, 1] - e[1] * w[:, 0]
        up = (a[1] <= P[:, 1]) & (b[1] > P[:, 1]) & (cr > 0)
        dn = (a[1] > P[:, 1]) & (b[1] <= P[:, 1]) & (cr < 0)
        wn += up.astype(np.int64) - dn.astype(np.int64)
    s = np.where(wn != 0, -1.0, 1.0)
    return (s * np.sqrt(d2)).reshape(xy.shape[:-1])
