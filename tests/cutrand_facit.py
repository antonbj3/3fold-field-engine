"""Exact classical fixtures from the reviewed FACIT klassiska.py functions.

Kept in tests so the boundary patch works above the null-space patch alone. Benchmark probes use
the original FACIT API; the source hash and fixture extraction are recorded in SOURCE_MANIFEST.json.
"""
import math
import numpy as np

def ellipsoid_poisson_integral(semi_axes):
    """int u dV = (8 pi abc / 15) / (2 sum 1/a_i^2).  Sphere radius R: 4 pi R^5 / 45."""
    a = np.asarray(semi_axes, dtype=np.float64)
    return (8.0 * math.pi * float(np.prod(a)) / 15.0) / (2.0 * float(np.sum(1.0 / a ** 2)))


def equilateral_triangle(side, center=(0.0, 0.0), angle=0.0):
    """Vertices of an equilateral triangle with centroid at center, one vertex at polar angle
    angle + pi/2."""
    return regular_polygon(3, side / math.sqrt(3.0), center=center, angle=angle)


def _rect_torsion(w, t, terms=200):
    """Saint-Venant rectangle w x t (w >= t): J = w t^3 / 3 [1 - 192 t / (pi^5 w) sum_odd tanh(n pi w / 2t) / n^5]."""
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


def regular_polygon(n, circumradius, center=(0.0, 0.0), angle=0.0):
    """Counter-clockwise vertices, first vertex at polar angle angle + pi/2."""
    k = np.arange(n)
    t = angle + math.pi / 2.0 + 2.0 * math.pi * k / n
    return np.stack([center[0] + circumradius * np.cos(t), center[1] + circumradius * np.sin(t)], 1)


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
