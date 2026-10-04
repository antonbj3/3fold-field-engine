"""Exact affine gap over closed projected triangle intersections.

Coordinates are exact represented numbers (binary floats become dyadic rationals).
This certifies the supplied facets under vertical projection, with a common unit.
It does not enclose uncertain coordinates or certify general 3D collision.
Degenerate projections return UNKNOWN. No geometric tolerance or floating prune.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
import math


def _q(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Q)):
        raise ValueError('finite represented rational required')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite coordinate')
    return Q(value)


def _triangle(vertices):
    tri = tuple(tuple(_q(v) for v in p) for p in vertices)
    if len(tri) != 3 or any(len(p) != 3 for p in tri):
        raise ValueError('three 3D vertices required')
    return tri


def _cross(a, b):
    return a[0]*b[1] - a[1]*b[0]


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1])


def _orientation(t):
    return _cross(_sub(t[1], t[0]), _sub(t[2], t[0]))


def _inside(p, t):
    sign = 1 if _orientation(t) > 0 else -1
    return all(sign*_cross(_sub(t[(i+1) % 3], t[i]), _sub(p, t[i])) >= 0
               for i in range(3))


def _height(t, p):
    det = _orientation(t)
    a = _cross(_sub(p, t[0]), _sub(t[2], t[0]))/det
    b = _cross(_sub(t[1], t[0]), _sub(p, t[0]))/det
    return (1-a-b)*t[0][2] + a*t[1][2] + b*t[2][2]


def outward_float(value):
    """Smallest binary64 bracket; overflow/underflow never narrow the bracket."""
    value = _q(value)
    try:
        f = float(value)
    except OverflowError:
        f = math.inf if value > 0 else -math.inf
    if f == math.inf:
        return math.nextafter(f, -math.inf), f
    if f == -math.inf:
        return f, math.nextafter(f, math.inf)
    exact_f = Q(f)
    return (math.nextafter(f, -math.inf) if exact_f > value else f,
            math.nextafter(f, math.inf) if exact_f < value else f)


@dataclass(frozen=True)
class ProjectedGap:
    status: str
    unit: str
    exact: Q | None = None
    lower: float | None = None
    upper: float | None = None
    witness_xy: tuple | None = None
    reason: str = ''


def triangle_gap(upper, lower, *, unit):
    """Minimum z_upper-z_lower, including shared points and collinear edges."""
    if unit not in ('m', 'mm'):
        raise ValueError('explicit common length unit m or mm required')
    u, l = _triangle(upper), _triangle(lower)
    if not _orientation(u) or not _orientation(l):
        return ProjectedGap('UNKNOWN', unit, reason='degenerate projected facet')
    points = {p[:2] for p in u if _inside(p, l)}
    points.update(p[:2] for p in l if _inside(p, u))
    for i in range(3):
        a, av = u[i], _sub(u[(i+1) % 3], u[i])
        for j in range(3):
            b, bv = l[j], _sub(l[(j+1) % 3], l[j])
            den = _cross(av, bv)
            if den:
                delta = _sub(b, a)
                t, s = _cross(delta, bv)/den, _cross(delta, av)/den
                if 0 <= t <= 1 and 0 <= s <= 1:
                    points.add((a[0]+t*av[0], a[1]+t*av[1]))
    if not points:
        return ProjectedGap('EMPTY', unit, reason='disjoint closed projections')
    value, p = min((_height(u, p)-_height(l, p), p) for p in points)
    lo, hi = outward_float(value)
    return ProjectedGap('CERTIFIED', unit, value, lo, hi, p,
                        'exact represented affine facets')


def surface_gap(upper_facets, lower_facets, *, unit):
    """Exhaustive all-pair query; any degenerate facet prevents global certification."""
    upper_facets, lower_facets = tuple(upper_facets), tuple(lower_facets)
    if unit not in ('m', 'mm'):
        raise ValueError('explicit common length unit m or mm required')
    results = [triangle_gap(u, l, unit=unit) for u in upper_facets for l in lower_facets]
    if any(r.status == 'UNKNOWN' for r in results):
        return ProjectedGap('UNKNOWN', unit, reason='unresolved pair in complete search')
    values = [r for r in results if r.status == 'CERTIFIED']
    return min(values, key=lambda r: r.exact) if values else ProjectedGap('EMPTY', unit)
