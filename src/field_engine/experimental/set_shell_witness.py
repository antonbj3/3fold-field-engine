"""Exact shape-independent obstruction under explicit digital set requirements.

Q=P Minkowski-summed with a radius-R Euclidean ball. T and P are finite unions
of closed axis-aligned boxes. If q in Q, o outside T, and |q-o| <= w, then no K
satisfies Q subset K and K+B_w subset T. This is a necessary-condition witness;
absence of a witness does not prove feasibility. Anatomy and model binding are
separate. All coordinates and radii share the explicit unit.
"""
from dataclasses import dataclass
from fractions import Fraction as Q


def _q(v):
    if isinstance(v, bool) or not isinstance(v, (int, float, str, Q)):
        raise ValueError('finite represented rational required')
    return Q(v)


def _point(p):
    p = tuple(_q(v) for v in p)
    if not p:
        raise ValueError('empty coordinate vector')
    return p


def _boxes(boxes, n):
    out = []
    for box in boxes:
        lo, hi = map(_point, box)
        if len(lo) != n or len(hi) != n or any(a > b for a, b in zip(lo, hi)):
            raise ValueError('invalid closed box')
        out.append((lo, hi))
    return tuple(out)


def _distance_squared(p, box):
    lo, hi = box
    return sum((max(a-x, Q(0), x-b)**2 for x, a, b in zip(p, lo, hi)), Q(0))


@dataclass(frozen=True)
class ShellWitness:
    status: str
    unit: str
    reason: str
    protected_distance_squared: Q | None = None
    shell_distance_squared: Q | None = None


def verify_shell_witness(source_boxes, protected_boxes, protected_radius,
                         shell_width, q, outside, *, unit):
    if unit not in ('m', 'mm'):
        raise ValueError('explicit common length unit m or mm required')
    q, o = _point(q), _point(outside)
    if len(q) != len(o):
        raise ValueError('dimension mismatch')
    T, P = _boxes(source_boxes, len(q)), _boxes(protected_boxes, len(q))
    R, w = _q(protected_radius), _q(shell_width)
    if R < 0 or w < 0:
        raise ValueError('negative Euclidean radius')
    if not P:
        return ShellWitness('UNKNOWN', unit, 'empty mandatory set')
    pd = min(_distance_squared(q, box) for box in P)
    od = sum(((x-y)**2 for x, y in zip(q, o)), Q(0))
    # A point on a closed source-box boundary is inside T.
    outside_all = all(_distance_squared(o, box) > 0 for box in T)
    valid = pd <= R*R and od <= w*w and outside_all
    return ShellWitness('INFEASIBLE' if valid else 'UNKNOWN', unit,
                        'exact set obstruction' if valid else 'no verified obstruction', pd, od)


def sdf_shell_margin(maximum_interval, shell_width):
    """Interval for sup_Q phi_T + w; supplied numeric endpoints are not a proof.

    Routing needs a verified Euclidean SDF AND verified supremum over the full Q.
    This arithmetic helper returns no feasibility status or caller-flag license.
    """
    lo, hi = map(_q, maximum_interval)
    w = _q(shell_width)
    if lo > hi or w < 0:
        raise ValueError('invalid bound or radius')
    return lo+w, hi+w


@dataclass(frozen=True)
class BoxShell:
    status: str
    unit: str
    extent_margin: Q


def verify_box_shell(source_box, mandatory_box, shell_width, *, unit):
    """Recompute Q+B_w subset T for two explicit closed boxes, any dimension.

    Every coordinate's support is shifted by exactly w under Euclidean dilation.
    A positive face excess supplies a shape-independent obstruction; containment
    passes only this necessary set condition, not other shape/material conditions.
    """
    if unit not in ('m', 'mm'):
        raise ValueError('explicit common length unit m or mm required')
    n = len(source_box[0])
    T, Qbox = _boxes([source_box], n)[0], _boxes([mandatory_box], n)[0]
    w = _q(shell_width)
    if w < 0:
        raise ValueError('negative Euclidean radius')
    margin = max([hi+w-thi for hi, thi in zip(Qbox[1], T[1])] +
                 [tlo+w-lo for lo, tlo in zip(Qbox[0], T[0])])
    return BoxShell('INFEASIBLE' if margin > 0 else 'CONTAINED', unit, margin)
