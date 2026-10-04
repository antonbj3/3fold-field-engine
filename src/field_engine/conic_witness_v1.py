"""Exact supplied-witness checks; these functions do not solve feasibility.

For Ax=b, x belongs to a product of cones in column order. 'nonnegative'
means R_+^d, 'soc' means {(t,u): t>=||u||_2}, and 'free' means R^d.
A valid y satisfies A.T y in the dual product and y.b<0. A rejected witness
returns UNKNOWN, including weak infeasibility. Static model certificates
have no automatic implication for dynamics, measured geometry or rounding.

Inputs are int, Fraction, or exact rational strings. Floats and bools are
rejected; a decimal string denotes that exact rational model value.
"""
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json


def _q(x):
    if isinstance(x, bool) or not isinstance(x, (int, Fraction, str)):
        raise TypeError('expected int, Fraction or rational string')
    return Fraction(x)


def _matrix(rows):
    rows = tuple(tuple(_q(x) for x in row) for row in rows)
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('expected nonempty rectangular matrix')
    return rows


def _digest(payload):
    return sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class WitnessCheck:
    status: str
    margin: Fraction
    model_sha256: str
    reason: str


def check_conic_witness(A, b, blocks, y):
    """Return INFEASIBLE only after exact verification against these inputs.

    blocks is a sequence of (kind, dimension) in column order. A physical
    Coulomb cone must first be mapped to the stated standard Lorentz cone.
    Witness size is len(b); verification reads A and all cone blocks.
    """
    A = _matrix(A)
    b, y = tuple(map(_q, b)), tuple(map(_q, y))
    blocks = tuple(tuple(block) for block in blocks)
    if len(b) != len(A) or len(y) != len(A):
        raise ValueError('right-hand-side and witness dimensions must match rows')
    for block in blocks:
        if len(block) != 2:
            raise ValueError('block must be (kind, dimension)')
        kind, size = block
        if kind not in ('nonnegative', 'soc', 'free'):
            raise ValueError('unsupported cone')
        if type(size) is not int or size < (2 if kind == 'soc' else 1):
            raise ValueError('invalid cone dimension')
    if sum(size for _, size in blocks) != len(A[0]):
        raise ValueError('cone dimensions must match columns')
    digest = _digest([[[str(x) for x in r] for r in A], list(map(str, b)), blocks])
    pullback = tuple(sum(row[j] * wi for row, wi in zip(A, y)) for j in range(len(A[0])))
    margin = -sum(bi * wi for bi, wi in zip(b, y))
    start = 0
    for kind, size in blocks:
        v = pullback[start:start + size]
        start += size
        if kind == 'nonnegative':
            ok = all(x >= 0 for x in v)
        elif kind == 'free':
            ok = all(x == 0 for x in v)
        else:
            ok = v[0] >= 0 and v[0] ** 2 >= sum(x*x for x in v[1:])
        if not ok:
            return WitnessCheck('UNKNOWN', margin, digest, 'dual membership failed')
    if margin <= 0:
        return WitnessCheck('UNKNOWN', margin, digest, 'strict separation absent')
    return WitnessCheck('INFEASIBLE', margin, digest, 'exact conic separator')


def check_strict_support(normals, gaps, weights):
    """For n_i.t+delta<=gap_i, certify that delta>0 is impossible.

    This also handles zero optimum: closed containment can still be feasible.
    Normals/gaps must describe true supporting inequalities of the model.
    Three selected supports can suffice in the planar translation problem.
    """
    normals = _matrix(normals)
    gaps, weights = tuple(map(_q, gaps)), tuple(map(_q, weights))
    if len(gaps) != len(normals) or len(weights) != len(normals):
        raise ValueError('support dimensions differ')
    digest = _digest([[[str(x) for x in r] for r in normals], list(map(str, gaps))])
    bound = sum(w*g for w, g in zip(weights, gaps))
    valid = all(w >= 0 for w in weights) and sum(weights) == 1
    valid = valid and all(sum(w*n[j] for w, n in zip(weights, normals)) == 0
                          for j in range(len(normals[0])))
    if valid and bound <= 0:
        return WitnessCheck('NO_STRICT_FIT', -bound, digest, 'balanced margin upper bound')
    return WitnessCheck('UNKNOWN', -bound, digest, 'no checked nonpositive margin bound')
