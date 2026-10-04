"""Quantified CCD over endpoint boxes, interpolated skins and finite branches.

All arithmetic in certificates is exact on represented inputs. Completeness of
the supplied branch family is a caller model precondition. UNKNOWN never accepts
the full step. See docs/experimental_ccd_box.md for quantifiers and limitations.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
import math

SAFE, COLLISION, UNKNOWN = "SÄKER", "KOLLISION", "OSÄKER"


def exact(v):
    if isinstance(v, bool):
        raise ValueError("boolean is not a geometric coordinate")
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("finite coordinates required")
    q = Q(v)
    # Some Rational implementations (including NumPy integers) expose a
    # fixed-width numerator. Certificates require unbounded Python ints.
    return Q(int(q.numerator), int(q.denominator))


def _tensor(value, shape):
    if not shape:
        return exact(value)
    if len(value) != shape[0]:
        raise ValueError("wrong input shape")
    return tuple(_tensor(x, shape[1:]) for x in value)


@dataclass(frozen=True)
class Branch:
    centers: tuple
    radii: tuple
    skin_lower: tuple
    skin_upper: tuple
    id: str = "motion"

    def __post_init__(self):
        for name, shape in (("centers", (2, 4, 3)), ("radii", (2, 4, 3)),
                            ("skin_lower", (2, 4)), ("skin_upper", (2, 4))):
            object.__setattr__(self, name, _tensor(getattr(self, name), shape))
        if any(r < 0 for t in self.radii for v in t for r in v):
            raise ValueError("nonnegative box radii required")
        if any(a < 0 or a > b for ta, tb in zip(self.skin_lower, self.skin_upper)
               for a, b in zip(ta, tb)):
            raise ValueError("skin intervals must be ordered and nonnegative")

    @classmethod
    def from_arrays(cls, centers, radii=None, skin_lower=None, skin_upper=None, *, id="motion"):
        zeros = tuple(tuple((0, 0, 0) for _ in range(4)) for _ in range(2))
        skins = ((0, 0, 0, 0), (0, 0, 0, 0))
        return cls(centers, zeros if radii is None else radii,
                   skins if skin_lower is None else skin_lower,
                   skins if skin_upper is None else skin_upper, id)


def _dot(a, b):
    terms = tuple(x*y for x, y in zip(a, b))
    return terms[0]+terms[1]+terms[2]


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _lerp(a, b, t):
    return (1-t)*a+t*b


def _root_domains(kind):
    if kind == "PT":
        return (((Q(1),),), ((Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0)), (Q(0), Q(0), Q(1))))
    if kind == "EE":
        edge = ((Q(1), Q(0)), (Q(0), Q(1)))
        return (edge, edge)
    raise ValueError("kind must be PT or EE")


def _indices(kind):
    return ((0,), (1, 2, 3)) if kind == "PT" else ((0, 1), (2, 3))


def _relative(branch, kind, t, wa, wb):
    center, radius = [Q(0)]*3, [Q(0)]*3
    hlo, hhi = Q(0), Q(0)
    for sign, indices, weights in zip((1, -1), _indices(kind), (wa, wb)):
        for i, w in zip(indices, weights):
            for k in range(3):
                center[k] += sign*w*_lerp(branch.centers[0][i][k], branch.centers[1][i][k], t)
                radius[k] += w*_lerp(branch.radii[0][i][k], branch.radii[1][i][k], t)
            hlo += w*_lerp(branch.skin_lower[0][i], branch.skin_lower[1][i], t)
            hhi += w*_lerp(branch.skin_upper[0][i], branch.skin_upper[1][i], t)
    return tuple(center), tuple(radius), hlo, hhi


def _directions(branch, kind):
    dirs = [(Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0)), (Q(0), Q(0), Q(1))]
    for x in branch.centers:
        if kind == "PT":
            dirs.append(_cross(_sub(x[2], x[1]), _sub(x[3], x[1])))
        else:
            dirs.append(_cross(_sub(x[1], x[0]), _sub(x[3], x[2])))
        for i in _indices(kind)[0]:
            for j in _indices(kind)[1]:
                dirs.append(_sub(x[i], x[j]))
    return tuple(dict.fromkeys(n for n in dirs if any(n)))


def support_separated(branch, kind, interval=(Q(0), Q(1)), domains=None, directions=None):
    """Prove separation using licensed scalar projections, never distance corners."""
    domains = _root_domains(kind) if domains is None else domains
    corners = [_relative(branch, kind, t, a, b)
               for t, a, b in product(interval, domains[0], domains[1])]
    for n in _directions(branch, kind) if directions is None else directions:
        n2 = _dot(n, n)
        if not n2:
            continue
        for sign in (1, -1):
            good = True
            for c, r, _, h in corners:
                p = sign*_dot(c, n)-_dot(r, tuple(abs(x) for x in n))
                if p <= 0 or p*p <= h*h*n2:
                    good = False
                    break
            if good:
                return True
    # Interval inclusion of relative coordinates; not a corner claim for norm.
    lo = [min(c[k]-r[k] for c, r, _, _ in corners) for k in range(3)]
    hi = [max(c[k]+r[k] for c, r, _, _ in corners) for k in range(3)]
    d2 = sum((a if a > 0 else -b if b < 0 else Q(0))**2 for a, b in zip(lo, hi))
    h = max(row[3] for row in corners)
    return d2 > h*h


@dataclass(frozen=True)
class _I:
    lo: Q
    hi: Q

    def __add__(self, other):
        return _I(self.lo+other.lo, self.hi+other.hi)

    def __neg__(self):
        return _I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self+(-other)

    def __mul__(self, other):
        v = (self.lo*other.lo, self.lo*other.hi, self.hi*other.lo, self.hi*other.hi)
        return _I(min(v), max(v))


def _endpoint_intervals(branch, t):
    return tuple(tuple(_I(c-r, c+r) for c, r in zip(v, rad))
                 for v, rad in zip(branch.centers[t], branch.radii[t]))


def _orient(a, b, c, axes):
    i, j = axes
    return (b[i]-a[i])*(c[j]-a[j])-(b[j]-a[j])*(c[i]-a[i])


def universal_crossing(branch, kind):
    """All PT worlds cross a nondegenerate moving triangle; times need not agree.

    Endpoint volume intervals plus swept projected containment are sufficient.
    Interval inclusion is valid despite dependencies; no endpoint-only license
    is assumed for the cubic signed volume or the quadratic orientation in t.
    """
    if kind != "PT":
        return False
    ends = [_endpoint_intervals(branch, t) for t in (0, 1)]
    determinants = []
    for p, a, b, c in ends:
        determinants.append(_dot(_cross(_sub(b, a), _sub(c, a)), _sub(p, a)))
    a, b = determinants
    if not ((a.lo > 0 and b.hi < 0) or (a.hi < 0 and b.lo > 0)):
        return False
    swept = tuple(tuple(_I(min(ends[0][i][k].lo, ends[1][i][k].lo),
                            max(ends[0][i][k].hi, ends[1][i][k].hi))
                        for k in range(3)) for i in range(4))
    p, a, b, c = swept
    for axes in ((0, 1), (0, 2), (1, 2)):
        orientation = _orient(a, b, c, axes)
        edges = (_orient(a, b, p, axes), _orient(b, c, p, axes), _orient(c, a, p, axes))
        if orientation.lo > 0 and all(e.lo > 0 for e in edges):
            return True
        if orientation.hi < 0 and all(e.hi < 0 for e in edges):
            return True
    return False


def _probe_domains(branch, kind, t):
    domains = _root_domains(kind)
    mids = tuple(tuple(sum(v[k] for v in domain)/len(domain)
                       for k in range(len(domain[0]))) for domain in domains)
    aa, bb = domains[0]+(mids[0],), domains[1]+(mids[1],)
    if kind == "PT":
        p, a, b, c = tuple(tuple(_lerp(branch.centers[0][i][k], branch.centers[1][i][k], t)
                                 for k in range(3)) for i in range(4))
        for axes in ((0, 1), (0, 2), (1, 2)):
            d = _orient(a, b, c, axes)
            if d:
                w = (_orient(b, c, p, axes)/d, _orient(c, a, p, axes)/d, _orient(a, b, p, axes)/d)
                if all(z >= 0 for z in w):
                    bb += (w,)
    return aa, bb


def universal_skin_contact(branch, kind):
    """A fixed time and fixed simplex points contact for every endpoint world."""
    for t in (Q(0), Q(1, 2), Q(1)):
        aa, bb = _probe_domains(branch, kind, t)
        for a, b in product(aa, bb):
            c, r, h, _ = _relative(branch, kind, t, a, b)
            if sum((abs(x)+e)**2 for x, e in zip(c, r)) <= h*h:
                return True
    return False


def mixed_worlds(branch, kind, directions=None):
    """Prove both a free world and a contact world exist within one branch.

    Contact feasibility uses the exact fixed-time relative box, not distance
    extrema at world corners. Free feasibility uses one consistent endpoint
    assignment maximizing a fixed-direction projection with minimum skins.
    """
    domains = _root_domains(kind)
    corners = [_relative(branch, kind, t, a, b)
               for t, a, b in product((Q(0), Q(1)), domains[0], domains[1])]
    free = False
    for n in _directions(branch, kind) if directions is None else directions:
        n2 = _dot(n, n)
        for sign in (1, -1):
            if all((p := sign*_dot(c, n)+_dot(r, tuple(abs(x) for x in n))) > 0
                   and p*p > h*h*n2 for c, r, h, _ in corners):
                free = True
                break
        if free:
            break
    if not free:
        return False
    for t in (Q(0), Q(1, 2), Q(1)):
        aa, bb = _probe_domains(branch, kind, t)
        for a, b in product(aa, bb):
            c, r, _, h = _relative(branch, kind, t, a, b)
            d2 = sum(max(abs(x)-e, Q(0))**2 for x, e in zip(c, r))
            if d2 <= h*h:
                return True
    return False


def _split_domain(domain):
    pairs = [(sum((x-y)**2 for x, y in zip(domain[i], domain[j])), i, j)
             for i in range(len(domain)) for j in range(i+1, len(domain))]
    _, i, j = max(pairs)
    middle = tuple((x+y)/2 for x, y in zip(domain[i], domain[j]))
    left, right = list(domain), list(domain)
    left[j], right[i] = middle, middle
    return tuple(left), tuple(right)


@dataclass(frozen=True)
class BranchDecision:
    id: str
    status: str
    method: str
    safe_prefix: Q
    cells: int


@dataclass(frozen=True)
class BoxDecision:
    status: str
    safe_prefix: Q
    branches: tuple
    assurance: str = "MODEL_CONDITIONAL_ON_INPUT_AND_BRANCH_COVERAGE"

    def prefix_float(self):
        """Round a certified prefix down, including non-binary rational bounds."""
        v = float(self.safe_prefix)
        if Q(v) > self.safe_prefix:
            v = math.nextafter(v, -math.inf)
        return v


def _branch_decision(branch, kind, max_cells, max_depth, refine, prove_mixed):
    directions = _directions(branch, kind)
    if support_separated(branch, kind, directions=directions):
        return BranchDecision(branch.id, SAFE, "exact_uncertainty_support", Q(1), 1)
    if universal_crossing(branch, kind):
        return BranchDecision(branch.id, COLLISION, "universal_orientation_crossing", Q(0), 1)
    if universal_skin_contact(branch, kind):
        return BranchDecision(branch.id, COLLISION, "universal_fixed_time_skin", Q(0), 1)
    if prove_mixed and mixed_worlds(branch, kind, directions):
        return BranchDecision(branch.id, UNKNOWN, "proved_mixed_worlds", Q(0), 1)
    if all(r == 0 for t in branch.radii for v in t for r in v) and all(
            h == 0 for t in branch.skin_upper for h in t):
        # Exact reserve is only legal if conversion back to binary64 is exact.
        values = [c for t in branch.centers for v in t for c in v]
        try:
            representable = all(exact(float(c)) == c for c in values)
        except (OverflowError, ValueError):
            representable = False
        if representable:
            from .ccd_band import certify
            d = certify([[[float(c) for c in v] for v in t] for t in branch.centers], kind)
            return BranchDecision(branch.id, d.status, "reviewed_zero_box_reserve:"+d.method,
                                  Q(1) if d.status == SAFE else Q(0), 1)
    if not refine:
        return BranchDecision(branch.id, UNKNOWN, "unresolved_support", Q(0), 1)
    # A disjoint interior partition with shared closed boundaries; separation of
    # every closed cell proves separation of their union. Budget leaves UNKNOWN.
    stack = [(Q(0), Q(1), _root_domains(kind), 0)]
    unresolved, cells = [], 0
    while stack and cells < max_cells:
        lo, hi, domains, depth = stack.pop()
        cells += 1
        if support_separated(branch, kind, (lo, hi), domains, directions):
            continue
        if depth >= max_depth:
            unresolved.append(lo)
            continue
        if depth % 3 == 0:
            mid = (lo+hi)/2
            stack.extend(((mid, hi, domains, depth+1), (lo, mid, domains, depth+1)))
        else:
            group = 1 if kind == "PT" else (depth % 2)
            da, db = _split_domain(domains[group])
            for subdomain in (db, da):
                child = list(domains); child[group] = subdomain
                stack.append((lo, hi, tuple(child), depth+1))
    unresolved.extend(row[0] for row in stack)
    if unresolved:
        return BranchDecision(branch.id, UNKNOWN, "bounded_time_simplex_inclusion", min(unresolved), cells)
    return BranchDecision(branch.id, SAFE, "refined_time_simplex_inclusion", Q(1), cells)


def certify_boxes(branches, kind, *, complete, max_cells=512, max_depth=18, refine=True, prove_mixed=True):
    """Universally quantify endpoint worlds and then the finite motion branch union.

    Complete=True asserts upstream coverage; it does not verify dynamics. Any
    unlisted motion or geometry outside supplied boxes voids that model guarantee.
    safe_prefix certifies [0,prefix) when prefix<1, closed [0,1] when SAFE.
    """
    _root_domains(kind)
    branches = tuple(branches)
    if any(not isinstance(b, Branch) for b in branches):
        raise TypeError("branches must be immutable Branch objects")
    if type(complete) is not bool:
        raise TypeError("complete must explicitly be True or False")
    if type(max_cells) is not int or type(max_depth) is not int or max_cells < 0 or max_depth < 0:
        raise ValueError("nonnegative integer budgets required")
    if type(refine) is not bool or type(prove_mixed) is not bool:
        raise TypeError("refine and prove_mixed must be boolean")
    if not complete or not branches:
        return BoxDecision(UNKNOWN, Q(0), ())
    if max_cells == 0:
        out = tuple(BranchDecision(b.id, UNKNOWN, "zero_budget", Q(0), 0) for b in branches)
    else:
        out = tuple(_branch_decision(b, kind, max_cells, max_depth, refine, prove_mixed) for b in branches)
    status = SAFE if all(d.status == SAFE for d in out) else COLLISION if all(
        d.status == COLLISION for d in out) else UNKNOWN
    return BoxDecision(status, min(d.safe_prefix for d in out), out)
