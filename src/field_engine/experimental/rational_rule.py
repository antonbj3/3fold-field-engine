"""Exact rational replay of cellwise quadratic surrogate preference rules.

The declared score is max_i sqrt(max(0, theta_hat' Q_i theta_hat)). Interval
matrices cover coefficients and affine guards cover branch validity. Every
non-excluded branch pair and every selected component must have a strict
dominance witness. Anchor guards can require, e.g., nonnegative signed outputs.
Upstream surrogate-to-observable binding and coefficient accuracy are separate.
Branch inequalities alone are conditional on active branches. A whole-box
rule additionally needs an exactly verified branch cover for each design.
No corner-only inference, floating sqrt, or empty conjunction grants a rule.
"""
from dataclasses import dataclass
from fractions import Fraction as Q


def _q(v):
    if isinstance(v, bool) or not isinstance(v, (int, float, str, Q)):
        raise ValueError('finite represented rational required')
    return Q(v)


def _iv(v):
    lo, hi = map(_q, v)
    if lo > hi:
        raise ValueError('reversed interval')
    return lo, hi


def _add(a, b):
    return a[0]+b[0], a[1]+b[1]


def _mul(a, b):
    vals = [x*y for x in a for y in b]
    return min(vals), max(vals)


def _box(box):
    b = tuple(_iv(v) for v in box)
    if not b or len(b) > 8 or any(lo >= hi for lo, hi in b):
        raise ValueError('positive-volume box, dimension 1..8 required')
    return b


def affine_range(coeff, box):
    box = _box(box)
    if len(coeff) != len(box)+1:
        raise ValueError('affine dimension mismatch')
    out = _iv(coeff[0])
    for c, x in zip(coeff[1:], box):
        out = _add(out, _mul(_iv(c), x))
    return out


def quadratic_weights(box):
    b = _box(box)
    c = [(Q(1), Q(1))] + [((lo+hi)/2, (lo+hi)/2) for lo, hi in b]
    r = [(Q(0), Q(0))] + [(lo-cc[0], hi-cc[0]) for (lo, hi), cc in zip(b, c[1:])]
    n = len(c)
    return [[_add(_add(_add(_mul(c[i], c[j]), _mul(c[i], r[j])),
                            _mul(r[i], c[j])), _mul(r[i], r[j]))
             for j in range(n)] for i in range(n)]


def quadratic_range(matrix, weights):
    n = len(weights)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError('quadratic dimension mismatch')
    out = (Q(0), Q(0))
    for i in range(n):
        for j in range(n):
            out = _add(out, _mul(_iv(matrix[i][j]), weights[i][j]))
    return out


def exact_partition(root_box, cells):
    """Verify a guillotine tiling, including shared closed boundaries.

    Partition volume alone cannot detect overlapping cells plus equal-size holes.
    Returns False for invalid, missing, duplicate or non-guillotine partitions.
    """
    try:
        root = _box(root_box)
        boxes = tuple(_box(c) for c in cells)
        if not boxes or len(set(boxes)) != len(boxes):
            return False
        if any(len(b) != len(root) or any(lo < a or hi > z for (lo, hi), (a, z) in zip(b, root)) for b in boxes):
            return False
        pending = [(root, boxes)]
        while pending:
            region, pieces = pending.pop()
            if len(pieces) == 1:
                if pieces[0] != region:
                    return False
                continue
            split = None
            for axis, (lo, hi) in enumerate(region):
                cuts = sorted({b[axis][1] for b in pieces if lo < b[axis][1] < hi})
                for cut in cuts:
                    left = tuple(b for b in pieces if b[axis][1] <= cut)
                    right = tuple(b for b in pieces if b[axis][0] >= cut)
                    if left and right and len(left)+len(right) == len(pieces):
                        split = axis, cut, left, right
                        break
                if split:
                    break
            if split is None:
                return False
            axis, cut, left, right = split
            lreg, rreg = list(region), list(region)
            lreg[axis], rreg[axis] = (region[axis][0], cut), (cut, region[axis][1])
            pending.extend([(tuple(lreg), left), (tuple(rreg), right)])
        return True
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return False


@dataclass(frozen=True)
class RuleReplay:
    status: str
    reason: str
    cells: int = 0
    inequalities: int = 0
    domain_coverage: str = 'UNKNOWN'
    covered_cells: int = 0


def replay_rule(root_box, cells, models, *, selected, other, margin):
    """Models: design -> branch id -> components, guards, anchor_guards.

    Cells use `box`, `potential_masks`, and `pairs` containing selected_mask,
    other_mask, other_anchor, and one inequality per selected_component. All
    reported numeric witness bounds are checked against recomputed exact ranges.
    """
    count = 0
    covered_cells = 0
    try:
        margin = _q(margin)
        if margin < 0 or selected == other or not models[selected] or not models[other]:
            return RuleReplay('UNKNOWN', 'nonempty distinct design families required')
        cells = tuple(cells)
        if not exact_partition(root_box, [c['box'] for c in cells]):
            return RuleReplay('UNKNOWN', 'unverified partition')
        for cell in cells:
            box = cell['box']
            weights = quadratic_weights(box)
            possible = {}
            for design in (selected, other):
                possible[design] = {m for m, b in models[design].items()
                                    if not any(affine_range(g, box)[0] > 0 for g in b.get('guards', []))}
                declared = cell['potential_masks'][design]
                if not possible[design] or len(set(declared)) != len(declared) or not possible[design] <= set(declared) or not set(declared) <= set(models[design]):
                    return RuleReplay('UNKNOWN', 'empty or omitted branch')
            # A branch whose every guard has upper bound <= 0 is active
            # throughout the cell for every represented coefficient value.
            # This is a sufficient cover; harder unions retain UNKNOWN coverage.
            if all(any(all(affine_range(g, box)[1] <= 0
                           for g in branch.get('guards', []))
                       for branch in models[design].values())
                   for design in (selected, other)):
                covered_cells += 1
            pairs = cell['pairs']
            required = {(a, b) for a in cell['potential_masks'][selected] for b in cell['potential_masks'][other]}
            if len(pairs) != len(required) or {(p['selected_mask'], p['other_mask']) for p in pairs} != required:
                return RuleReplay('UNKNOWN', 'missing or duplicate branch pair')
            for pair in pairs:
                a = models[selected][pair['selected_mask']]
                b = models[other][pair['other_mask']]
                anchor = pair['other_anchor']
                if type(anchor) is not int or not 0 <= anchor < len(b['components']):
                    return RuleReplay('UNKNOWN', 'invalid anchor')
                if any(affine_range(g, box)[0] < 0 for g in b.get('anchor_guards', {}).get(anchor, [])):
                    return RuleReplay('UNKNOWN', 'unlicensed signed anchor')
                inequalities = pair['inequalities']
                if not a['components'] or len(inequalities) != len(a['components']) or {r['selected_component'] for r in inequalities} != set(range(len(a['components']))):
                    return RuleReplay('UNKNOWN', 'missing or duplicate component')
                for row in inequalities:
                    sq = a['components'][row['selected_component']]
                    bq = b['components'][anchor]
                    # Dimension checking occurs independently before paired zip.
                    sqmax = max(Q(0), quadratic_range(sq, weights)[1])
                    other_lower = quadratic_range(bq, weights)[0]
                    diff = [[(_iv(v)[0]-_iv(w)[1], _iv(v)[1]-_iv(w)[0])
                             for v, w in zip(br, sr)] for br, sr in zip(bq, sq)]
                    dl = quadratic_range(diff, weights)[0]
                    u = _q(row['selected_norm_upper'])
                    claimed = _q(row['difference_squared_lower'])
                    rhs = _q(row['rhs_upper'])
                    # q_other-max(0,q_selected) = min(q_other-q_selected,q_other).
                    # Both lower bounds are required for clamped quadratic scores.
                    if u < 0 or u*u < sqmax or claimed > dl or rhs < 2*margin*u+margin*margin or claimed <= rhs or other_lower <= rhs:
                        return RuleReplay('UNKNOWN', 'invalid dominance inequality', len(cells), count)
                    count += 1
        covered = covered_cells == len(cells)
        return RuleReplay('CERTIFIED_SURROGATE_RULE' if covered else 'CERTIFIED_BRANCH_CONDITIONAL_RULE',
                          'exact cell tiling and all branch/component inequalities; '
                          + ('verified whole-box branch cover' if covered else 'branch existence/coverage remains UNKNOWN'),
                          len(cells), count, 'CERTIFIED' if covered else 'UNKNOWN', covered_cells)
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError):
        return RuleReplay('UNKNOWN', 'invalid exact proof input', inequalities=count)
