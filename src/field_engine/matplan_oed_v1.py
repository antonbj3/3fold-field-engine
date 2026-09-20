#!/usr/bin/env python3
"""Which points to measure on a part, chosen by what they would tell you about its dimensions.

A drawing declares dimensions; an inspection measures points. The link between them is the
sensitivity of the surface to each declared parameter, and it is available directly from the
expression tree the part was designed as: d(sdf)/d(theta) at a surface point is exactly how far that
point moves when that dimension moves. Each candidate probe therefore contributes a rank-one Fisher
matrix J J^T / sigma^2, and choosing an inspection plan is choosing a SET of them.

The naive plan is to spread probes evenly over the part. That is the plan that fails, and it fails
structurally rather than by a little: an even spread can cover the whole surface and still never
excite one parameter, and a parameter no probe excites is not estimated at all, however many probes
are taken. On a mounting plate with six declared dimensions -- width, height, thickness, boss radius,
hole radius, hole spacing -- measured with 8 probes:

    even spread   rank 5 of 6,  sigma_min exactly 0,  mean CRB 168.845 um  (worst 1000 um)
    chosen        rank 6 of 6,  sigma_min 3.73e10,    mean CRB   3.386 um  (worst 4.9 um)

and doubling the even plan to 16 probes does not help: still rank 5 of 6, still 168.192 um. It is
not a resolution problem. The parameter an even spread never excites stays unestimated, and the
prior alone then sets its bound -- which is what the 1000 um worst case is: the prior, untouched.

The two rank-deficient readings above are identical to the run this was measured in, to six digits.
The chosen plan's mean CRB is 3.386 um here against 3.3166 um there, a 2 % difference from one
detail of the tie-break: that run rotated between candidates that tie exactly on both keys, where
this module keeps the first. Both reach rank 6 of 6 at k = 8; the carried numbers are in the data
beside this module.

WHY THE SELECTION RULE IS RANK FIRST. While the accumulated Fisher is rank-deficient, sigma_min is
exactly zero for EVERY candidate, because any parameter nothing has excited contributes a zero
column. An E-optimal argmax therefore has no gradient at all over the whole regime k < K and
degenerates to input order, buying redundant copies of one probe. `greedy_oed` orders candidates
lexicographically by (rank gain per cost, information gain in bits per cost): on the plateau only
the rank term moves, and once the rank is full the priced objective takes over. `tie_break="sigma"`
reproduces the ungradiented rule so the defect stays measurable rather than merely described.

The information gain is log-determinant, hence submodular, which is what makes greedy on it worth
running; a sigma_min-per-cost objective is not submodular and greedy on it carries no such bound.

WHAT THIS DOES NOT DO. It chooses WHERE to measure given a part and a sigma. It does not decide
whether the resulting tolerance is acceptable -- turning a plan into an accept/reject decision, with
a probability that the verdict flips, is a separate instrument and is not in this module.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ikarus_v1"))
import expr as IKE                                                              # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "laplaceflode_v1")
MM = 1000.0


class Monteringsplatta:
    """A mounting plate with a central boss and a four-hole pattern, as an expression tree.

    theta = (width, height, thickness, boss radius, hole radius, hole spacing in x), all in metres.
    The hole spacing in y and the boss height are fixed, so six dimensions are declared."""

    parameternamn = ("W", "H", "T", "R_boss", "R_hal", "Dx")
    nominell_theta = np.array([0.100, 0.080, 0.015, 0.020, 0.0045, 0.070])
    H_BOSS = 0.010
    DY = 0.050

    @classmethod
    def trad_mm(cls, theta):
        W, H, T, R_boss, R_hal, Dx = (float(v) for v in theta)
        bas = IKE.box((W * MM / 2, H * MM / 2, T * MM / 2))
        boss = IKE.cylinder(radius=R_boss * MM, height=cls.H_BOSS * MM, axis="z",
                            center=(0.0, 0.0, (T / 2 + cls.H_BOSS / 2) * MM))
        kropp = IKE.union(bas, boss)
        hal = None
        for sx in (1, -1):
            for sy in (1, -1):
                c = IKE.cylinder(radius=R_hal * MM, height=T * 3 * MM, axis="z",
                                 center=(sx * Dx * MM / 2, sy * cls.DY * MM / 2, 0.0))
                hal = c if hal is None else IKE.union(hal, c)
        return IKE.subtract(kropp, hal)

    @classmethod
    def sdf(cls, p_m, theta):
        return IKE.eval_sdf_py(cls.trad_mm(theta), tuple(float(v) * MM for v in p_m)) / MM

    @classmethod
    def kandidatpunkter(cls, theta, n=120):
        """Candidate probe points on the declared surfaces, in a fixed deterministic order.

        The order matters: it is what an even plan samples along, and what a degenerate selection
        rule falls back on. Both baselines in this module read the same list."""
        W, H, T, R_boss, R_hal, Dx = (float(v) for v in theta)
        p = []
        for y in np.linspace(-H * 0.35, H * 0.35, 5):
            for z in (-T * 0.25, 0.0, T * 0.25):
                p += [[W / 2, y, z], [-W / 2, y, z]]
        for x in np.linspace(-W * 0.35, W * 0.35, 5):
            for z in (-T * 0.25, 0.0, T * 0.25):
                p += [[x, H / 2, z], [x, -H / 2, z]]
        centra = [(sx * Dx / 2, sy * cls.DY / 2) for sx in (1, -1) for sy in (1, -1)]
        for x in np.linspace(-W * 0.40, W * 0.40, 5):
            for y in np.linspace(-H * 0.40, H * 0.40, 5):
                if math.hypot(x, y) > R_boss + 0.004 and all(
                        math.hypot(x - cx, y - cy) > R_hal + 0.003 for cx, cy in centra):
                    p.append([x, y, T / 2])
        for a in np.linspace(0.0, 2 * np.pi, 12, endpoint=False):
            for z in (T / 2 + cls.H_BOSS * 0.3, T / 2 + cls.H_BOSS * 0.7):
                p.append([R_boss * math.cos(a), R_boss * math.sin(a), z])
        for r in (R_boss * 0.3, R_boss * 0.7):
            for a in np.linspace(0.0, 2 * np.pi, 6, endpoint=False):
                p.append([r * math.cos(a), r * math.sin(a), T / 2 + cls.H_BOSS])
        for cx, cy in centra:
            for a in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
                for z in (-T * 0.3, 0.0, T * 0.3):
                    p.append([cx + R_hal * math.cos(a), cy + R_hal * math.sin(a), z])
        return np.array(p, dtype=np.float64)[:n]


def kanslighet(del_, p, theta, h=1.0e-6):
    """d(sdf)/d(theta) at one surface point, by central differences on the tree."""
    theta = np.asarray(theta, dtype=np.float64)
    J = np.zeros(len(theta))
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        J[j] = (del_.sdf(p, tp) - del_.sdf(p, tm)) / (2 * h)
    return J


def fisherpool(del_, theta, punkter, sigma=2.0e-6):
    """One rank-one Fisher matrix per candidate probe."""
    return [np.outer(J, J) / sigma ** 2
            for J in (kanslighet(del_, p, theta) for p in punkter)]


def _spektrum(F, tol=1e-10):
    """(sigma_min, rank, largest eigenvalue). The rank tolerance is RELATIVE to the largest
    eigenvalue, so the reading does not change when the Fisher is expressed in other units.

    Fails closed on a non-finite matrix: a probe that could not be evaluated must not be
    eigendecomposed, because the decomposition of a matrix with a NaN in it is not merely wrong, it
    is unordered, and the smallest eigenvalue read off it is then arbitrary."""
    F = np.asarray(F, dtype=np.float64)
    if not np.all(np.isfinite(F)):
        raise ValueError("Fisher matrix carries non-finite entries and cannot be decomposed")
    w = np.linalg.eigvalsh(0.5 * (F + F.T))
    topp = max(float(w[-1]), 1e-300)
    return float(max(float(w[0]), 0.0)), int((w > tol * topp).sum()), topp


def informationsbitar(ack, F, prior):
    """Gaussian information gain in bits of adding F: 1/2 log2 det(I + (ack + prior)^-1 F)."""
    A = 0.5 * ((ack + prior) + (ack + prior).T)
    t1, d1 = np.linalg.slogdet(A)
    t2, d2 = np.linalg.slogdet(0.5 * ((A + F) + (A + F).T))
    if t1 <= 0 or t2 <= 0:
        raise ValueError("prior plus accumulated Fisher is not positive definite")
    return float((d2 - d1) / (2 * np.log(2.0)))


def greedy_oed(pool, k, prior, tie_break="rank"):
    """Pick k probes. Returns (indices, sigma_min after each pick).

    tie_break="rank"  the lexicographic rule: rank gain first, information gain in bits second.
    tie_break="sigma" the ungradiented E-optimal argmax, kept so its failure stays reproducible."""
    if tie_break not in ("rank", "sigma"):
        raise ValueError(f"tie_break must be 'rank' or 'sigma': {tie_break!r}")
    pool = [np.asarray(F, dtype=np.float64) for F in pool]
    K = pool[0].shape[0]
    P = np.asarray(prior, dtype=np.float64)
    ack = np.zeros((K, K))
    _, r_ack, _ = _spektrum(ack)
    valda, spar, kvar = [], [], list(range(len(pool)))
    for _ in range(min(k, len(pool))):
        bast, bast_nyckel = None, None
        for i in kvar:
            s, r, _ = _spektrum(ack + pool[i])
            nyckel = (s,) if tie_break == "sigma" else (r - r_ack,
                                                        informationsbitar(ack, pool[i], P))
            if bast_nyckel is None or nyckel > bast_nyckel:
                bast, bast_nyckel = i, nyckel
        ack = ack + pool[bast]
        s_ack, r_ack, _ = _spektrum(ack)
        valda.append(bast)
        spar.append(float(s_ack))
        kvar.remove(bast)
    return valda, spar


def jamn_plan(n_pool, k):
    """The baseline: k probes spread evenly over the candidate list."""
    return np.round(np.linspace(0, n_pool - 1, k)).astype(int).tolist()


def crb(pool, index, prior):
    """Cramer-Rao bound per parameter under a plan, in metres."""
    F = np.sum([pool[i] for i in index], axis=0)
    return np.sqrt(np.maximum(0.0, np.diag(np.linalg.inv(F + prior))))


def utvardera_plan(del_=Monteringsplatta, k=8, sigma=2.0e-6, prior_precision=1.0e6):
    """Chosen against evenly spread, on the same pool, the same sigma and the same prior."""
    theta = del_.nominell_theta
    punkter = del_.kandidatpunkter(theta)
    pool = fisherpool(del_, theta, punkter, sigma=sigma)
    P = np.eye(len(theta)) * prior_precision

    ut = {}
    for namn, index in (("vald", greedy_oed(pool, k, P)[0]),
                        ("jamn", jamn_plan(len(pool), k))):
        F = np.sum([pool[i] for i in index], axis=0)
        s, r, _ = _spektrum(F)
        c = crb(pool, index, P)
        ut[namn] = {"index": list(map(int, index)), "rang": r, "sigmin": s,
                    "crb_m": c.tolist(), "medel_crb_m": float(np.mean(c)),
                    "max_crb_m": float(np.max(c))}
    ut["n_parametrar"] = len(theta)
    ut["n_pool"] = len(pool)
    ut["k"] = k
    return ut


def las_referens(sokvag=None):
    with open(os.path.join(sokvag or DATA, "matplan.json")) as fh:
        return json.load(fh)


def _selftest():
    r = utvardera_plan(k=8)
    v, j = r["vald"], r["jamn"]
    print(f"plate, {r['n_parametrar']} declared dimensions, {r['n_pool']} candidate probes, k={r['k']}")
    print(f"  chosen      rank {v['rang']}/{r['n_parametrar']}  sigma_min {v['sigmin']:.4e}  "
          f"mean CRB {1e6*v['medel_crb_m']:.3f} um  worst {1e6*v['max_crb_m']:.1f} um")
    print(f"  even spread rank {j['rang']}/{r['n_parametrar']}  sigma_min {j['sigmin']:.4e}  "
          f"mean CRB {1e6*j['medel_crb_m']:.3f} um  worst {1e6*j['max_crb_m']:.1f} um")
    r16 = utvardera_plan(k=16)
    print(f"  even spread at k=16 is still rank {r16['jamn']['rang']}/{r['n_parametrar']}, "
          f"mean CRB {1e6*r16['jamn']['medel_crb_m']:.3f} um")

    fel = []
    if v["rang"] != r["n_parametrar"]:
        fel.append("the chosen plan must reach full rank")
    if j["rang"] >= r["n_parametrar"]:
        fel.append("the even plan is supposed to be rank-deficient; the comparison is meaningless")
    if j["sigmin"] != 0.0:
        fel.append("a rank-deficient Fisher must have sigma_min exactly zero")
    if not (v["medel_crb_m"] < j["medel_crb_m"] / 10.0):
        fel.append("the chosen plan must beat the even one by more than a factor ten")
    if r16["jamn"]["rang"] >= r["n_parametrar"]:
        fel.append("doubling the even plan is supposed not to fix the rank")
    try:
        _spektrum(np.array([[1.0, 0.0], [0.0, np.nan]]))
        fel.append("a non-finite Fisher must raise")
    except ValueError:
        pass
    if fel:
        raise SystemExit("FAIL: " + "; ".join(fel))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
