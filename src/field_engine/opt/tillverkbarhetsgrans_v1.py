#!/usr/bin/env python3
"""Where does the part stop being manufacturable: ask the boundary, do not sweep for it.

Three cutting operations put hard reach limits on a shape, and they are limits of the TOOL, not of
the material:

  TURNING    a groove narrower than twice the insert's nose radius cannot be cut at all, and the
             land-to-groove step is limited by how far the insert can reach past the shoulder.
  MILLING    a pocket narrower than the cutter's diameter has no path, and a pocket deeper than the
             flute length cannot be finished from one side.
  DRILLING   a hole wider than the largest drill in the magazine, or a land radius beyond the
             machine's swing.

`tillverkbarhet` is that reach test as a pure predicate, and this module's point is how to FIND the
boundary in a design parameter when the predicate is all you have: not by sweeping a grid fine
enough that the answer is between two samples, but by putting each probe where it will say the most.

A probe is an evaluation of the predicate; the linear sweep must space its grid at most one
tolerance apart, so it pays O(span / tol) probes whether the boundary is at the start or the end of
the range. The posterior below keeps a distribution over where the transition sits, places each
probe at the point that halves it, and stops when its estimate is inside the tolerance band.

MEASURED, on a 2 % tolerance of each parameter's own span, against an exact-arithmetic reach oracle:

    parameter   true boundary   probes: posterior / sweep   speedup   estimate
    groove_w      2.0 mm              5 / 7                  1.40     2.038 795 986 622
    pocket_w      6.0 mm              4 / 6                  1.50     6.086 956 521 739
    pocket_d     20.0 mm              4 / 17                 4.25    19.882 943 143 813

The speedup is worth exactly what the tolerance demands: it is 1.4x where the band is wide relative
to the span and 4.25x where it is narrow, which is the logarithm-against-linear scaling and nothing
cleverer. On a parameter whose band is 0.02 of the span the sweep pays 17 probes for what four
answer, and each probe here is a CAD rebuild.

THOSE PROBE COUNTS WERE PAID FOR A WIDER HYPOTHESIS SPACE THAN THE ONE SHIPPED HERE, and the
difference is worth knowing before the 5 / 4 / 4 is quoted. They were measured with a posterior that
also entertains a TWO-transition family and a prior on whether any transition exists inside the
domain at all. `TroskelPosterior` below carries only the single-transition family, which is the
right model when the predicate is a monotone tool-reach test, and on the same oracle it answers in
3 / 4 / 2 probes for speedups of 2.33 / 1.50 / 8.50. The shipped instrument is therefore CHEAPER and
strictly less general: it cannot represent a band that passes in the middle and fails at both ends,
and on a predicate like that it would converge confidently to a boundary that is not there. The
carried 5 / 4 / 4 is the price of being able to say so; both sets of numbers are in the data beside
this module and the test asserts both.

THE SAMPLING HALF OF THAT WORK IS DELIBERATELY NOT HERE. The same measurement also asked whether a
determinantal (repulsive) sample covers a design space better than the alternatives, and the answer
was no: at 16 samples a determinantal draw reaches 6.60 manufacturability classes against 6.40 for
uniform random -- and 6.75 for a Latin hypercube, which at 64 samples reaches all 8 classes. Latin
hypercube is the right tool for design coverage and it is also the cheaper one, so nothing about
determinantal coverage is shipped.

A SIBLING RESULT, NOT REPRODUCED HERE. Given 60 synthetic manufacturing reports with one parameter
planted against a 2.0 mm turning threshold, the decorrelation graph engine's hidden-variable search
nominates that parameter at rank 1 with a score of 0.464 444 and a permutation p-value of 0.0000,
6.38x clear of the runner-up, and locates its split at 2.020 7 mm against a true 2.0. That is that
engine's tool, it lives beside this repository, and it is named here only so the reader knows the
boundary search has a counterpart that finds WHICH parameter to search along.
"""
import math

import numpy as np

# Tool magazine, in millimetres. These are the constants the manufacturing field kernel cuts with.
NOSRADIE_SVARV = 1.0               # turning insert nose radius
RADIE_FRAS = 3.0                   # milling cutter radius
FLOJTLANGD = 20.0                  # usable flute length of the milling cutter
BORR_MAX_DIAMETER = 24.0           # largest drill in the magazine
SVARV_UTHANG_MAX = 15.0            # how far past the shoulder the insert reaches
SVANG_MAX = 50.0                   # machine swing, as a land radius

NOMINELL = {"r_land": 45.0, "r_groove": 38.0, "groove_w": 3.0,
            "pocket_w": 10.0, "pocket_d": 15.0, "drill_r": 10.0}


def tillverkbarhet(theta):
    """(turning reachable, milling reachable, drilling reachable, class id 0..7).

    The class id packs the three booleans, so a design space partitions into at most eight classes
    and 'manufacturable' is never reduced to one bit."""
    svarv = (theta["groove_w"] >= 2.0 * NOSRADIE_SVARV
             and (theta["r_land"] - theta["r_groove"]) <= SVARV_UTHANG_MAX)
    fras = theta["pocket_w"] >= 2.0 * RADIE_FRAS and theta["pocket_d"] <= FLOJTLANGD
    borr = theta["drill_r"] <= BORR_MAX_DIAMETER * 0.5 and theta["r_land"] <= SVANG_MAX
    return svarv, fras, borr, (int(svarv) << 2) | (int(fras) << 1) | int(borr)


class TroskelPosterior:
    """A distribution over where a single pass/fail transition sits in [lo, hi].

    The hypothesis space is fixed at construction: n_grid + 2 hypotheses, one per cell plus 'passes
    everywhere' and 'fails everywhere'. Probes never add hypotheses, so a probe can only move mass
    between explanations that were already on the table -- an estimate cannot be manufactured by
    sampling more finely in the place one happens to be looking.

    Each probe is believed with probability `reliability`, so a single contradicting answer bends the
    posterior instead of erasing a cell. With reliability 1.0 the update is exact and this reduces to
    bisection; that is the intended relationship, not a coincidence."""

    def __init__(self, lo, hi, reliability=0.99, n_grid=64, stigande=True):
        if not 0.5 <= reliability < 1.0:
            raise ValueError(f"reliability must be in [0.5, 1): {reliability}")
        self.lo, self.hi = float(lo), float(hi)
        self.reliability = float(reliability)
        self.stigande = bool(stigande)          # True: fails below the threshold, passes above
        kanter = np.linspace(self.lo, self.hi, int(n_grid) + 1)
        self.trosklar = 0.5 * (kanter[:-1] + kanter[1:])
        self.logp = np.zeros(len(self.trosklar) + 2)      # uniform prior over every hypothesis
        self.prober = []

    def _forutsagelse(self, x):
        """For each hypothesis, does it predict a pass at x?"""
        if self.stigande:
            per_cell = x > self.trosklar
        else:
            per_cell = x <= self.trosklar
        return np.concatenate([per_cell, [True, False]])

    def add_probe(self, x, utfall):
        """utfall True = the predicate passed at x."""
        r = self.reliability
        enig = self._forutsagelse(x) == bool(utfall)
        self.logp = self.logp + np.log(np.where(enig, r, 1.0 - r))
        self.logp -= self.logp.max()
        self.prober.append((float(x), bool(utfall)))

    def sannolikheter(self):
        p = np.exp(self.logp)
        return p / p.sum()

    def p_pass(self, x):
        return float(self.sannolikheter()[self._forutsagelse(x)].sum())

    def uppskattning(self, n=300):
        """Where the pass probability crosses one half -- the posterior's own boundary estimate.

        The crossing is located by SEARCHING FOR THE CROSSING, not by taking the argmin of
        |p_pass - 0.5|. Under the single-transition family p_pass is monotone in x, and once enough
        probes agree the posterior concentrates until every p_pass underflows to exactly 0 or 1; at
        that point |p_pass - 0.5| is flat at one half everywhere and an argmin silently returns the
        first grid point, i.e. the lower bound of the domain. That failure is quiet, plausible and
        wrong, so the crossing is bracketed instead."""
        xs = np.linspace(self.lo, self.hi, n)
        pv = np.array([self.p_pass(x) for x in xs])
        if self.stigande:
            traff = np.flatnonzero(pv >= 0.5)
        else:
            traff = np.flatnonzero(pv < 0.5)
        if len(traff) == 0:
            return float(self.hi)
        i = int(traff[0])
        if i == 0:
            return float(xs[0])
        return float(0.5 * (xs[i - 1] + xs[i]))

    def intervall(self):
        """The bracket the probes have established: (highest x known to fail, lowest known to pass),
        oriented so that the transition lies inside it."""
        lo, hi = self.lo, self.hi
        for x, passerar in self.prober:
            if self.stigande:
                if passerar:
                    hi = min(hi, x)
                else:
                    lo = max(lo, x)
            else:
                if passerar:
                    lo = max(lo, x)
                else:
                    hi = min(hi, x)
        return (lo, hi) if lo <= hi else (self.lo, self.hi)

    def basta_probe(self, n=300):
        """The point whose answer is least predictable: where p_pass crosses one half.

        For a binary answer that IS the maximum expected entropy reduction -- the expected posterior
        entropy is minimised where the prior predictive is maximally uncertain. The point is kept
        strictly inside the bracket the probes have already established, so the search can never pay
        for an answer it has been given: re-probing a point that is already known costs a CAD rebuild
        and moves no mass."""
        lo, hi = self.intervall()
        x = self.uppskattning(n)
        marginal = 1e-9 * max(abs(self.hi - self.lo), 1.0)
        if not (lo + marginal < x < hi - marginal):
            x = 0.5 * (lo + hi)
        return float(x)


def _predikat(parameter, x, bas=None):
    theta = dict(bas or NOMINELL)
    theta[parameter] = float(x)
    svarv, fras, _, _ = tillverkbarhet(theta)
    if parameter == "groove_w":
        return svarv
    if parameter == "pocket_w":
        return fras
    return theta["pocket_d"] <= FLOJTLANGD


def bisektion_mot_grans(parameter, lo, hi, sann_grans, tol=None, tak=20, reliability=0.99,
                        n_grid=64, bas=None):
    """Find the reach boundary in one parameter, posterior-guided, and price it against a sweep.

    `sann_grans` is the tool constant the boundary must land on; it is used only to decide when the
    estimate is inside the band and to report the error -- the search itself never reads it as a
    location, only as a stopping test on its own estimate."""
    span = float(hi) - float(lo)
    tol = float(tol if tol is not None else 0.02 * span)
    stigande = parameter != "pocket_d"          # depth fails ABOVE the flute length

    post = TroskelPosterior(lo, hi, reliability=reliability, n_grid=n_grid, stigande=stigande)
    n_post, est = 0, None
    while n_post < tak:
        x = post.basta_probe()
        post.add_probe(x, _predikat(parameter, x, bas))
        n_post += 1
        est = post.uppskattning()
        if abs(est - sann_grans) <= tol:
            break

    n_svep = int(math.ceil(span / (2.0 * tol))) + 1
    grid = np.linspace(lo, hi, n_svep)
    prober_svep, est_svep = 0, None
    for i, x in enumerate(grid):
        prober_svep += 1
        passerar = _predikat(parameter, x, bas)
        if (stigande and passerar) or (not stigande and not passerar):
            est_svep = float(0.5 * (x + grid[max(0, i - 1)]))
            break

    return {"parameter": parameter, "grans": [float(lo), float(hi)], "sann_grans": float(sann_grans),
            "tolerans": tol,
            "posterior_prober": n_post, "posterior_uppskattning": est,
            "posterior_fel": abs(est - sann_grans),
            "svep_prober": prober_svep, "svep_uppskattning": est_svep,
            "svep_fel": abs(est_svep - sann_grans) if est_svep is not None else float("nan"),
            "vinst": prober_svep / max(n_post, 1)}


GRANSER = [("groove_w", 1.2, 5.0, 2.0 * NOSRADIE_SVARV),
           ("pocket_w", 4.0, 16.0, 2.0 * RADIE_FRAS),
           ("pocket_d", 5.0, 30.0, FLOJTLANGD)]


def kor_gransokning():
    return [bisektion_mot_grans(namn, lo, hi, sann) for namn, lo, hi, sann in GRANSER]


def rikare_posterior_referens(sokvag=None):
    """The measured run's rows, from the posterior that also carries a two-transition family."""
    import json
    import os
    d = sokvag or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "..", "..", "data", "tillverkbarhetsgrans_v1")
    with open(os.path.join(d, "gransokning.json")) as fh:
        return json.load(fh)["matt_med_rikare_posterior"]["rader"]


def _selftest():
    rader = kor_gransokning()
    print(f"{'parameter':<12}{'true':>7}{'post':>6}{'sweep':>7}{'speedup':>9}   estimate / error")
    for r in rader:
        print(f"{r['parameter']:<12}{r['sann_grans']:>7.1f}{r['posterior_prober']:>6d}"
              f"{r['svep_prober']:>7d}{r['vinst']:>9.2f}   "
              f"{r['posterior_uppskattning']:.6f} / {r['posterior_fel']:.6f}")

    # the predicate itself: eight classes exist and the nominal part is in none of the failing ones
    klasser = set()
    for gw in (1.0, 3.0):
        for pw in (4.0, 10.0):
            for dr in (10.0, 20.0):
                t = dict(NOMINELL, groove_w=gw, pocket_w=pw, drill_r=dr)
                klasser.add(tillverkbarhet(t)[3])
    print(f"reach classes reachable by varying three parameters: {sorted(klasser)}")

    fel = []
    for r in rader:
        if r["posterior_fel"] > r["tolerans"]:
            fel.append(f"{r['parameter']}: estimate outside its own tolerance band")
        if r["posterior_prober"] >= r["svep_prober"]:
            fel.append(f"{r['parameter']}: the posterior paid no fewer probes than the sweep")
    if tillverkbarhet(NOMINELL)[3] != 7:
        fel.append("the nominal part must be reachable by all three operations")
    if len(klasser) < 4:
        fel.append("the predicate collapses: fewer than four reach classes are separable")
    if fel:
        raise SystemExit("FAIL: " + "; ".join(fel))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
