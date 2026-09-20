#!/usr/bin/env python3
"""Spend one resolution budget on several goals at once, without letting the loudest goal win.

Given per-block sensitivities of several goals to the stored resolution, the allocation that
minimises total squared goal error under a linear budget has a closed form: resolution proportional
to the two-thirds power of the sensitivity, with a water level set by bisection on the budget.
`vattenfyllning` is that law, and it is not the hard part.

THE HARD PART IS WHICH SENSITIVITY GOES IN. Four goals on one cell disagree about units by two
orders of magnitude -- a clearance in metres, a compliance in joules, a contact margin, a volume --
so a raw max or sum over the goals allocates the whole budget to whichever goal happens to be
measured in the smaller unit. `pareto_ref_vikter` divides each goal's sensitivity by a REFERENCE
value of that same goal before combining, which makes the combination dimensionless and the
allocation scale-free.

MEASURED, four goals over 50 blocks, eight policies at five budgets, 40 of 40 field hashes equal
across two runs:

    reference-normalised Pareto, 20 % budget    2 640 B    all four goals inside 5 % (max 2.29 %)
    three asynchronous per-goal fields          3 988 B    -33.8 %
    three synchronous per-goal fields           7 828 B    -66.3 %

THOSE BYTES ARE A COST FORMULA, not files. They are four bytes per stored sample plus a header,
computed from the allocation; nothing was written to a disk and measured.

AND THE REFERENCE IS NOT FREE. The normalisation needs a reference value per goal, and a fully
resolved reference field costs 13 284 B on its own. Charged symmetrically, the 2 640 B becomes
15 520 B, which is 120.5 % of the 12 880 B full field -- the allocation costs MORE than storing
everything. So "20 % of the budget" holds only when the reference comes from somewhere that is
already being paid for, an exact CAD model or a previously computed field, and the moment it has to
be produced for this purpose the saving is gone. That is the condition under which the numbers
above are true, and it is not a footnote.

On a three-dimensional cell the same policy reaches all goals within 4.173 % at 617 980 B against
753 652 B for per-goal fields, while a uniform grid needs 5 119 452 B at the same budget and still
reads 19.76 % on one goal. Uniform is not a weaker allocation here; on the contact count it never
converges at all, reading 8.33 % even when given 100 % of the budget.

AN OPTIONAL SIGNAL THAT NEEDS NO REFERENCE AT ALL. Everything above needs a reference value per
goal. `oenighetsvikt` does not: where two independent operators answering the same question agree,
resolution buys nothing; where they disagree, one of them is wrong. Measured on the assembly, one
iteration, no reference run, at the same 20 % budget:

    contact margin       0.093 %      inside the gate
    routing clearance    0.017 %      inside the gate
    compliance           5.421 %      OUTSIDE it
    stored             2 395 759 B    69 % of the reference-normalised allocation's 3 481 935 B

So it is cheaper and it fails, and it fails for a reason that is structural rather than incidental:
the disagreement is measured between a field reading and an exact reading of the same GEOMETRY, and
compliance is not a property of the geometry at a point. There is no second operator to disagree
about the load path, so the signal is blind to it and allocates nothing there. It works where two
operators exist to be compared -- contact and routing, where it is two and three orders of magnitude
inside the gate at two thirds of the cost -- and it is exposed here as an optional weight for
exactly those, not as a replacement.

One more caution on that measurement: the contact-patch gaps it is credited with preserving were
ALREADY exact at a 10 % budget, on all five patches, before any disagreement weighting was applied.
The disagreement sat in other blocks. The 0.093 % is therefore not evidence that the signal found
the contact-critical blocks; it is evidence that it did not break them.
"""
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "data", "kropps_id_falt_v1")

M_GOLV = 1.0
M_TAK = 17 ** 3                    # the finest level in the block menu


def vattenfyllning(c, budget, mult=None, m_golv=M_GOLV, m_tak=M_TAK, iterationer=120):
    """Resolution per block under a linear budget: M_i proportional to (c_i^2 / mult_i)^(1/3).

    `c` is a per-block sensitivity, `mult` the per-block cost weight (how many channels that block
    pays for), and `budget` a fraction of the fully resolved cost. The water level is found by
    bisection in log space, which is monotone in the budget and needs no derivative.

    A block whose sensitivity is zero still gets the floor: dropping it entirely would make the
    field non-addressable, and one sample is the cheapest way to stay addressable."""
    c = np.asarray(c, dtype=np.float64)
    mult = np.ones_like(c) if mult is None else np.asarray(mult, dtype=np.float64)
    if np.any(mult <= 0):
        raise ValueError("every block's cost weight must be positive")
    mal = float(budget) * float(m_tak) * float(mult.sum())

    def M_av(lam):
        return np.clip((2.0 * lam * c ** 2 / mult) ** (1.0 / 3.0), m_golv, m_tak)

    lo, hi = 1e-100, 1e100
    for _ in range(iterationer):
        mid = np.sqrt(lo * hi)
        if float(np.dot(mult, M_av(mid))) > mal:
            hi = mid
        else:
            lo = mid
    return M_av(lo), float(lo)


def pareto_ref_vikter(c_per_mal, referenser):
    """Combine several goals' per-block sensitivities into one scale-free weight.

    Each goal is divided by a reference VALUE of that goal, so the comparison is in units of "share
    of the goal" rather than in the goal's own units, and the combination is the worst case over
    goals: a block matters as much as the goal it hurts most."""
    c = np.asarray(c_per_mal, dtype=np.float64)
    r = np.abs(np.asarray(referenser, dtype=np.float64)).reshape(-1, 1)
    if np.any(r == 0.0):
        raise ValueError("a goal with a zero reference cannot be normalised")
    return np.max(np.abs(c) / r, axis=0)


def rasumma_vikter(c_per_mal):
    """The goal-blind baseline the normalisation is measured against: the raw maximum over goals,
    in whatever units each goal happens to use."""
    return np.max(np.abs(np.asarray(c_per_mal, dtype=np.float64)), axis=0)


def oenighetsvikt(gap_a, gap_b):
    """Per-block weight from the DISAGREEMENT between two operators that answer the same question.

    Where two independent readings of the same quantity agree, neither is likely to be the one that
    is wrong, and resolution spent there buys nothing; where they disagree, at least one of them is
    wrong and that is where the budget is worth spending. Nothing about the truth is needed, which
    is the point: this is a signal available BEFORE any reference run exists.

    It is a signal with a known blind spot -- see `oenighetsutfall` for what it costs."""
    a = np.asarray(gap_a, dtype=np.float64)
    b = np.asarray(gap_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("the two operators must answer for the same blocks")
    return np.abs(a - b)


def oenighetsutfall(sokvag=None):
    """The measured disagreement allocation, with the gate it fails and the reason."""
    with open(os.path.join(sokvag or DATA, "oenighet.json")) as fh:
        d = json.load(fh)
    fel = d["kandidat"]["fel_procent"]
    namn = ("kontaktmarginal", "ruttclearance", "compliance")
    return {"budget": d["budget"], "bytes": d["kandidat"]["bytes"],
            "fel_procent": dict(zip(namn, fel)),
            "innanfor_grinden": {n: f <= d["grind_procent"] for n, f in zip(namn, fel)},
            "grind_procent": d["grind_procent"],
            "andel_av_pareto_ref": d["kandidat"]["bytes"] / d["jamforelser"]["pareto_ref_B0.2"]["bytes"],
            "jamforelser": d["jamforelser"],
            "blind_for": "compliance, which has no second operator to disagree about",
            "patchgap_redan_exakta_vid_10_procent": d["patchgap_vid_10_procent"]["max_gapdiff_m"] == 0.0,
            "byteidentiska_korningar": d["tva_korningar"]["container_sha_lika"]}


def bytes_for_allokering(M, mult=None, huvud=80, bytes_per_prov=4):
    """The cost formula the byte counts in this module are: a header plus four bytes per sample."""
    M = np.asarray(M, dtype=np.float64)
    mult = np.ones_like(M) if mult is None else np.asarray(mult, dtype=np.float64)
    return int(huvud + np.round(bytes_per_prov * float(np.dot(mult, M))))


def malfel(M, c_per_mal, referenser):
    """Relative error per goal under an allocation, in the same 'share of the goal' units the
    weights are built in: sum over blocks of (c / M)^2, against the goal's reference."""
    M = np.asarray(M, dtype=np.float64)
    c = np.asarray(c_per_mal, dtype=np.float64)
    r = np.abs(np.asarray(referenser, dtype=np.float64))
    return np.sqrt(np.sum((c / M) ** 2, axis=1)) / r


def las_allokering(sokvag=None):
    with open(os.path.join(sokvag or DATA, "allokering.json")) as fh:
        return json.load(fh)


def syntetiskt_oenighetsproblem(n_block=60, fro=20260920):
    """Two operators reading the same blocks, plus a goal neither of them can see.

    Operator A and B agree everywhere except in a known set of blocks. A first goal's error lives in
    exactly those blocks -- the signal should find it. A second goal's error lives in blocks where
    the two agree perfectly -- the signal is blind to it, by construction, which is the whole point
    of shipping this as an option rather than a default."""
    rng = np.random.default_rng(fro)
    gap_a = rng.normal(0.0, 0.002, n_block)
    gap_b = gap_a.copy()
    oeniga = np.arange(5, 20)
    gap_b[oeniga] += rng.uniform(0.01, 0.05, len(oeniga))
    c_synligt = np.full(n_block, 0.01)
    c_synligt[oeniga] = 1.0
    c_osynligt = np.full(n_block, 0.01)
    c_osynligt[np.arange(40, 55)] = 1.0
    return gap_a, gap_b, oeniga, c_synligt, c_osynligt


def syntetiskt_flermalsproblem(n_block=50, n_mal=4, fro=20260920):
    """Four goals whose sensitivities differ by two orders of magnitude in scale, which is the
    situation the normalisation exists for: goal 1 is in units about 100x goal 3."""
    rng = np.random.default_rng(fro)
    skalor = np.array([57.0, 12.0, 0.5, 3.0])[:n_mal]
    c = np.abs(rng.gamma(1.4, 1.0, size=(n_mal, n_block))) * skalor.reshape(-1, 1)
    referenser = c.sum(axis=1) * 0.5
    return c, referenser


def _selftest():
    c, ref = syntetiskt_flermalsproblem()
    budget = 0.20

    M_ref, _ = vattenfyllning(pareto_ref_vikter(c, ref), budget)
    M_rasum, _ = vattenfyllning(rasumma_vikter(c), budget)
    b_ref, b_rasum = bytes_for_allokering(M_ref), bytes_for_allokering(M_rasum)
    f_ref, f_rasum = malfel(M_ref, c, ref), malfel(M_rasum, c, ref)
    print(f"budget {budget:.0%}: reference-normalised {b_ref} B, worst goal {f_ref.max():.4%}")
    print(f"            raw scale  {b_rasum} B, worst goal {f_rasum.max():.4%}")
    print(f"  per goal normalised {np.array2string(f_ref, precision=5)}")
    print(f"  per goal raw        {np.array2string(f_rasum, precision=5)}")

    # the budget is actually spent, and a larger budget is never a worse allocation
    forra = None
    for b in (0.05, 0.1, 0.2, 0.4, 1.0):
        M, _ = vattenfyllning(pareto_ref_vikter(c, ref), b)
        varsta = malfel(M, c, ref).max()
        if forra is not None and varsta > forra + 1e-12:
            raise SystemExit("FAIL: a larger budget gave a worse worst-goal error")
        forra = varsta

    d = las_allokering()
    tva = d["flermal_2d"]
    tre = d["flermal_3d"]
    acc = tva["referenskostnad"]
    print(f"carried 2-D: {tva['pareto_ref_bytes']} B against {tva['per_mal_asynkront_bytes']} B "
          f"asynchronous and {tva['per_mal_synkront_bytes']} B synchronous")
    print(f"carried reference charge: {acc['reference_plus_candidate_bytes']} B = "
          f"{acc['reference_plus_candidate_percent']:.4f} % of the full field")
    print(f"carried 3-D: pareto {tre['pareto_ref']['bytes']} B, per goal "
          f"{tre['per_goal']['bytes']} B, uniform {tre['uniform']['bytes']} B")

    # the optional signal, live: it finds what two operators disagree about and nothing else
    gap_a, gap_b, oeniga, c_synligt, c_osynligt = syntetiskt_oenighetsproblem()
    w = oenighetsvikt(gap_a, gap_b)
    M_o, _ = vattenfyllning(w, budget)
    andel_oeniga = M_o[oeniga].sum() / M_o.sum()
    f_synligt = malfel(M_o, c_synligt.reshape(1, -1), np.array([1.0]))[0]
    f_osynligt = malfel(M_o, c_osynligt.reshape(1, -1), np.array([1.0]))[0]
    print(f"disagreement signal: {len(oeniga)} of {len(w)} blocks disagree and take "
          f"{andel_oeniga:.1%} of the allocation")
    print(f"  goal it can see  {f_synligt:.5f}   goal it is blind to {f_osynligt:.5f}")
    o = oenighetsutfall()
    print(f"carried: {o['bytes']} B = {o['andel_av_pareto_ref']:.1%} of the normalised allocation, "
          + ", ".join(f"{n} {v:.3f} %" for n, v in o["fel_procent"].items()))
    print(f"  inside the {o['grind_procent']:.0f} % gate: "
          + ", ".join(f"{n}={v}" for n, v in o["innanfor_grinden"].items()))

    fel = []
    if f_ref.max() >= f_rasum.max():
        fel.append("normalising by a goal reference must beat combining raw scales")
    if b_ref > bytes_for_allokering(np.full(len(M_ref), M_TAK)):
        fel.append("the allocation costs more than full resolution")
    if acc["reference_plus_candidate_percent"] <= 100.0:
        fel.append("the reference charge must be carried as measured, above 100 %")
    if tre["uniform"]["bytes"] <= tre["pareto_ref"]["bytes"]:
        fel.append("the carried uniform baseline is not larger than the allocation")
    if andel_oeniga <= len(oeniga) / len(w):
        fel.append("the disagreement signal must concentrate on the blocks that disagree")
    if f_synligt >= f_osynligt:
        fel.append("the disagreement signal must be better on the goal it can see than on the "
                   "goal it is blind to; without that gap it is not the signal it claims to be")
    if o["innanfor_grinden"]["compliance"]:
        fel.append("the carried compliance failure must be carried as measured")
    if not (o["innanfor_grinden"]["kontaktmarginal"] and o["innanfor_grinden"]["ruttclearance"]):
        fel.append("the two goals the signal does carry must be carried as measured")
    try:
        pareto_ref_vikter(c, np.array([1.0, 0.0, 1.0, 1.0]))
        fel.append("a zero goal reference must raise")
    except ValueError:
        pass
    if fel:
        raise SystemExit("FAIL: " + "; ".join(fel))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
