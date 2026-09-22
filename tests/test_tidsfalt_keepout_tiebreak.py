"""The incremental owner channel's tie-break: exact int64, lowest particle id, two back ends.

The previous float KDTree left an exact distance tie to the tree's internal order, so the owner
channel was not a function of the particles alone and two implementations could disagree. The rule
now compares squared distances as exact int64 in fixed point (1 quantum = 1 nm) and breaks a tie by
the lowest particle id. This file runs the five audited scenes (a growing pile, three bodies on sinusoids,
a body that jumps 3 cells/tick, a body removed at tick 4, a shrinking pile) and locks: 0 divergent
ticks with the removal pass, exactly 91 rule-change cells against the float KDTree (jump_3cells 66,
disappear_at_4 25, the rest 0), every one an exact tie, broken identically by the tree-and-ballot
engine and by an independent chunked int64 brute force.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tidsfalt_keepout_v1 as TK                                                    # noqa: E402
import test_tidsfalt_keepout_inkrementell as I                                      # noqa: E402


def _scener():
    return [I.Hog(), I.TreKroppar(ticks=200),
            I.EdgeScene("jump_3cells", jump_cells=3),
            I.EdgeScene("disappear_at_4", jump_cells=3, disappear_at=4),
            I.ShrinkScene()]


def _gammal_agare_full(punkter, agare, cellcentra, radie):
    """The previous rebuild path: one KDTree k=1, float `<=`, ties left to scipy."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(punkter, dtype=np.float64))
    d, idx = tree.query(cellcentra, k=1, workers=1)
    kanal = np.full(len(cellcentra), -1, dtype=np.int32)
    traff = d <= float(radie)
    kanal[traff] = np.asarray(agare, dtype=np.int32)[idx[traff]]
    return kanal


def _tie_character(P, cellcentra, radie, cell):
    """How many in-range particles share the minimal int64 squared distance on one cell."""
    Pq = TK.kvantisera(P)
    Cq = TK.kvantisera(cellcentra)
    r2q = np.int64(TK.r2_kvant(radie))
    dq = Pq - Cq[np.int64(cell)]
    q2 = dq[:, 0] * dq[:, 0] + dq[:, 1] * dq[:, 1] + dq[:, 2] * dq[:, 2]
    mq = q2 <= r2q
    if not bool(np.any(mq)):
        return 0
    return int(np.count_nonzero(q2[mq] == q2[mq].min()))


@pytest.fixture(scope="module")
def scener():
    return _scener()


def test_noll_avvikande_tick_med_borttagning(scener):
    """The incremental dom and owner channel equal the full rebuild on every tick of the five
    scenes, under the exact tie-break rule: 0 of 40 / 200 / 8 / 8 / 6."""
    for sc in scener:
        rows = I._run(sc)
        assert [r["tick"] for r in rows if r["dom_sha_a"] != r["dom_sha_del"]] == []
        assert all(r["owner_del_equal"] for r in rows)


def test_brytregeln_ror_exakt_91_tie_celler(scener):
    """Against the float KDTree the rule changes exactly 91 cells: jump_3cells 66, disappear_at_4
    25, and none on the pile or the three bodies. Every changed cell is an exact tie (at least two
    particles share the minimal int64 squared distance), never a range-membership change."""
    vantade = {"hog": 0, "tre_kroppar": 0, "jump_3cells": 66, "disappear_at_4": 25, "shrink": 0}
    total = 0
    for sc in scener:
        n = 0
        for f in range(sc.ticks):
            P, O, _ = sc.particle_state(f)
            gammal = _gammal_agare_full(P, O, sc.centers, sc.radius)
            ny = TK.agare_full(P, O, sc.centers, sc.radius)
            for c in np.nonzero(gammal != ny)[0]:
                assert _tie_character(P, sc.centers, sc.radius, int(c)) >= 2
                n += 1
        assert n == vantade[sc.name], (sc.name, n)
        total += n
    assert total == 91


def test_tva_implementationer_enas_pa_tie_cellerna(scener):
    """The tree-and-ballot engine (`agare_full` / `agare_delvis`) and the independent chunked int64
    brute force return the same owner on every one of the 91 tie cells."""
    for sc in scener:
        for f in range(sc.ticks):
            P, O, _ = sc.particle_state(f)
            gammal = _gammal_agare_full(P, O, sc.centers, sc.radius)
            ny = TK.agare_full(P, O, sc.centers, sc.radius)
            cells = np.nonzero(gammal != ny)[0]
            if len(cells) == 0:
                continue
            ref = TK.agare_brute_kvant(P, O, sc.centers, sc.radius, cells)
            assert np.array_equal(ny[cells], ref)
            assert np.array_equal(TK.agare_delvis(P, O, sc.centers, sc.radius, cells), ref)


def test_tva_implementationer_enas_pa_hela_kanalen(scener):
    """Beyond the tie cells, the two implementations agree on the whole owner channel: every tick
    of the two tie scenes and a sampled set of ticks of the three-body scene."""
    for sc in scener:
        ticks = range(sc.ticks)
        if sc.name == "tre_kroppar":
            ticks = [0, 1, 50, 100, 150, 199]
        for f in ticks:
            P, O, _ = sc.particle_state(f)
            idx = np.arange(len(sc.centers), dtype=np.int64)
            assert np.array_equal(TK.agare_full(P, O, sc.centers, sc.radius),
                                  TK.agare_brute_kvant(P, O, sc.centers, sc.radius, idx))


def test_symmetrisk_lika_bryts_till_lagsta_id():
    """A hand-built exact tie: two particles at equal distance, different owners. The lowest
    particle id wins, in both implementations, and the result follows the id when the order is
    swapped -- so the owner is a function of the particle arrays, not of any tree's structure."""
    centers = np.array([[0.0, 0.0, 0.0]])
    P = np.array([[0.0005, 0.0, 0.0], [-0.0005, 0.0, 0.0]])
    O = np.array([7, 3], dtype=np.int32)
    assert int(TK.agare_delvis(P, O, centers, 0.001, [0])[0]) == 7
    assert int(TK.agare_full(P, O, centers, 0.001)[0]) == 7
    assert int(TK.agare_brute_kvant(P, O, centers, 0.001, [0])[0]) == 7
    P2, O2 = P[::-1].copy(), O[::-1].copy()
    assert int(TK.agare_delvis(P2, O2, centers, 0.001, [0])[0]) == 3
    assert int(TK.agare_full(P2, O2, centers, 0.001)[0]) == 3
    assert int(TK.agare_brute_kvant(P2, O2, centers, 0.001, [0])[0]) == 3
