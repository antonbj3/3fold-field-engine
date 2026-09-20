"""Allocating resolution by what two operators disagree about, and the goal that leaves behind.

The signal needs no reference run, which is what makes it worth having, and it is blind to any goal
that is not a property of the geometry at a point, which is what keeps it optional. Both halves run
live on a synthetic pair of operators; the measured assembly numbers, including the gate this signal
fails, are carried.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine"))

import pareto_allokering_v1 as PA                                                   # noqa: E402


@pytest.fixture(scope="module")
def problem():
    return PA.syntetiskt_oenighetsproblem()


@pytest.fixture(scope="module")
def utfall():
    return PA.oenighetsutfall()


# --- the signal itself, live ----------------------------------------------------------------------
def test_vikten_ar_ren_oenighet(problem):
    """Nothing about the truth enters: the weight is the gap between two readings, which is what
    makes it available before any reference exists."""
    gap_a, gap_b, oeniga, _, _ = problem
    w = PA.oenighetsvikt(gap_a, gap_b)
    assert np.all(w >= 0.0)
    assert np.all(w[oeniga] > 0.009)
    ovriga = np.setdiff1d(np.arange(len(w)), oeniga)
    assert np.all(w[ovriga] == 0.0)
    assert np.allclose(PA.oenighetsvikt(gap_a, gap_a), 0.0)


def test_olika_langa_operatorer_vagras():
    with pytest.raises(ValueError):
        PA.oenighetsvikt(np.zeros(5), np.zeros(6))


def test_budgeten_hamnar_dar_de_ar_oeniga(problem):
    gap_a, gap_b, oeniga, _, _ = problem
    M, _ = PA.vattenfyllning(PA.oenighetsvikt(gap_a, gap_b), 0.2)
    andel = M[oeniga].sum() / M.sum()
    assert andel > 0.9
    assert andel > len(oeniga) / len(M)


def test_signalen_ar_blind_for_ett_mal_utan_andra_operator(problem):
    """The mechanism behind the compliance failure, reproduced: a goal whose error lives where the
    two operators agree perfectly gets no budget, and comes out far worse."""
    gap_a, gap_b, _, c_synligt, c_osynligt = problem
    M, _ = PA.vattenfyllning(PA.oenighetsvikt(gap_a, gap_b), 0.2)
    ett = np.array([1.0])
    f_synligt = PA.malfel(M, c_synligt.reshape(1, -1), ett)[0]
    f_osynligt = PA.malfel(M, c_osynligt.reshape(1, -1), ett)[0]
    assert f_synligt < f_osynligt / 10.0


def test_normaliserad_pareto_ser_bada_malen(problem):
    """The reference-normalised weight, which does need a reference, does not have that blind spot:
    given both goals it keeps them within a factor of a few of each other."""
    _, _, _, c_synligt, c_osynligt = problem
    c = np.stack([c_synligt, c_osynligt])
    ref = np.array([1.0, 1.0])
    M, _ = PA.vattenfyllning(PA.pareto_ref_vikter(c, ref), 0.2)
    f = PA.malfel(M, c, ref)
    assert f.max() / f.min() < 3.0


# --- the measured allocation, carried -------------------------------------------------------------
def test_tva_mal_innanfor_grinden_ett_utanfor(utfall):
    assert utfall["fel_procent"]["kontaktmarginal"] == pytest.approx(0.09314434895, rel=1e-9)
    assert utfall["fel_procent"]["ruttclearance"] == pytest.approx(0.01749297509, rel=1e-9)
    assert utfall["fel_procent"]["compliance"] == pytest.approx(5.42094944403, rel=1e-9)
    assert utfall["innanfor_grinden"] == {"kontaktmarginal": True, "ruttclearance": True,
                                          "compliance": False}


def test_signalen_ar_billigare_an_den_normaliserade(utfall):
    assert utfall["bytes"] == 2395759
    assert utfall["jamforelser"]["pareto_ref_B0.2"]["bytes"] == 3481935
    assert utfall["andel_av_pareto_ref"] == pytest.approx(0.688, abs=1e-3)


def test_den_normaliserade_klarar_compliance_dar_signalen_inte_gor_det(utfall):
    """The comparison that decides it: at 1.4x the bytes the normalised allocation reads 1.96 % on
    the goal where the disagreement signal reads 5.42 %."""
    p = utfall["jamforelser"]["pareto_ref_B0.2"]["fel_procent"]
    u = utfall["jamforelser"]["uniform_n9"]["fel_procent"]
    assert p[2] == pytest.approx(1.9576234817, rel=1e-9)
    assert p[2] < utfall["grind_procent"]
    assert u[2] == pytest.approx(4.0813479795, rel=1e-9)
    assert utfall["fel_procent"]["compliance"] > p[2]


def test_kontaktvinsten_ar_inte_signalens_fortjanst(utfall):
    """The contact-patch gaps were already exact at a 10 % budget, before any disagreement
    weighting. The 0.093 % says the signal did not break them, not that it found them."""
    assert utfall["patchgap_redan_exakta_vid_10_procent"] is True


def test_korningarna_ar_byteidentiska(utfall):
    assert utfall["byteidentiska_korningar"] is True


def test_blindfacket_star_i_modulen(utfall):
    assert "compliance" in utfall["blind_for"]
    assert "second operator" in utfall["blind_for"]
