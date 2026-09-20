"""Shape derivatives at the leaves, the expression tree they must agree with, and the reach boundary.

Everything here runs live except the contact-composed gradient, which needs a solver that is not in
this repository and travels as data. Whole file is about three seconds.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine", "opt"))

import formderivata_v1 as FD                                                        # noqa: E402
import tillverkbarhetsgrans_v1 as TG                                                # noqa: E402


# --- the two declarations of the same finger ------------------------------------------------------
@pytest.fixture(scope="module")
def paritet():
    return FD.tradparitet()


def test_tradet_och_slutna_formen_ar_samma_kropp(paritet):
    """400 points, four face tilts: 1.73e-17 m is double precision, not agreement by tolerance."""
    assert paritet["max_abs_diff_m"] == pytest.approx(1.734723475976807e-17, rel=1e-6)
    assert paritet["max_abs_diff_m"] < 1e-16


def test_tradets_form(paritet):
    assert paritet["nodantal"] == 7
    assert paritet["lovoperationer"] == ["box", "halfspace"]


def test_kappa_har_ingen_primitiv(paritet):
    """The third shape parameter is not expressible; the cost of the nearest face that is, in metres."""
    k = paritet["kappa_ej_uttryckbar_m"]
    assert k["5.0"] == pytest.approx(2.4999843751816186e-06, rel=1e-9)
    assert k["20.0"] == pytest.approx(0.00015974481593780676, rel=1e-9)


def test_lovderivator_mot_central_fd():
    """d sdf/d theta and d grad/d theta from the leaf formulas, against central FD at h = 1e-6, per
    branch, with branch-flipping points skipped rather than averaged in."""
    per = FD.lovderivator_mot_fd()["per_steg"]["1e-06"]
    assert per["pad"]["n"] == 320 and per["face"]["n"] == 80
    assert per["pad"]["ds"] == pytest.approx(8.235523374366949e-12, rel=1e-6)
    assert per["pad"]["dg"] == pytest.approx(4.466897962629446e-10, rel=1e-6)
    assert per["face"]["ds"] == pytest.approx(4.3087061696311935e-12, rel=1e-6)
    assert per["face"]["dg"] == pytest.approx(1.0841823272489393e-10, rel=1e-6)


def test_fingrarna_delar_theta_och_speglar_rent():
    """The right finger is the left one mirrored in x, so the pair cannot drift apart under one
    theta -- the x components of the gradient and of d grad/d theta flip and nothing else does."""
    import numpy as np
    th = np.array([0.004, 0.02, 5.0])
    p = np.array([-0.07, 0.01, 0.005])
    sv, gv, stv, _, gtv, grenv = FD.sdf_finger(p, th, +1)
    sh, gh, sth, _, gth, grenh = FD.sdf_finger(np.array([-p[0], p[1], p[2]]), th, -1)
    assert sv == pytest.approx(sh, rel=1e-15)
    assert grenv == grenh
    assert gv[0] == pytest.approx(-gh[0], rel=1e-15)
    assert gv[1] == pytest.approx(gh[1], rel=1e-15)
    assert np.allclose(stv, sth, rtol=1e-15, atol=0.0)


def test_adjointreferensen_bars_som_data():
    """Composed through an exact-cone contact solver that lives in the motion engine, so it is
    carried, not re-derived: 3.22e-11 against FD over three step lengths, at a two-contact stick
    set with a condition number of 2.21."""
    a = FD.adjointreferens()
    assert a["max_relerr"] == pytest.approx(3.216932928122318e-11, rel=1e-9)
    assert [r["h"] for r in a["rader"]] == [1e-05, 3e-06, 1e-06]
    assert a["kontaktetiketter"] == ["stick", "stick"]
    assert a["adjoint_grad"][0] == 0.0                 # d/dr is identically zero at this grip


# --- the reach boundary ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sokning():
    return TG.kor_gransokning()


def test_reachtestet_delar_atta_klasser():
    """Three operations, three bits, eight classes -- manufacturability is not one bit."""
    assert TG.tillverkbarhet(TG.NOMINELL)[3] == 7
    trang = dict(TG.NOMINELL, groove_w=1.0, pocket_w=4.0, drill_r=20.0)
    assert TG.tillverkbarhet(trang) == (False, False, False, 0)
    assert TG.tillverkbarhet(dict(TG.NOMINELL, pocket_d=30.0))[1] is False


def test_posteriorn_hittar_varje_grans_inom_sitt_eget_band(sokning):
    for r in sokning:
        assert r["posterior_fel"] <= r["tolerans"], r["parameter"]


def test_posteriorn_betalar_farre_prober_an_svepet(sokning):
    """The mechanism: the sweep's cost is span/tol, the posterior's is its logarithm."""
    for r in sokning:
        assert r["posterior_prober"] < r["svep_prober"], r["parameter"]
    djup = [r for r in sokning if r["parameter"] == "pocket_d"][0]
    assert djup["svep_prober"] == 17
    assert djup["vinst"] > 4.0


def test_shippad_posterior_ar_billigare_och_mindre_generell(sokning):
    """This repository's single-transition posterior answers in 3 / 4 / 2 probes. The carried run
    paid 5 / 4 / 4 for a posterior that can also represent two transitions."""
    assert [r["posterior_prober"] for r in sokning] == [3, 4, 2]
    matt = TG.rikare_posterior_referens()
    assert [r["bisection_probes"] for r in matt] == [5, 4, 4]
    assert [r["sweep_probes"] for r in matt] == [7, 6, 17]
    assert [r["speedup"] for r in matt] == [1.4, 1.5, 4.25]
    assert matt[2]["bisection_est"] == pytest.approx(19.88294314381271, rel=1e-12)


def test_reliability_ett_ar_ren_bisektion():
    """With a perfectly believed probe the update is exact, which is the stated relationship."""
    p = TG.TroskelPosterior(0.0, 1.0, reliability=0.999999, n_grid=64)
    for _ in range(12):
        x = p.basta_probe()
        p.add_probe(x, x > 0.3)
    assert p.uppskattning() == pytest.approx(0.3, abs=0.02)


def test_posteriorn_vagrar_en_omojlig_reliability():
    with pytest.raises(ValueError):
        TG.TroskelPosterior(0.0, 1.0, reliability=1.0)
    with pytest.raises(ValueError):
        TG.TroskelPosterior(0.0, 1.0, reliability=0.2)


def test_tackningsjamforelsen_ar_inte_shippad():
    """The determinantal coverage leg is deliberately absent: a Latin hypercube matched or beat it."""
    import json
    with open(os.path.join(ROOT, "data", "tillverkbarhetsgrans_v1", "gransokning.json")) as fh:
        s = json.load(fh)["ej_shippad_tackningsjamforelse"]
    assert s["K_16"]["lhs"]["classes_mean"] >= s["K_16"]["dpp"]["classes_mean"]
    assert s["K_64"]["lhs"]["classes_mean"] == 8.0
    assert not hasattr(TG, "sample_dpp")


@pytest.mark.parametrize("modul", ["formderivata_v1.py", "tillverkbarhetsgrans_v1.py"])
def test_modulernas_egna_selftester(modul):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "field_engine", "opt", modul)],
                       capture_output=True, text=True, timeout=200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
