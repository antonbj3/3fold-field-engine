"""Routing by current through the distance field, and choosing where to measure a part.

The routing half runs live on two corpus tasks (about four seconds) and is checked against the
carried per-task rows of the full 11-task run; the measurement-planning half runs the whole
comparison live in under a second. The eleven-task sweep and the factory-hall runs take two minutes
and are carried.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine"))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine", "opt"))

import laplaceflode_v1 as LF                                                        # noqa: E402
import matplan_oed_v1 as OED                                                        # noqa: E402


@pytest.fixture(scope="module")
def ref():
    return LF.las_korpusreferens()


# --- routing --------------------------------------------------------------------------------------
def test_korpusen_i_repot_ar_den_som_mattes(ref):
    """The 11 tasks the run used are the 11 tasks committed here, by id."""
    carried = {r["task_id"] for r in ref["uppgifter"]}
    here = {t["task_id"] for t in LF.korpusuppgifter()}
    assert carried == here
    assert len(here) == 11


def test_flodet_loser_alla_elva_dar_centrumlinjen_inte_gor_det(ref):
    rader = ref["uppgifter"]
    assert sum(1 for r in rader if np.isfinite(r["len_lap_mm"])) == 11
    assert sum(1 for r in rader if np.isfinite(r["len_astar_mm"])) == 11
    assert sum(1 for r in rader if r["naive_feasible"]) == 3


def test_clearance_minst_lika_pa_sex_av_elva(ref):
    """Six, not the seven the run's summary reported: the count is recomputed from the rows."""
    s = LF.sammanfatta(ref["uppgifter"])
    assert s["clearance_minst_lika"] == 6
    per = {r["task_id"]: r for r in ref["uppgifter"]}
    t3 = per["t20260730_0003_00000000"]
    t4 = per["t20260730_0004_00000000"]
    assert t3["clr_lap_mm"] / t3["clr_astar_mm"] - 1 == pytest.approx(0.650, abs=0.005)
    assert t4["clr_lap_mm"] / t4["clr_astar_mm"] - 1 == pytest.approx(0.465, abs=0.005)


def test_svarighetsmattet_ar_fyndet(ref):
    """Effective resistance against the route's minimum clearance: Spearman -0.8636."""
    d = ref["svarighet"]
    assert d["spearman_reff_vs_clearance"] == pytest.approx(-0.8636, abs=5e-4)
    assert d["pearson_reff_vs_clearance"] == pytest.approx(-0.8315, abs=5e-4)
    assert d["mean_reff_hard_tasks"] > d["mean_reff_easy_tasks"]
    assert d["mean_reff_hard_tasks"] / d["mean_reff_easy_tasks"] == pytest.approx(1.65, abs=0.02)


def test_samtidig_routing_vinner_inte(ref):
    """Two routes: the flow solution is longer and five times slower at the same zero crossings.
    Four routes: it has one crossing where trying all 24 sequential orders has none."""
    h = ref["fabrikshall"]
    assert h["K=2"]["lap_tot_len_mm"] == 387.0
    assert h["K=2"]["seq_best_len_mm"] == 345.0
    assert h["K=2"]["lap_crossings"] == h["K=2"]["seq_best_crossings"] == 0
    assert h["K=2"]["lap_time_s"] > 4 * h["K=2"]["seq_time_s"]
    assert h["K=4"]["lap_crossings"] == 1
    assert h["K=4"]["seq_best_crossings"] == 0
    assert h["K=4"]["n_perms"] == 24


def test_tryckfallspastaendet_ar_inte_reproducerat(ref):
    """The summary claimed the flow route costs 0.03 % less pressure drop. Its own stored judge
    readings say it costs 2.87 % MORE. The claim is withdrawn and the rows are carried."""
    e = ref["ej_reproducerat"]
    flode, geodetik = e["lagrat_r_flode"][0], e["lagrat_r_geodetik"][0]
    assert flode > geodetik
    assert flode / geodetik - 1 == pytest.approx(0.0287, abs=5e-4)
    assert e["pastadd_relativ_andring"] < 0


def test_flodesrutten_reproduceras_live_pa_de_tva_svaraste(ref):
    """Live, against the repository's own corpus: the two tasks with the highest carried effective
    resistance come back with the same resistance and the same route clearance."""
    for r in sorted(ref["uppgifter"], key=lambda r: -r["r_eff"])[:2]:
        ut = LF.kor_uppgift(r["task_id"])
        assert ut["flode_loste"] and ut["astar_loste"]
        assert ut["r_eff"] == pytest.approx(r["r_eff"], rel=0.02)
        assert ut["clr_flode_mm"] == pytest.approx(r["clr_lap_mm"], rel=0.02)
        assert ut["clr_astar_mm"] == pytest.approx(r["clr_astar_mm"], rel=0.02)


def test_konduktansen_kommer_ur_avstandsfaltet():
    """A wider corridor must conduct more. Two identical grids, one with the obstacle moved out."""
    occ = np.zeros((20, 20), dtype=bool)
    occ[8:12, 5:15] = True
    edt = np.ones_like(occ, dtype=float)
    n, karta, u, v, kond_smal, _ = LF.bygg_gitterlaplacian(occ, edt, 1.0)
    _, _, _, _, kond_bred, _ = LF.bygg_gitterlaplacian(occ, edt * 3.0, 1.0)
    assert kond_bred.mean() > kond_smal.mean()
    assert kond_smal.min() >= LF.KONDUKTANS_GOLV ** LF.P_KONDUKTANS - 1e-12


def test_effektiv_resistans_ar_en_potentialskillnad():
    """On a free grid with no obstacle the resistance between two near cells must be smaller than
    between two distant ones -- the quantity is a property of the space, not of the solver."""
    occ = np.zeros((16, 16), dtype=bool)
    edt = np.full(occ.shape, 2.0)
    n, karta, u, v, kond, L = LF.bygg_gitterlaplacian(occ, edt, 1.0)
    s = int(karta[1, 1])
    nara, langt = int(karta[1, 3]), int(karta[14, 14])
    _, r_nara, i1 = LF.los_potential(L, n, s, nara)
    _, r_langt, i2 = LF.los_potential(L, n, s, langt)
    assert i1 == 0 and i2 == 0
    assert 0 < r_nara < r_langt


# --- measurement planning -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plan():
    return OED.utvardera_plan(k=8)


def test_jamn_spridning_ar_rangdefekt(plan):
    """Five of six dimensions, however many probes: an even spread never excites the sixth."""
    assert plan["jamn"]["rang"] == 5
    assert plan["vald"]["rang"] == 6
    assert plan["n_parametrar"] == 6
    assert plan["jamn"]["sigmin"] == 0.0
    assert plan["vald"]["sigmin"] > 1e9


def test_crb_sanks_femtio_gangor(plan):
    assert plan["jamn"]["medel_crb_m"] == pytest.approx(1.688451679617938e-04, rel=1e-4)
    assert plan["vald"]["medel_crb_m"] < plan["jamn"]["medel_crb_m"] / 40.0
    assert plan["vald"]["max_crb_m"] < 1e-5
    assert plan["jamn"]["max_crb_m"] == pytest.approx(1e-3, rel=1e-6)   # the prior, untouched


def test_dubbla_prober_botar_inte_rangbristen():
    r = OED.utvardera_plan(k=16)
    assert r["jamn"]["rang"] == 5
    assert r["jamn"]["medel_crb_m"] == pytest.approx(1.6819244729477043e-04, rel=1e-4)
    assert r["vald"]["rang"] == 6


def test_sigma_min_regeln_ar_ogradierad(plan):
    """The kept defect: on a rank-deficient accumulation every candidate scores exactly zero, so an
    E-optimal argmax cannot choose and falls back on input order."""
    theta = OED.Monteringsplatta.nominell_theta
    pool = OED.fisherpool(OED.Monteringsplatta, theta,
                          OED.Monteringsplatta.kandidatpunkter(theta))
    P = np.eye(len(theta)) * 1.0e6
    _, spar = OED.greedy_oed(pool, 4, P, tie_break="sigma")
    assert all(s == 0.0 for s in spar)
    _, spar_rank = OED.greedy_oed(pool, 4, P, tie_break="rank")
    assert all(s == 0.0 for s in spar_rank)      # still rank-deficient at k=4, so this is fair
    idx_sigma, _ = OED.greedy_oed(pool, 8, P, tie_break="sigma")
    idx_rank, _ = OED.greedy_oed(pool, 8, P, tie_break="rank")
    assert OED._spektrum(np.sum([pool[i] for i in idx_rank], axis=0))[1] == 6
    assert OED._spektrum(np.sum([pool[i] for i in idx_sigma], axis=0))[1] <= 6


def test_ogiltig_tiebreak_och_icke_finit_fisher_vagras():
    theta = OED.Monteringsplatta.nominell_theta
    pool = OED.fisherpool(OED.Monteringsplatta, theta,
                          OED.Monteringsplatta.kandidatpunkter(theta, n=8))
    with pytest.raises(ValueError):
        OED.greedy_oed(pool, 2, np.eye(6) * 1e6, tie_break="e-optimal")
    with pytest.raises(ValueError):
        OED._spektrum(np.array([[1.0, 0.0], [0.0, np.nan]]))


def test_den_matta_planen_bars_som_data():
    r = OED.las_referens()["monteringsplatta"]["results"]
    assert r["k_8"]["oed"]["rank"] == 6 and r["k_8"]["unif"]["rank"] == 5
    assert r["k_8"]["oed"]["mean_crb"] == pytest.approx(3.3165503410043685e-06, rel=1e-9)
    assert r["k_8"]["unif"]["mean_crb"] == pytest.approx(1.688451679617938e-04, rel=1e-9)
    assert r["k_16"]["unif"]["rank"] == 5


@pytest.mark.parametrize("modul", [os.path.join("opt", "laplaceflode_v1.py"), "matplan_oed_v1.py"])
def test_modulernas_egna_selftester(modul):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "field_engine", modul)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
