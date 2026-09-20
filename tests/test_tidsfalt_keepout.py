"""A keep-out box against an occupancy that changes, and the tick the field does not fit in.

The mechanism runs live on a small synthetic pile in a fraction of a second: a convex hull of the
same particles claims the corridor from the first frame, while the particle count says it is empty
for thirty. The measured 200 000-particle run needs a particle simulator and a GPU, so its numbers
are carried -- including the three that do not support the reading they were given.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine"))

import tidsfalt_keepout_v1 as TK                                                    # noqa: E402


@pytest.fixture(scope="module")
def syntetisk():
    return TK.kor_syntetisk()


@pytest.fixture(scope="module")
def matning():
    return TK.las_matning()


# --- the mechanism, live --------------------------------------------------------------------------
def test_holjet_broar_gapet_innan_det_finns_nagot_dar(syntetisk):
    """The hull claims material in the corridor from frame 0; the first particle arrives at 30."""
    assert syntetisk["forsta_holjeintrang"] == 0
    assert syntetisk["forsta_verkligt_intrang"] == 30
    assert syntetisk["forsta_holjeintrang"] < syntetisk["forsta_verkligt_intrang"]
    assert len(syntetisk["holjets_falska_ramar"]) == 30


def test_partikelraknaren_ar_monoton_efter_spillet(syntetisk):
    """Once the spill starts the count only grows: the detector is not flickering."""
    efter = [r["partiklar_i_ladan"] for r in syntetisk["rader"][30:]]
    assert efter == sorted(efter)
    assert efter[-1] > 0


def test_ockupansen_sager_var_material_finns():
    """A cell is occupied because a particle is in it. An empty region rasterises empty, which is
    exactly what the hull cannot do."""
    ramar = TK.syntetisk_hog()
    occ = TK.ockupans_fran_partiklar(ramar[5], (-0.16, -0.06, 0.0), (40, 16, 8), 0.008, 0.006)
    assert occ.any() and not occ.all()
    mitten = occ[18:22]                       # the corridor, before the spill
    assert not mitten.any()
    sen = TK.ockupans_fran_partiklar(ramar[-1], (-0.16, -0.06, 0.0), (40, 16, 8), 0.008, 0.006)
    assert sen[18:22].any()                   # after the spill it is occupied, and only then


def test_tom_ram_ger_tom_ockupans():
    occ = TK.ockupans_fran_partiklar(np.zeros((0, 3)), (0.0, 0.0, 0.0), (4, 4, 4), 0.01, 0.005)
    assert occ.shape == (4, 4, 4) and not occ.any()
    assert TK.keepout_intrang(np.zeros((0, 3)), TK.KEEPOUT_MIN, TK.KEEPOUT_MAX) == 0


def test_keepoutladan_ar_en_ren_inneslutning():
    p = np.array([[0.0, 0.0, 0.01], [0.5, 0.0, 0.01], [0.0, 0.0, 0.05]])
    assert TK.keepout_intrang(p, TK.KEEPOUT_MIN, TK.KEEPOUT_MAX) == 1


# --- the measured run, carried --------------------------------------------------------------------
def test_ruttomlaggning_och_forsta_intrang(matning):
    s = matning["sammanfattning"]
    assert s["divert_frame"] == 19
    assert s["divert_time_s"] == pytest.approx(0.20, rel=1e-9)
    assert s["first_breach_frame"] == 73
    assert matning["rader"][73]["time_s"] == pytest.approx(0.74, rel=1e-9)
    assert matning["keepout_partiklar_vid_slutet"] == 227
    assert matning["n_uppdateringar"] == 200


def test_lagring_och_takt(matning):
    s, lag = matning["sammanfattning"], matning["lagring"]
    assert lag["pareto_to_uniform_ratio_pct"] == pytest.approx(33.5715, abs=1e-3)
    assert lag["pareto_winning_bytes"] == 935709
    assert lag["uniform_winning_bytes"] == 2787213
    assert s["field_median_build_ms"] == pytest.approx(84.6647, abs=1e-3)
    assert s["cad_median_time_ms"] == pytest.approx(4.8475, abs=1e-3)
    assert s["field_median_build_ms"] / s["cad_median_time_ms"] == pytest.approx(17.47, abs=0.01)
    assert s["field_total_time_ms"] / s["cad_total_time_ms"] == pytest.approx(17.98, abs=0.01)
    assert s["field_median_build_ms"] > 8 * TK.TICK_MS


# --- the speed limit, carried ---------------------------------------------------------------------
def test_inkrementell_uppdatering_nar_inte_takten(matning):
    g = TK.inkrementell_grans(matning)
    assert g["kvot_median"] == pytest.approx(0.41717, abs=1e-4)
    assert g["ms_median"] == pytest.approx(35.3175, abs=1e-3)
    assert g["ramar_inom_tick"] == 0.0
    assert g["kvot_som_kravs"] == pytest.approx(0.11812, abs=1e-4)
    assert g["basta_kvot"] > g["kvot_som_kravs"]


def test_tva_oberoende_sparrar(matning):
    """The occupied-block share is 70 %, not 8 %, and the overhead alone is the whole tick."""
    g = TK.inkrementell_grans(matning)
    assert g["andrade_block_av_ockuperade"] == pytest.approx(0.6989, abs=1e-3)
    assert g["andrade_block_av_alla"] == pytest.approx(0.0774, abs=1e-3)
    assert g["overhead_ms"] == pytest.approx(10.4184, abs=1e-3)
    assert g["overhead_ms"] > TK.TICK_MS


def test_genvagen_kostar_noggrannhet(matning):
    """291 stale blocks and 14.26 mm of drift at two seconds, so 35.32 ms is a lower bound."""
    g = TK.inkrementell_grans(matning)
    assert g["inaktuella_block"] == 291
    assert g["drift_m"] == pytest.approx(0.01426046, rel=1e-6)


# --- the three readings that do not survive -------------------------------------------------------
def test_faltets_egen_detektor_utloser_sent(matning):
    """The run's own field detector first fires at frame 143, not at 73. 'The field catches what
    the hull misses' is not what was measured."""
    assert matning["faltets_egen_forsta_detektion"] == 143
    assert matning["referensens_forsta_detektion"] == 73
    assert matning["faltets_egen_forsta_detektion"] > matning["referensens_forsta_detektion"]


def test_holjets_missar_prisas_mot_partikelantalet(matning):
    """The disagreement is computed against the particle count, never against the field, so it
    prices the hull's error and not the field's advantage."""
    assert "particle count" in matning["matgranser"]["cad_missad_jamfors_mot"]
    missade = [r for r in matning["rader"] if r.get("cad_missed_breach")]
    assert len(missade) == 59
    assert all(r["ref_keepout_breach"] for r in missade)


def test_ruttbenet_ar_ingen_matning(matning):
    """The recorded route clearance is a single constant over all 200 frames, and the recorded hull
    route error has a median of 98.8 % over a 28-100 % spread. Neither is evidence, and neither is
    cited by this module."""
    assert matning["matgranser"]["ruttclearance_unika_varden"] == [0.5]
    lo, hi = matning["matgranser"]["cad_ruttfel_procent_spann"]
    assert hi > 99.0 and lo > 28.0
    fel = sorted(r["cad_route_err_pct"] for r in matning["rader"])
    assert fel[len(fel) // 2] > 98.0


def test_modulens_egen_selftest():
    p = subprocess.run([sys.executable,
                        os.path.join(ROOT, "src", "field_engine", "tidsfalt_keepout_v1.py")],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
