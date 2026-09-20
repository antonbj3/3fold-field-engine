"""The owner-indexed field's container, the allocation law, and every caveat that travels with them.

Two things are locked here. The MECHANISMS run live on a small synthetic cell: the container is a
function of the field alone, an unstored owner channel reads the declared bound, and normalising a
goal by its own reference beats combining raw scales. The MEASURED numbers -- storage ratios, gate
deviations, byte accounts -- come from runs that take a minute or more each and are carried as data,
together with the three audit results that cut them down.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine"))

import kropps_id_falt_v1 as KF                                                      # noqa: E402
import pareto_allokering_v1 as PA                                                   # noqa: E402


@pytest.fixture(scope="module")
def data():
    return KF.las_allokering()


# --- the container, live --------------------------------------------------------------------------
def test_behallaren_ar_en_funktion_av_faltet():
    """Two builds of the same declaration give the same bytes. Without this no storage ratio means
    anything, because the numerator would not be reproducible."""
    a, b = KF.bygg_syntetisk_cell(), KF.bygg_syntetisk_cell()
    assert a.sha256() == b.sha256()
    assert a.serialisera() == b.serialisera()


def test_ostorda_agarkanaler_lamnar_bunden():
    """Outside an owner's window the channel returns the declared bound, not a small number that
    contact logic could mistake for a gap."""
    f = KF.bygg_syntetisk_cell()
    langt = f.query_target(f.ursprung + np.array([0.35, 0.35, 0.25]), 999)
    assert float(langt[0]) == KF.UTANFOR_BOUND_M
    assert KF.UTANFOR_BOUND_M > 0.01


def test_lagrad_agarkanal_ar_ett_prov():
    f = KF.bygg_syntetisk_cell()
    ow, bi = f.poster[0]
    ny, nz = f.blockform[1], f.blockform[2]
    ijk = np.array([bi // (ny * nz), (bi // nz) % ny, bi % nz])
    P = f.ursprung + (ijk + 0.5) * f.blockstorlek
    v = float(f.query_target(P, int(ow))[0])
    assert v != KF.UTANFOR_BOUND_M
    assert abs(v) < 0.2


def test_fraga_utanfor_domanen_vagras():
    """A field that quietly answers outside its own support cannot be audited."""
    f = KF.bygg_syntetisk_cell()
    with pytest.raises(ValueError):
        f.query(np.array([[-1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        f.query(np.array([[99.0, 0.0, 0.0]]))


def test_kostnadsformeln_ar_monoton_i_nivan():
    """A finer level costs more, and the formula is what every ratio here is built from."""
    assert KF.nivaprover(0) == 1 and KF.nivaprover(5) == 17 ** 3
    grov = KF.lagrade_bytes(np.zeros(10, np.uint8), np.zeros(4, np.uint8))
    fin = KF.lagrade_bytes(np.full(10, 5, np.uint8), np.zeros(4, np.uint8))
    assert fin > grov


# --- the measured cell, carried -------------------------------------------------------------------
def test_lagringskvoten_pa_maskincellen(data):
    m = data["maskincell"]
    assert m["bytes"] == 2836836
    assert m["uniform_bytes"] == 58320276
    assert m["kvot"] == pytest.approx(20.558212036226276, rel=1e-12)
    assert m["kvot"] == pytest.approx(m["uniform_bytes"] / m["bytes"], rel=1e-15)


def test_grindarna_pa_maskincellen(data):
    m = data["maskincell"]
    assert m["compliance_avvikelse"] == pytest.approx(0.03989965913578663, rel=1e-12)
    assert m["compliance_avvikelse"] < data["compliance_mot_grinden"]["compliance_grind"]
    assert m["rutt"]["length"] == pytest.approx(7.2, rel=1e-12)
    assert m["stabilitet_avvikelse"] == pytest.approx(3.1193201202837435e-12, rel=1e-9)
    assert m["matta_allokeringsfall"] == 88
    assert m["identitet_a"] == m["identitet_b"]


def test_44_av_48_ar_inte_44_ratta_punkter(data):
    """10 missing and 6 added, on 16 of 16 pairs. A count that is 92 % right can be a support set
    that is qualitatively wrong."""
    m, e = data["maskincell"], data["extern_kroppsmask"]
    assert m["kontakt_antal"] == 44 and m["referens_kontakt_antal"] == 48
    assert m["kontakt_par"] == 16
    assert e["saknade_punkter"] == 10
    assert e["tillkomna_punkter"] == 6
    assert e["saknade_punkter"] - e["tillkomna_punkter"] == 48 - 44


def test_skalar_union_plus_extern_mask_klarar_alla_grindar(data):
    """The representation claim FALLS: the same scalar union with an external body mask gets 48/48.
    What the run showed was an integration defect, not a limit of the union."""
    e = data["extern_kroppsmask"]
    assert e["kontakt_antal"] == 48
    assert e["passerar_alla"] is True
    assert e["gapfel_m"] < 1e-16
    u = e["skalar_union_utan_mask"]
    assert u["antal"] == 452
    assert u["kontaktantalsfel"] > 8.0


def test_motcellen_passerar_noll_av_atta(data):
    """The 20x does not transfer: on a bolted assembly every budget from 1 % to 100 % fails."""
    mc = data["motcell"]
    assert (mc["passerande"], mc["av"]) == (0, 8)
    assert all(r["passerar_alla"] is False for r in mc["rader"].values())
    assert all(r["stabilitet"] > 0.1 for r in mc["rader"].values())


# --- the allocation law, live ---------------------------------------------------------------------
def test_referensnormalisering_slar_ra_skala():
    """Goals whose units differ by 100x: normalising equalises the error, raw scale does not."""
    c, ref = PA.syntetiskt_flermalsproblem()
    M_ref, _ = PA.vattenfyllning(PA.pareto_ref_vikter(c, ref), 0.2)
    M_ra, _ = PA.vattenfyllning(PA.rasumma_vikter(c), 0.2)
    f_ref, f_ra = PA.malfel(M_ref, c, ref), PA.malfel(M_ra, c, ref)
    assert f_ref.max() < f_ra.max()
    assert f_ref.max() - f_ref.min() < f_ra.max() - f_ra.min()


def test_vattenfyllningen_spenderar_budgeten_monotont():
    c, ref = PA.syntetiskt_flermalsproblem()
    w = PA.pareto_ref_vikter(c, ref)
    varsta = [PA.malfel(PA.vattenfyllning(w, b)[0], c, ref).max()
              for b in (0.05, 0.1, 0.2, 0.4, 1.0)]
    assert varsta == sorted(varsta, reverse=True)
    bytes_ = [PA.bytes_for_allokering(PA.vattenfyllning(w, b)[0]) for b in (0.05, 0.2, 1.0)]
    assert bytes_ == sorted(bytes_)


def test_golv_och_tak_halls():
    c, ref = PA.syntetiskt_flermalsproblem()
    M, _ = PA.vattenfyllning(PA.pareto_ref_vikter(c, ref), 0.01)
    assert M.min() >= PA.M_GOLV
    M, _ = PA.vattenfyllning(PA.pareto_ref_vikter(c, ref), 10.0)
    assert M.max() <= PA.M_TAK


def test_nollreferens_vagras():
    c, _ = PA.syntetiskt_flermalsproblem()
    with pytest.raises(ValueError):
        PA.pareto_ref_vikter(c, np.array([1.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        PA.vattenfyllning(np.ones(4), 0.2, mult=np.array([1.0, 0.0, 1.0, 1.0]))


# --- the allocation's measured numbers and their conditions ---------------------------------------
def test_flermalsallokeringen_2d(data):
    t = data["flermal_2d"]
    assert t["pareto_ref_bytes"] == 2640
    assert t["per_mal_asynkront_bytes"] == 3988
    assert t["per_mal_synkront_bytes"] == 7828
    assert 1 - t["pareto_ref_bytes"] / t["per_mal_asynkront_bytes"] == pytest.approx(0.338, abs=1e-3)
    assert 1 - t["pareto_ref_bytes"] / t["per_mal_synkront_bytes"] == pytest.approx(0.663, abs=1e-3)


def test_referensen_kostar_mer_an_den_sparar(data):
    """Charged symmetrically the allocation costs 120.5 % of the full field. The saving holds only
    when the reference is already being paid for elsewhere."""
    a = data["flermal_2d"]["referenskostnad"]
    assert a["nominal_candidate_bytes"] == 2640
    assert a["reference_float32_bytes"] == 13284
    assert a["reference_plus_candidate_bytes"] == 15520
    assert a["reference_plus_candidate_percent"] == pytest.approx(120.4968944099379, rel=1e-12)
    assert a["reference_plus_candidate_percent"] > 100.0
    assert a["nominal_full_bytes"] == 12880


def test_flermalsallokeringen_3d(data):
    t = data["flermal_3d"]
    assert t["pareto_ref"]["bytes"] == 617980
    assert t["per_goal"]["bytes"] == 753652
    assert max(t["pareto_ref"]["err_pct"]) == pytest.approx(4.172612033810036, rel=1e-12)
    assert t["uniform"]["bytes"] > 8 * t["pareto_ref"]["bytes"]
    assert max(t["uniform"]["err_pct"]) > 19.0


def test_uniform_missar_kontaktantalet_aven_vid_full_budget(data):
    """Uniform is not merely a coarser allocation here: given 100 % of the budget it still reads
    8.33 % on the contact count, above a 5 % band, and the scalar union reads 698 %."""
    rader = {(r["kind"], r["budget"]): r for r in data["uniform_missar_kontakt"]}
    agare_full = rader[("owner", 1.0)]
    assert agare_full["bytes"] == 58320276
    assert agare_full["values"]["contact"]["count"] == 44
    assert agare_full["errors"]["contact_count_error"] == pytest.approx(0.08333333333333333, rel=1e-12)
    assert agare_full["errors"]["contact_count_error"] > 0.05
    skalar_full = rader[("scalar", 1.0)]
    assert skalar_full["errors"]["contact_count_error"] > 6.0
    assert skalar_full["errors"]["pass_contact"] is False


# --- what the assembly says about whether a field is needed at all --------------------------------
def test_assemblyns_lagringsandel(data):
    a = data["assembly"]
    assert a["block"] == 400 and a["kanaler"] == 620
    assert a["pareto_bytes"] == 886263
    assert a["uniform_full_bytes"] == 20051151
    assert a["andel_av_full"] == pytest.approx(4.420010601885148, rel=1e-12)
    assert a["uniform_minsta_passerande_bytes"] == 2980431


def test_cad_vagen_gor_samma_tre_mal_utan_falt(data):
    """91.17 ms without a field against 96.06 ms with one, while reading 1.1 MB of input -- not the
    5.8 kB the report credited. For these three goals on rigid parts the field earns nothing."""
    c = data["cad_vagen_utan_falt"]
    assert c["cad_median_ms"] < c["falt_median_ms"]
    assert c["inlasta_bytes_totalt"] > 100 * data["assembly"]["cad_referens_pastadd_bytes"]


def test_kontaktmarginalens_referens_ar_natberoende(data):
    """An independent triangle assembly of the same parts moves the contact-margin reference by
    100.2 %, at a condition number of 9.7e9. The goal is not mesh-independent."""
    m = data["m1_natberoende"]
    assert m["rel_fel_procent"] == pytest.approx(100.20075114426771, rel=1e-9)
    assert m["kondition"] > 1e9


def test_kontaktparitet_faller_mot_stralad_narfas(data):
    """Five pairs from the field against three from a ray-traced narrow phase: the two bolt pairs
    exist only in the field's reading."""
    nominell = [p for p in data["kontaktparitet"] if p["etikett"] == "nominal"][0]
    assert len(nominell["falt_par"]) == 5
    assert len(nominell["rt_par"]) == 3
    assert set(nominell["rt_par"]) < set(nominell["falt_par"])


def test_bitidentiteten_star(data):
    assert data["bitidentitet"]["artefakter"] == 42


@pytest.mark.parametrize("modul", ["kropps_id_falt_v1.py", "pareto_allokering_v1.py"])
def test_modulernas_egna_selftester(modul):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "field_engine", modul)],
                       capture_output=True, text=True, timeout=200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
