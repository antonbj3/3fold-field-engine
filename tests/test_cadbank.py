"""The CAD bench's gate counts, its determinism ledger, and the grammar reader's extraction leg.

Every count here is RE-DERIVED from the carried per-part records by the bench's own gate functions,
so the assertion locks the number to the measurement and not to a second copy of the number. The
70-part measurement itself needs the exact B-rep kernel and five minutes per process, which is why
its records are data; nothing in this file exceeds a second.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine"))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine", "recipe"))

import cadbank_v1 as CB                                                             # noqa: E402
import kravlasare_regex_v1 as KR                                                    # noqa: E402


@pytest.fixture(scope="module")
def delar():
    return CB.las_delar()


def test_katalogen_ar_70_delar(delar):
    assert len(delar) == 70
    assert len({d["id"] for d in delar}) == 70


def test_familj_a_volym_yta_masscentrum(delar):
    """The field against the exact solid: the volume integral is where it loses, and the centre of
    mass is where it does not."""
    a = CB.familjeutfall(delar)["a"]
    assert (a["volym"], a["yta"], a["masscentrum"]) == (13, 27, 67)
    assert a["samtidigt"] == 11


def test_familj_b_avstandsfaltet(delar):
    """The distance field within 1.5 pitch of the exact surface distance."""
    assert CB.familjeutfall(delar)["b"]["avstandsfalt"] == 67


def test_familj_c_rundresan(delar):
    """STEP -> field -> section loft -> STEP. 13/70 is the round trip's honest number."""
    c = CB.familjeutfall(delar)["c"]
    assert c["samtidigt"] == 13
    assert c["volym"] == 13


def test_tillverkningsgrindarna():
    """Turning against the tool nose radius, milling and drilling against a voxel oracle."""
    t = CB.tillverkningsutfall()
    assert t["svarv"] == (4, 4)
    assert t["fras_borr"] == (2, 2)
    assert t["max_svarvfel_mm"] == pytest.approx(0.037368601148976666, rel=1e-12)


def test_determinism_8817():
    """8 817 = 4 868 parts + 2 340 screening + 1 284 generation + 53 manufacturing + 264 physics
    scalars + 8 extra hashes, all equal across two fresh processes."""
    d = CB.determinismutfall()
    assert d["jamforelser"] == 8817
    assert d["lika"] == 8817
    g = d["grupper"]
    assert (g["parts"]["comparisons"], g["screening"]["comparisons"],
            g["generation"]["comparisons"], g["manufacturing"]["comparisons"]) == (4868, 2340, 1284, 53)
    # the measured run's own report printed the pre-audit total; the audit re-derived it in separate processes
    assert d["rapporterat_i_korningen"]["comparisons"] == 8809


def test_determinismjamforaren_falskt_positivt():
    """A dropped field must count as a comparison that is NOT equal. If it did not, a run that
    stopped writing half its arrays would report perfect agreement."""
    a = {"arrays": {"x": {"sha256": "aa"}, "y": {"sha256": "bb"}}, "numbers": {"v": "cc"},
         "metrics_sha256": "dd"}
    assert CB.jamfor_hashposter(a, a) == (4, 4)
    b = {"arrays": {"x": {"sha256": "aa"}}, "numbers": {"v": "cc"}, "metrics_sha256": "dd"}
    assert CB.jamfor_hashposter(a, b) == (4, 3)
    c = dict(a, metrics_sha256="ee")
    assert CB.jamfor_hashposter(a, c) == (4, 3)


def test_densitetskvotsgrinden_ar_inte_geometrisk():
    """15/15 at a maximum of 0.046 31242218 -- and the same leg is blind to the pitch, so it is a
    density-ratio check on a fixed mesh, not geometric validation."""
    d = CB.densitetskvotsgrind()
    assert (d["stelhet_passerade"], d["n_delar"]) == (15, 15)
    assert d["max_relativ_avvikelse"] == pytest.approx(0.04631242218003605, rel=1e-12)
    assert d["halverad_pitch_identisk"] is True
    assert d["halverad_pitch_numeriska_falt"] == 250
    assert d["geometrisk_validering"] is False


def test_max_avvikelsen_ar_simp_lagen():
    """The measured maximum is the closed-form SIMP ratio at that part's density to eleven digits,
    which is the proof that no geometry entered the number."""
    d = CB.densitetskvotsgrind()
    assert CB.simp_forhallande(0.876238617139) == pytest.approx(d["max_relativ_avvikelse"], rel=1e-9)
    assert CB.simp_forhallande(0.5) == pytest.approx(0.0459990784091, rel=1e-9)


def test_kontakt_och_gradientbenen():
    d = CB.densitetskvotsgrind()
    assert (d["kontaktmarginal_passerade"], d["gradient_passerade"]) == (15, 15)
    assert d["max_kontaktmarginalfel"] == pytest.approx(2.7830982906463244e-05, rel=1e-12)


def test_grammatiklasaren_extraherar_139_av_195():
    """The reader runs live here: 40 intentions, about half a millisecond each, no model."""
    m = KR.matt_extraktion(KR.las_intentioner())
    assert (m["traffar"], m["falt"]) == (139, 195)
    assert round(m["andel"], 4) == 0.7128


def test_grammatiklasaren_per_falt():
    """Two field names the grammar has no pattern for at all -- the misses are not all near-misses."""
    per = KR.matt_extraktion(KR.las_intentioner())["per_falt"]
    assert per["material"] == (30, 30)
    assert per["thickness"] == (23, 25)
    assert per["height"] == (0, 5)
    assert per["radius"] == (0, 2)


def test_verdiktbenet_bars_som_data():
    """The verdict leg was measured through a claim-type checker that is not a dependency here, so
    its numbers travel as data with the caveat attached, not as a live assertion."""
    import json
    with open(os.path.join(CB.DATA, "regexutfall.json")) as fh:
        r = json.load(fh)["matt_grammatikbaslinje"]
    assert r["n_accepted"] == 29
    assert r["false_accepts"] == 1
    assert r["verdict_accuracy"] == 0.925
    assert r["overall_field_accuracy"] == 0.7128            # the leg this file reproduces live


@pytest.mark.parametrize("modul", ["cadbank_v1.py", "recipe/kravlasare_regex_v1.py"])
def test_modulernas_egna_selftester(modul):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "field_engine", modul)],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
