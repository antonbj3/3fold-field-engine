"""The volume meter change, separated from the engine, and the pitch that is the real gain.

The point of this file is that the meter change is locked at what it is worth on UNCHANGED field
bytes -- 13 -> 17 of 70 -- and not at the 48/70 that an OR of both meters plus a refined pitch
reaches together. The sphere test is the mechanism behind it and runs live in about a second.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine"))

import faltvolym_v1 as FV                                                           # noqa: E402


@pytest.fixture(scope="module")
def data():
    return FV.las_matarbyte()


def test_matarbytet_pa_oforandrade_faltbytes(data):
    """13 -> 17 of 70, and the two readings come off the same field hash per part."""
    m = FV.matarbytet(data["delar"])
    assert (m["voxel"], m["marching_cubes"], m["n"]) == (13, 17, 70)
    assert m["identiska_faltbytes"] is True


def test_en_del_regredierar(data):
    """Five parts cross into the band and one crosses out. 'Zero regressions' was wrong."""
    m = FV.matarbytet(data["delar"])
    assert len(m["vunna"]) == 5
    assert m["forlorade"] == ["dpp_r1_51"]
    assert m["marching_cubes"] - m["voxel"] == 4
    assert data["matarbytets_regression"] == ["dpp_r1_51"]


def test_hela_vinsten_ar_pitchen_inte_mataren(data):
    """On the refined field the isosurface is the WORSE meter: 46 against 47, and the OR of both
    reaches 48. So the +35 belongs to the pitch, not to the meter."""
    h = data["hela_katalogen"]
    assert h["or_voxel_mc_med_cadstyrd_pitch"] == 48
    assert h["mc_ensam"] == 46
    assert h["voxel_ensam"] == 47
    assert h["mc_ensam"] < h["voxel_ensam"]


def test_finfunktionsvald_pitch_pa_tunnvaggsdelarna(data):
    """3 -> 15 of 19 on the isosurface reading, 1 -> 15 on the Riemann sum, for 88x the field time."""
    f = FV.finfunktionsutfall(data)
    assert f["n_c2"] == 19
    assert (f["begard_pitch"]["mc"], f["vald_pitch"]["mc"]) == (3, 15)
    assert (f["begard_pitch"]["voxel"], f["vald_pitch"]["voxel"]) == (1, 15)
    assert f["kostnadskvot_median"] == pytest.approx(88.128671, rel=1e-6)
    assert f["sekunder_begard"] == pytest.approx(0.44893777, rel=1e-6)
    assert f["sekunder_vald"] == pytest.approx(147.00411, rel=1e-6)


def test_fyra_delar_star_kvar_roda_och_tva_bryter_sin_egen_grans(data):
    f = FV.finfunktionsutfall(data)
    assert len(f["kvar_rott"]) == 4
    assert f["bundna_fel"] == ["dpp_r1_59", "dpp_r3_33"]


def test_platt_vagg_ar_osynlig_for_finfunktionsmatningen(data):
    """The counter-test: a 10 x 10 x 0.1 mm plate gets the same pitch and a bit-identical field with
    the choice on and off, while its occupancy reads zero against an exact 10 mm^3."""
    p = FV.platt_vagg_ar_osynlig(data)
    assert p["pitch_av"] == p["pitch_auto"] == 1.0
    assert p["faltet_oforandrat"] is True
    assert p["voxelvolym"] == 0.0
    assert p["matt_feature_radie"] is None
    assert p["exakt_volym_mm3"] == pytest.approx(10.0, rel=1e-12)


def test_konvergensordningen_ar_ouppklarad(data):
    """Halving the pitch makes it WORSE on 14 of 70 parts for one meter and 13 for the other, so no
    asymptotic order may be claimed from this catalogue."""
    o = FV.ordning_ouppklarad(data)
    assert (o["voxel_samre"], o["mc_samre"]) == (14, 13)
    assert o["voxel_median_ordning"] == pytest.approx(1.5128959413, rel=1e-9)
    assert o["mc_median_ordning"] == pytest.approx(1.879288091, rel=1e-9)


def test_gitterfas_ar_mekanismen():
    """Live, on an analytic sphere: shifting the lattice by 0.137 pitch changes no geometry, moves
    the Riemann sum by about 6 % of the exact volume and the isosurface by about 0.03 %."""
    g = FV.gitterfasprov()
    assert g["voxel_spridning"] > 0.05
    assert g["mc_spridning"] < 0.001
    assert g["mc_spridning"] < g["voxel_spridning"] / 100.0


def test_matarna_pa_ett_fritt_falt():
    """Both meters on one field, with the isosurface's honest refusal when there is no crossing."""
    import numpy as np
    h = 0.5
    ax = [np.arange(-8.0, 8.0 + h, h) for _ in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    sfar = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 5.0
    assert FV.voxel_riemann_volym(sfar, h) > 0.0
    assert FV.marching_cubes_volym(sfar, h)[0] > 0.0
    assert FV.marching_cubes_volym(np.full_like(sfar, 1.0), h) == (0.0, 0)
    assert FV.marching_cubes_volym(np.full_like(sfar, -1.0), h) == (0.0, 0)
    assert FV.voxel_riemann_volym(np.full_like(sfar, 1.0), h) == 0.0


def test_modulens_egen_selftest():
    p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "field_engine", "faltvolym_v1.py")],
                       capture_output=True, text=True, timeout=200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
