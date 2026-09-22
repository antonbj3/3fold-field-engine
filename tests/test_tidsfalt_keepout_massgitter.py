"""The mass-grid feed: occupancy is a threshold on the grid, and the depth error is under one cell.

A simulator hands the field engine its mass grid. Applying the field's material-in-cell reading
(mass > 0) puts the flag a B-spline node over the surface; applying a threshold in the feed instead,
`mass > theta*rho*dx**3`, puts it within one cell. This file locks that on three saved bed frames
(impact, largest single column, final collapse) against the highest particle per column from the same
trajectory: theta = 0.5 gives a p95 under one cell in every frame, and the presence reading does not.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine"))

import tidsfalt_keepout_v1 as TK                                                    # noqa: E402

DATA = os.path.join(ROOT, "data", "tidsfalt_keepout_v1", "massgitter_3ramar.npz")
RES = 96
# the audited per-frame p95 at theta = 0.5, mm (field dom against the particle top)
P95_UR_AUDITEN = ((1500, 0.48646033), (9500, 0.02105444), (25000, 0.19093736))


def _top_of(occ):
    idx = np.arange(RES, dtype=np.int32)[None, :, None]
    return np.max(np.where(occ, idx, np.int32(-1)), axis=1).astype(np.int32)


@pytest.fixture(scope="module")
def gitter():
    z = np.load(DATA)
    return dict(mass=z["mass"], surface_top=z["surface_top"], steps=z["steps"],
                dx=float(z["dx"]), rho=float(z["rho"]))


def _p95(gitter, frame, theta):
    dx, rho = gitter["dx"], gitter["rho"]
    occ = TK.ockupans_fran_massgitter(gitter["mass"][frame], rho, dx, theta)
    top = _top_of(occ).astype(np.float64)
    pt = gitter["surface_top"][frame].astype(np.float64)
    m = (top >= 0) & np.isfinite(pt)
    return float(np.quantile((top[m] * dx - pt[m]) * 1000.0, 0.95)), int(m.sum())


def test_referensdata_ar_under_en_megabyte(gitter):
    """Three frames of a 96^3 grid, lzma-compressed, are carried as reference data under 1 MB."""
    assert os.path.getsize(DATA) <= 1_000_000
    assert gitter["mass"].shape == (3, RES, RES, RES)
    assert gitter["mass"].dtype == np.float32
    assert tuple(int(s) for s in gitter["steps"]) == (1500, 9500, 25000)


def test_occupancyn_ar_troskeln_pa_massgittret(gitter):
    """The function only applies the feed's one line; the default is half a cell mass."""
    massa = gitter["mass"][0]
    troskel = 0.5 * gitter["rho"] * gitter["dx"] ** 3
    fordon = TK.ockupans_fran_massgitter(massa, gitter["rho"], gitter["dx"])
    assert fordon.dtype == bool
    assert np.array_equal(fordon, massa > troskel)
    assert np.array_equal(fordon, TK.ockupans_fran_massgitter(massa, gitter["rho"],
                                                              gitter["dx"], theta=0.5))


def test_p95_under_en_cell_alla_tre_ramarna(gitter):
    """theta = 0.5: p95 <= 1 cell (2.0833 mm) in every saved frame, against the highest particle."""
    cell_mm = gitter["dx"] * 1000.0
    for f, step in enumerate(gitter["steps"]):
        p95, n = _p95(gitter, f, 0.5)
        assert n > 0
        assert p95 <= cell_mm, (int(step), p95)


def test_p95_talen_ar_auditens(gitter):
    """The three locked p95 readings, mm, on the saved frames."""
    for f, (step, vante) in enumerate(P95_UR_AUDITEN):
        p95, _ = _p95(gitter, f, 0.5)
        assert int(gitter["steps"][f]) == step
        assert p95 == pytest.approx(vante, abs=5e-4)


def test_narvarotroskeln_klarar_inte_gransen(gitter):
    """The mechanism: with the field's material-in-cell reading (theta = 0) the flag sits over the
    surface and the p95 passes one cell, so the threshold has to be in the feed."""
    cell_mm = gitter["dx"] * 1000.0
    p95_0, _ = _p95(gitter, 0, 0.0)
    assert p95_0 > cell_mm
    p95_25, _ = _p95(gitter, 0, 0.25)
    assert p95_25 > cell_mm
    p95_5, _ = _p95(gitter, 0, 0.5)
    assert p95_5 <= cell_mm
