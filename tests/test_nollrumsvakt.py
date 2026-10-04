"""A rank-deficient or inconsistent system gives INFEASIBEL, OSÄKER or a refusal, never a finite answer.

Two solvers here returned a quiet answer from a system whose null space made it meaningless:
the Laplace solve in laplaceflode_v1 (inlet and outlet in different rooms: no current path, yet a
finite effective resistance of -5.09e14 that ranks the task as the easiest) and the circle fit in
field_to_recipe_v1 (a flat strip: rank 2 of 3, yet an accepted cylinder whose radius is set by the
coordinate origin). The sketch solver already derives rank from its Jacobian; its refusals are
locked here as regression tests.
"""
import os
import sys

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("", "opt", "recipe"):
    sys.path.insert(0, os.path.join(ROOT, "src", "field_engine", _p))

import field_to_recipe_v1 as F                                                      # noqa: E402
import laplaceflode_v1 as LF                                                        # noqa: E402
from sketch_gcs_v1 import SketchGCS                                                 # noqa: E402


# --- laplaceflode_v1: the component-indicator null space -------------------------------------------
def _rum(dorr):
    occ = np.zeros((24, 24), dtype=bool)
    occ[:, 12] = True
    if dorr:
        occ[10:14, 12] = False
    return occ


@pytest.mark.parametrize("maxiter", [200, 2000])
def test_skilda_rum_ger_oandlig_resistans(maxiter):
    occ = _rum(dorr=False)
    n, karta, _u, _v, _k, L = LF.bygg_gitterlaplacian(occ, distance_transform_edt(~occ), 1.0)
    nk, etikett = LF.nollrum_komponenter(L)
    s, t = int(karta[5, 5]), int(karta[5, 18])
    assert nk == 2 and etikett[s] != etikett[t]
    _x, r_eff, info = LF.los_potential(L, n, s, t, maxiter=maxiter)
    assert r_eff == float("inf")
    assert info == LF.INFO_SKILDA_KOMPONENTER
    assert LF.r_eff_status(r_eff, info) == "INFEASIBEL"


def test_rutten_genom_skilda_rum_ar_infeasibel():
    ut = LF.rutt_genom_flode(_rum(dorr=False), 1.0, (5, 5), (5, 18), 1.0)
    assert ut["bana"] == []
    assert ut["r_eff"] == float("inf")
    assert ut["r_eff_status"] == "INFEASIBEL"


def test_en_dorr_ger_andlig_konvergerad_resistans():
    occ = _rum(dorr=True)
    n, karta, _u, _v, _k, L = LF.bygg_gitterlaplacian(occ, distance_transform_edt(~occ), 1.0)
    _x, r_eff, info = LF.los_potential(L, n, int(karta[5, 5]), int(karta[5, 18]))
    assert info == 0 and LF.r_eff_status(r_eff, info) == "OK"
    assert r_eff == pytest.approx(0.6098488455676194, rel=1e-9)


def test_ej_konvergerad_cg_ar_osaker():
    occ = np.zeros((40, 40), dtype=bool)
    n, karta, _u, _v, _k, L = LF.bygg_gitterlaplacian(occ, distance_transform_edt(~occ), 1.0)
    _x, r_eff, info = LF.los_potential(L, n, int(karta[1, 1]), int(karta[38, 38]), maxiter=2)
    assert info > 0
    assert np.isnan(r_eff)
    assert LF.r_eff_status(r_eff, info) == "OSÄKER"


# --- field_to_recipe_v1: a flat set determines no cylinder ----------------------------------------
def _plan_remsa(offset, lutning, pitch=0.5):
    """A marching-cubes strip of a (tilted) plane, from a float32 field as the reverse pass uses."""
    from skimage import measure
    n = np.array([0.0, np.sin(lutning), np.cos(lutning)])
    g = np.stack(np.meshgrid(*(np.arange(0, 40) * pitch,) * 3, indexing="ij"), -1)
    g = g + np.array([0.0, 0.0, offset - 10.0])
    f = (g @ n - (offset * np.cos(lutning) + 0.123)).astype(np.float32)
    V, T, _n, _v = measure.marching_cubes(f, level=0.0, spacing=(pitch,) * 3)
    V = V.astype(np.float64) + np.array([0.0, 0.0, offset - 10.0])
    normal, area, cent = F._tri_geometri(V, T)
    y = cent @ np.array([0.0, np.cos(lutning), -np.sin(lutning)])
    return normal, area, cent, np.where(np.abs(y - np.median(y)) < 1.0)[0]


@pytest.mark.parametrize("offset,lutning", [(0.0, 0.0), (37.3, 0.0), (37.3, 0.61)])
def test_plan_remsa_far_ingen_cylinder(offset, lutning):
    """Before the guard these read r = 5.89 mm, 20.0 mm and 258 km (inward), all inside the band."""
    normal, area, cent, idx = _plan_remsa(offset, lutning)
    assert F._cyl_passning(normal, area, cent, idx)[3] == float("inf")
    assert F._cyl_passning_trimmad(normal, area, cent, idx, F.PLAN_TOL_PITCH * 0.5) is None


def test_verklig_kvartsborrning_passas_fortfarande():
    from skimage import measure
    pitch = 0.5
    g = np.stack(np.meshgrid(*(np.arange(0, 40) * pitch - 10.0,) * 3, indexing="ij"), -1)
    f = (6.0 - np.hypot(g[..., 0], g[..., 1])).astype(np.float32)
    V, T, _n, _v = measure.marching_cubes(f, level=0.0, spacing=(pitch,) * 3)
    normal, area, cent = F._tri_geometri(V.astype(np.float64) - 10.0, T)
    idx = np.where((cent[:, 0] > 0) & (cent[:, 1] > 0))[0]
    pas = F._cyl_passning_trimmad(normal, area, cent, idx, F.PLAN_TOL_PITCH * pitch)
    assert pas is not None
    assert pas[2] == pytest.approx(6.0, abs=0.01) and pas[4] == -1


# --- sketch_gcs_v1: rank from the Jacobian, locked -----------------------------------------------
def _rektangel(skala, fast_horn=True, extra=()):
    sk = SketchGCS()
    c, s = np.cos(0.3), np.sin(0.3)
    P = []
    for i, (x, y) in enumerate([(0, 0), (200, 0), (200, 120), (0, 120)]):
        P.append(sk.add_point(skala + c * x - s * y + (0.7 if i else 0.0),
                              skala + s * x + c * y - (0.4 if i else 0.0), fixed=(fast_horn and i == 0)))
    L = [sk.add_line(P[i], P[(i + 1) % 4]) for i in range(4)]
    sk.distance_p2p(P[0], P[1], 200.0)
    sk.distance_p2p(P[1], P[2], 120.0)
    sk.distance_p2p(P[2], P[3], 200.0)
    sk.distance_p2p(P[3], P[0], 120.0)
    sk.perpendicular(L[0], L[1])
    sk.parallel(L[0], L[2])
    sk.parallel(L[1], L[3])
    for e in extra:
        e(sk, P, L)
    return sk


@pytest.mark.parametrize("skala", [0.0, 1.0e4])
def test_skiss_med_fri_rotation_ar_underbestamd_trots_m_storre_an_n(skala):
    r = _rektangel(skala).solve()
    assert r.n_eqs > r.n_unknowns
    assert r.status == "UNDERBESTAMD" and r.dof == 1
    assert r.diagnosis["free_directions_named"]


def test_motsagande_skiss_ar_konflikt():
    diag = float(np.hypot(200.0, 120.0)) + 0.5
    r = _rektangel(0.0, extra=[lambda sk, P, L: sk.horizontal(L[0]),
                                lambda sk, P, L: sk.distance_p2p(P[0], P[2], diag)]).solve()
    assert r.status == "OVERBESTAMD_KONFLIKT"


# --- review additions: the guards must not depend on where the origin is, nor drop real routes ----
def _plan_genom_origo(offset, lutning, gir, pitch=0.5):
    """A marching-cubes strip of a plane that passes `offset` mm from the coordinate origin."""
    from skimage import measure
    n = np.array([np.sin(gir) * np.sin(lutning), np.cos(gir) * np.sin(lutning), np.cos(lutning)])
    lo = np.array([-10.0, -10.0, -10.0])
    g = lo + np.stack(np.meshgrid(*(np.arange(0, 41) * pitch,) * 3, indexing="ij"), -1)
    V, T, _n, _v = measure.marching_cubes((g @ n - offset).astype(np.float32), level=0.0,
                                          spacing=(pitch,) * 3)
    normal, area, cent = F._tri_geometri(V.astype(np.float64) + lo, T)
    e = np.cross(n, [1.0, 0.0, 0.0])
    y = cent @ (e / np.linalg.norm(e))
    return normal, area, cent, np.where(np.abs(y - np.median(y)) < 1.0)[0]


@pytest.mark.parametrize("offset,lutning,gir", [(0.0, 1e-3, 0.0), (0.0, 0.05, 0.0), (0.01, 0.01, 0.4)])
def test_plan_genom_origo_far_ingen_cylinder(offset, lutning, gir):
    """A column-normalised [u, v, 1] rank accepted these (r = 0.645 mm, 0.646 mm, 2.95e6 mm)."""
    normal, area, cent, idx = _plan_genom_origo(offset, lutning, gir)
    assert F._cyl_passning(normal, area, cent, idx)[3] == float("inf")
    assert F._cyl_passning_trimmad(normal, area, cent, idx, F.PLAN_TOL_PITCH * 0.5) is None


def test_verklig_cylinder_langt_fran_origo_passas():
    """A real r = 300 mm arc (80 mm chord) 1 km from the origin; the uncentred rank refused it."""
    from skimage import measure
    R, korda, pitch = 300.0, 80.0, 1.0
    sag = R - np.sqrt(R * R - (korda / 2) ** 2)
    lo = np.array([-korda / 2, -sag - 2 * pitch, 0.0])
    shape = (int(np.ceil(korda / pitch)) + 1, int(np.ceil((sag + 3 * pitch) / pitch)) + 3, 12)
    g = lo + np.stack(np.meshgrid(*(np.arange(k) for k in shape), indexing="ij"), -1) * pitch
    f = (R - np.hypot(g[..., 0], g[..., 1] + R)).astype(np.float32)
    V, T, _n, _v = measure.marching_cubes(f, level=0.0, spacing=(pitch,) * 3)
    V = V.astype(np.float64) + lo + np.array([1.0e6, 7.0e5, 5.0])
    normal, area, cent = F._tri_geometri(V, T)
    idx = np.where(np.abs(normal[:, 2]) < 0.3)[0]
    pas = F._cyl_passning_trimmad(normal, area, cent, idx, F.PLAN_TOL_PITCH * pitch)
    assert pas is not None and pas[2] == pytest.approx(R, abs=0.01)


def test_explicit_nolla_ar_ingen_kant():
    """A bridge stored as an explicit zero conductance connects nothing: INFEASIBEL, not OSÄKER."""
    from scipy import sparse
    occ = _rum(dorr=False)
    n, karta, _u, _v, _k, L = LF.bygg_gitterlaplacian(occ, distance_transform_edt(~occ), 1.0)
    a, b = int(karta[12, 11]), int(karta[12, 13])
    C = L.tocoo()
    L0 = sparse.coo_matrix((np.r_[C.data, 0.0, 0.0], (np.r_[C.row, a, b], np.r_[C.col, b, a])),
                           shape=(n, n)).tocsr()
    _x, r_eff, info = LF.los_potential(L0, n, int(karta[5, 5]), int(karta[5, 18]))
    assert LF.r_eff_status(r_eff, info) == "INFEASIBEL" and r_eff == float("inf")


@pytest.mark.parametrize("g", [1e-2, 1e-8])
def test_svag_brygga_ar_aldrig_infeasibel(g):
    """One edge of conductance g joins the rooms: a path exists, so never INFEASIBEL."""
    from scipy import sparse
    occ = _rum(dorr=False)
    n, karta, _u, _v, _k, L = LF.bygg_gitterlaplacian(occ, distance_transform_edt(~occ), 1.0)
    a, b = int(karta[12, 11]), int(karta[12, 13])
    B = sparse.coo_matrix(([-g, -g, g, g], ([a, b, a, b], [b, a, a, b])), shape=(n, n))
    _x, r_eff, info = LF.los_potential((L + B).tocsr(), n, int(karta[5, 5]), int(karta[5, 18]))
    assert LF.r_eff_status(r_eff, info) != "INFEASIBEL"
    if LF.r_eff_status(r_eff, info) == "OK":
        assert r_eff == pytest.approx(1.0 / g, rel=0.05)


def test_ej_konvergerad_losning_behaller_rutten():
    """A connected serpentine where CG needs more than maxiter: R_eff is withheld (OSÄKER), but the
    route through the corridor is real and is still returned."""
    lanes, langd, w = 12, 300, 3
    occ = np.ones((lanes * (w + 1) + 1, langd + 2), dtype=bool)
    for i in range(lanes):
        r0 = 1 + i * (w + 1)
        occ[r0:r0 + w, 1:langd + 1] = False
        if i < lanes - 1:
            c = langd - (w - 1) if i % 2 == 0 else 1
            occ[r0 + w, c:c + w] = False
    ut = LF.rutt_genom_flode(occ, 1.0, (2, 2), (1 + (lanes - 1) * (w + 1) + 1, 2), 0.5)
    assert ut["r_eff_status"] == "OSÄKER" and np.isnan(ut["r_eff"])
    assert len(ut["bana"]) > 0 and ut["clearance_mm"] >= 1.0


def test_inlopp_lika_med_utlopp_ger_noll():
    """s == t: R_eff is 0 exactly and the one-cell route stands (b[s], b[t] = 1, -1 would leave -e_s)."""
    occ = np.zeros((24, 24), dtype=bool)
    n, karta, _u, _v, _k, L = LF.bygg_gitterlaplacian(occ, distance_transform_edt(~occ), 1.0)
    _x, r_eff, info = LF.los_potential(L, n, int(karta[5, 5]), int(karta[5, 5]))
    assert r_eff == 0.0 and LF.r_eff_status(r_eff, info) == "OK"
    assert len(LF.rutt_genom_flode(occ, 1.0, (5, 5), (5, 5), 0.5)["bana"]) == 1
