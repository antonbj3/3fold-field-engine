"""The Landreman answer key (arXiv:2609.26742) is exact, and the engine's meters read it as measured.

Part 1 checks the key itself: residuals of div B, Euler and MHD force balance and B.grad psi by
complex-step derivatives, the closed-form volume and second moments against a quadrature of the
field-line map, the surface map, field-line closure, and the paper's own numbers for the sheared
family.  Part 2 runs engine meters on the exact torus with frozen bands from
build/SOL_FALT_FACIT_20261001 (raw/e1e2_landreman.json), so a meter that starts to assume
axisymmetry, or a key that drifts, fails here.
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(1, os.path.join(ROOT, "src", "field_engine"))

from field_engine.facit import landreman as LM          # noqa: E402
from field_engine.facit import kommutering as KO         # noqa: E402
pytest.importorskip("scipy")                              # the engine operators need it
from field_engine.facit import motorkoppling as MK       # noqa: E402

TOL = 1e-12


def _families():
    R = KO.rotation([0.3, 0.2, 1.0], 0.37)
    yield "eps0", LM.IntegerFamily(eps=0.0, delta=0.05)
    yield "eps0.5", LM.IntegerFamily(eps=0.5, delta=1 / 64)
    yield "eps0.8", LM.IntegerFamily(eps=0.8, delta=0.009)
    yield "stretch_c1.7", LM.IntegerFamily(eps=0.4, c=1.7, delta=0.03)
    yield "stretch_rotated", LM.IntegerFamily(a=1.2, b=0.7, c=1.1, R=R, delta=0.03)
    yield "shear_3.27", LM.ShearedFamily(2.0, 1.0, 3.5, 0.1)
    yield "shear_figA", LM.ShearedFamily(1.08, 3.0, 3.5, math.sqrt(0.49))


@pytest.mark.parametrize("name,fam", list(_families()))
def test_equilibrium_residuals_are_roundoff(name, fam):
    P = LM.interior_points(fam, 2000, seed=1)
    r = LM.residuals(fam, P)
    assert r["div"] < TOL * 10
    assert r["euler"] < TOL * 10 * r["max_B"] ** 2
    assert r["mhd"] < TOL * 10 * r["max_B"] ** 2
    assert r["invariance"] < TOL * 10 * r["max_B"] * r["max_grad_p"]


def test_branch_cut_form_matters_near_the_excluded_segment():
    """Regression for the two-form complex sqrt.  Inside the torus of fig. 2B the argument of
    1 + eps / conj(w)^2 stays below 84 degrees; in the second component of {psi <= delta} beside the
    excluded segment it reaches 180 degrees, and the single form sqrt((r + p) / 2) loses ~3e-3 there.
    The formulas hold on the whole smooth domain, so the residuals must be roundoff there too."""
    fam = LM.ShearedFamily(4.0, 3.5, 3.5, math.sqrt(0.49))
    rng = np.random.default_rng(2)
    lo, hi = fam.bbox()
    P = lo + (hi - lo) * rng.random((60000, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        P = P[np.isfinite(fam.psi(P)) & (fam.psi(P) <= fam.delta) & ~fam.inside_domain(P)][:3000]
    assert len(P) == 3000
    r = LM.residuals(fam, P)
    assert r["euler"] < TOL * 10 * r["max_B"] ** 2 and r["div"] < TOL * r["max_B"]


@pytest.mark.parametrize("name,fam", [f for f in _families() if isinstance(f[1], LM.ShearedFamily)])
def test_sheared_inside_domain_is_the_torus(name, fam):
    """{psi <= delta} in the bbox holds 1.6 x V for fig. 2A; restricted to inside_domain it is V
    (scrambled Sobol, 2^18 points), and every point of the surface map lies inside."""
    from scipy.stats import qmc
    rng = np.random.default_rng(4)
    k = fam.kb * np.sqrt(rng.random(4000))
    chi, ze = 2 * np.pi * rng.random(4000), 2 * np.pi * rng.random(4000)
    assert fam.inside_domain(fam.map_XYzeta(-k * np.cos(chi), k * np.sin(chi), ze)).all()
    lo, hi = fam.bbox()
    P = lo + (hi - lo) * qmc.Sobol(3, scramble=True, seed=1).random_base2(18)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.mean((fam.psi(P) <= fam.delta) & fam.inside_domain(P))
    assert abs(frac * np.prod(hi - lo) / fam.volume() - 1) < 0.03


@pytest.mark.parametrize("name,fam", [f for f in _families() if isinstance(f[1], LM.IntegerFamily)])
def test_volume_and_second_moments_closed_form(name, fam):
    """Closed forms against a Gauss x trapezoid quadrature of the map with a numerical Jacobian."""
    g, w = np.polynomial.legendre.leggauss(16)
    rho = 0.5 * math.sqrt(fam.delta) * (g + 1)
    wr = 0.5 * math.sqrt(fam.delta) * w
    al = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ze = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    RH, AL, ZE = np.meshgrid(rho, al, ze, indexing="ij")
    U, V = fam.u_axis + RH * np.cos(AL), RH * np.sin(AL)
    cols = []
    for k in range(3):
        a = [U.astype(complex), V.astype(complex), ZE.astype(complex)]
        a[k] = a[k] + 1e-30j
        cols.append(np.imag(fam.field_line(*a)) / 1e-30)
    det = np.abs(np.linalg.det(np.stack(cols, -1)))
    wgt = det * RH * wr[:, None, None] * (2 * np.pi / 48) * (2 * np.pi / 96)
    P = fam.field_line(U, V, ZE)
    assert abs(wgt.sum() / fam.volume() - 1) < 1e-13
    assert np.abs(det - abs(fam.jacobian_det())).max() < 1e-12
    S = np.einsum("abci,abcj,abc->ij", P, P, wgt)
    assert np.abs(S - fam.second_moment()).max() < 1e-12 * np.abs(fam.second_moment()).max()


@pytest.mark.parametrize("name,fam", list(_families()))
def test_surface_map_lands_on_its_flux_surface(name, fam):
    rng = np.random.default_rng(3)
    ps = fam.delta * rng.random(500)
    a, z = 2 * np.pi * rng.random(500), 2 * np.pi * rng.random(500)
    P = np.stack([fam.surface_points(p, ai, zi) for p, ai, zi in zip(ps, a, z)])
    assert np.abs(fam.psi(P) - ps).max() < 1e-13


def test_field_lines_are_the_map_and_close_after_one_period():
    fam = LM.IntegerFamily(eps=0.5, delta=1 / 64)
    u = fam.u_axis + 0.1 * np.cos(np.linspace(0, 6, 50))
    v = 0.1 * np.sin(np.linspace(0, 6, 50))
    z = np.linspace(0, 2 * np.pi, 50).astype(complex) + 1e-30j
    dr = np.imag(fam.field_line(u.astype(complex), v.astype(complex), z)) / 1e-30
    assert np.abs(dr - fam.B(fam.field_line(u, v, z.real))).max() < 1e-13
    P0 = fam.field_line(u[:8], v[:8], 0 * u[:8])
    h, P = 2 * np.pi / 1000, P0.copy()
    for _ in range(1000):
        k1 = fam.B(P); k2 = fam.B(P + h / 2 * k1); k3 = fam.B(P + h / 2 * k2); k4 = fam.B(P + h * k3)
        P = P + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.linalg.norm(P - P0, axis=1).max() < 1e-7        # iota = 2: RK4 closure is O(dt^4)


def test_sheared_family_reproduces_paper_numbers():
    fam = LM.ShearedFamily(2.0, 1.0, 3.5, 0.1)                   # paper (3.27)
    assert abs(fam.iota_axis() - 2.28690) < 5e-6
    assert abs(fam.iota(fam.delta, n_transits=40, steps_per_transit=400) - 2.2878) < 1e-4
    Vq = fam.volume(n_zeta=256, n_xi=128)
    assert abs(Vq / fam.volume(n_zeta=512, n_xi=256) - 1) < 1e-12


@pytest.mark.parametrize("name,fam", list(_families()) + [("shear_figB_lam2", LM.ShearedFamily(4.0, 3.5, 2.0, 0.7))])
def test_volume_by_divergence_theorem_on_the_surface_map(name, fam):
    """Second route to V that uses only the surface map: V = 1/3 oint x . n dA, tangents by complex
    step, periodic trapezoid (spectral).  For the sheared family this checks the paper's (3.34)
    reduction, including its 1/lam factor, which the self-convergence test above cannot see."""
    n = 64
    al, ze = np.meshgrid(np.linspace(0, 2 * np.pi, n, endpoint=False),
                         np.linspace(0, 2 * np.pi, 2 * n, endpoint=False), indexing="ij")
    P = fam.surface_points(fam.delta, al, ze)
    ta = np.imag(fam.surface_points(fam.delta, al + 1e-30j, ze.astype(complex))) / 1e-30
    tz = np.imag(fam.surface_points(fam.delta, al.astype(complex), ze + 1e-30j)) / 1e-30
    V = abs(np.einsum("abi,abi->", P, np.cross(ta, tz))) * (2 * np.pi / n) * (np.pi / n) / 3
    assert abs(V / fam.volume() - 1) < 1e-11


def test_integer_family_refuses_an_axis_outside_the_label_disk():
    """u_axis = -eps / (2 c^2) must lie inside q < 1/2; otherwise no surface exists and V would be a
    number for an empty body.  eps = 0.5 with vertical compression c = 0.6 gives |u_axis| = 0.69."""
    with pytest.raises(ValueError):
        LM.IntegerFamily(eps=0.5, c=0.6, delta=0.01)
    fam = LM.IntegerFamily(eps=0.3, c=0.8, delta=0.02)
    P = fam.surface_points(fam.delta, np.linspace(0, 6, 7), np.linspace(0, 6, 7))
    assert np.all(np.isfinite(P)) and np.abs(fam.psi(P) - fam.delta).max() < 1e-13


def test_commutation_factory_is_the_paper_stretch_and_rejects_the_rest():
    D = LM.D_INT
    assert len(KO.commutant_basis(D)) == 5                      # GL(2) block + one scalar
    fam = LM.IntegerFamily(eps=0.5)
    B = KO.stretch_field(KO.solovev_B0, np.diag([fam.a, fam.b, 1.0]))
    P = LM.interior_points(fam, 300)
    assert np.abs(B(P) - fam.B(P)).max() < 1e-14
    shear_xz = np.array([[1.0, 0, 0.3], [0, 1, 0], [0, 0, 1]])
    assert not KO.stretch_admissible(shear_xz, D)
    assert KO.stretch_admissible(KO.rotation([0, 1, 1], 0.4) @ np.diag([1.1, 0.9, 1.2]), D)


# --------------------------------------------------------------------------------------------------
# engine meters on the exact torus (scale 50 mm, delta = 1/64, eps = 0.5)
# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def torus():
    return LM.IntegerFamily(eps=0.5, delta=1 / 64, scale=50.0)


def test_engine_volume_meters_on_exact_torus(torus):
    """Frozen from raw/e1e2_landreman.json: marching cubes on psi - delta reads -0.078 (h/t)^2 with
    t = V/A = 2.52 mm, second order and phase-stable; the voxel Riemann sum is unbiased but moves
    with the lattice phase."""
    pytest.importorskip("skimage")
    pytest.importorskip("trimesh")
    import faltvolym_v1 as FV
    Vex = torus.volume()
    errs_mc = []
    for h in (2.0, 1.0):
        g, lo, _ = MK.landreman_lattice(torus, h, phase=(0.37, 0.61, 0.13))
        mc, dropped = FV.marching_cubes_volym(g, h, origin=lo)
        vox = FV.voxel_riemann_volym(g, h)
        assert dropped == 0
        assert abs(vox / Vex - 1) < 0.01
        errs_mc.append(mc / Vex - 1)
    assert -0.06 < errs_mc[0] < -0.04 and -0.015 < errs_mc[1] < -0.010
    assert 1.8 < math.log2(errs_mc[0] / errs_mc[1]) < 2.2


def test_engine_mesh_moments_converge_second_order_on_exact_vertices(torus):
    from field_engine.experimental.shape_massprop import femur_massprops as FM
    e = []
    for n in (16, 32):
        V, F = torus.surface_mesh(n, 4 * n)
        vol, first, S = FM.moments(V, F)
        e.append((vol / torus.volume() - 1, np.abs(S - torus.second_moment()).max() / np.abs(torus.second_moment()).max()))
        assert np.abs(first).max() < 1e-6 * abs(vol) * 50.0      # centroid exactly at the origin
    assert all(v < 0 for v, _ in e)                              # inscribed facets lose volume
    assert 1.9 < math.log2(e[0][0] / e[1][0]) < 2.1
    assert 1.9 < math.log2(e[0][1] / e[1][1]) < 2.1


def test_engine_edt_sdf_surface_band(torus):
    """Mesh -> +Z ray winding -> two EDTs -> half-pitch correction, through the engine's own CPU entry
    point (surface_raster_and_flood, metod raypar_vindning), read trilinearly at 2000 exact surface
    points: median |sd| <= 0.25 h with the correction, the raw transform ~2x that."""
    from scipy import ndimage
    import faltkarna_v1_mesh_to_sdf as M2S
    h = 1.0
    V, F = torus.surface_mesh(96, 384)
    g, lo, _ = MK.landreman_lattice(torus, h)
    rng = np.random.default_rng(0)
    P = torus.surface_points(torus.delta, 2 * np.pi * rng.random(2000), 2 * np.pi * rng.random(2000))
    med = {}
    for corr in (True, False):
        gmin, shp, _, solid, sd = M2S.surface_raster_and_flood(V, F, h, lo, global_shape=g.shape,
                                                               metod="raypar_vindning", ytkorrektion=corr)
        c = ((P - (lo + gmin * h)) / h).T
        med[corr] = np.median(np.abs(ndimage.map_coordinates(sd.astype(np.float64), c, order=1))) / h
    assert med[True] <= 0.25
    assert 1.6 < med[False] / med[True] < 2.4
    win = tuple(slice(gmin[i], gmin[i] + shp[i]) for i in range(3))
    inside = g < 0
    assert np.count_nonzero(inside) == np.count_nonzero(inside[win])     # window holds the whole body
    assert np.count_nonzero(solid != inside[win]) < 0.01 * np.count_nonzero(inside)


def test_interior_points_empty_domain_raises_instead_of_hanging():
    class Empty:
        delta = 1.0
        def bbox(self):
            return np.zeros(3), np.ones(3)
        def psi(self, P):
            return np.zeros(len(P))
        def inside_domain(self, P):
            return np.zeros(len(P), dtype=bool)
    with pytest.raises(RuntimeError):
        LM.interior_points(Empty(), 10, max_rounds=5)
