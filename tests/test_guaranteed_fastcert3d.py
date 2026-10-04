"""Two-stage verifier, shared affine-family certificates and integer exact kernels."""
from fractions import Fraction as F
from itertools import combinations
import dataclasses
import math
import numpy as np
import pytest
from field_engine.experimental import guaranteed_scalar3d as c
from field_engine.experimental import guaranteed_geometry3d as g
from field_engine.experimental import guaranteed_fastcert3d as fc


def octahedron():
    p = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    return c.convex_star_mesh(p, [(i, j, k) for i in (0, 1) for j in (2, 3) for k in (4, 5)])


def cases():
    box = c.structured_mesh(2, (2, 1, 1)); e = c.box_electrodes(box)
    L = c.structured_mesh(2, l_prism=True); eL = c.box_electrodes(L)
    contrast = [F(10**6) if i % 2 else F(1, 10**6) for i in range(len(box.tets))]
    return [('box_poisson', c.structured_mesh(3), 1, 1, None), ('box_R', box, 0, 1, e),
            ('L_poisson', L, 1, 1, None), ('L_R', L, 0, 1, eL), ('box_R_contrast', box, 0, contrast, e),
            ('octa_poisson', c.bisect_edges(octahedron(), c.longest_edges(octahedron(), range(8))), 1, 1, None)]


@pytest.mark.parametrize('degree', [1, 2])
@pytest.mark.parametrize('case', range(6))
def test_fast_interval_contains_exact_interval_and_is_as_tight(case, degree):
    name, mesh, src, k, e = cases()[case]
    v, z, _ = c.solve_fields(mesh, src, k, e, degree)
    ex = c.certificate(mesh, v, z, src, k, e, degree, with_energy_moments=True)
    fa = fc.certificate_fast(mesh, v, z, src, k, e, degree, with_energy_moments=True)
    assert fa['lower'] <= ex['lower'] <= fa['inner_lower_upper']
    assert fa['inner_upper_lower'] <= ex['upper'] <= fa['upper']
    scale = max(abs(ex['upper']), abs(ex['lower']))
    assert (fa['upper'] - fa['lower']) - (ex['upper'] - ex['lower']) <= 1e-11*scale
    assert all(a >= b for a, b in zip(fa['local_gap_upper'], ex['local_gap_upper']))
    assert sum(fa['local_gap_upper']) <= sum(ex['local_gap_upper'])*(1 + 1e-9) + 1e-11*scale
    assert fa['arithmetic'] == fc.ARITHMETIC_FAST and fa['mesh_sha256'] == mesh.sha256
    G, Q, L = fc._moment_intervals(fa)
    for i in range(3):
        for j in range(3):
            assert G[i][j][0] <= F(ex['energy_moments']['primal'][i][j]) <= G[i][j][1]
            assert Q[i][j][0] <= F(ex['energy_moments']['dual'][i][j]) <= Q[i][j][1]
    assert L[0] <= F(ex['energy_moments']['load_exact']) <= L[1]


@pytest.mark.parametrize('shift_exp,scale_exp', [(30, 0), (0, -100), (0, 150), (40, -20), (300, -200)])
def test_enclosure_survives_cancellation_and_extreme_scales(shift_exp, scale_exp):
    base = c.structured_mesh(2, l_prism=True); s = F(2)**scale_exp; t = F(2)**shift_exp
    A = [[s, s/3, 0], [0, s, 0], [0, 0, s*F(5, 4)]]; det = s**3*F(5, 4)
    mesh = c.affine_mesh(base, A, (t, -t, t/7))
    for src, e in ((1, None), (0, c.box_electrodes(base))):
        # Fields are proposals: transport them from the untransformed mesh.
        v, z0, _ = c.solve_fields(base, src, 1, e, 2); z = [x*(det if src else 1) for x in z0]
        ex = c.certificate(mesh, v, z, src, 1, e, 2)
        fa, ok, stage = fc.certify_decision(mesh, v, z, lambda lo, hi: True, src, 1, e, 2)
        assert ok and fa['lower'] <= ex['lower'] and ex['upper'] <= fa['upper']
        assert fa['upper'] - fa['lower'] <= (ex['upper'] - ex['lower'])*(1 + 1e-6) + 1e-9*abs(ex['upper'])
        if scale_exp > -150:
            assert stage == 'BINARY64'
        else:                       # volumes ~2**-600 underflow in binary64: stage two must take over
            assert stage == 'EXACT_FRACTION'
            with pytest.raises(fc.UndecidedEnclosure):
                fc.certificate_fast(mesh, v, z, src, 1, e, 2)


def test_iv_arithmetic_encloses_exact_rational_expressions():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(200)*10.0**rng.integers(-30, 30, 200); b = rng.standard_normal(200); d = np.abs(rng.standard_normal(200)) + .1
    A, Bv, D = fc.IV(a), fc.IV(b), fc.IV(d)
    expr = ((A*Bv - Bv.sq()) + A/D - (A + Bv)*(A - Bv)).sum(0)
    exact = sum((F(x)*F(y) - F(y)**2) + F(x)/F(z) - (F(x) + F(y))*(F(x) - F(y)) for x, y, z in zip(a, b, d))
    lo, hi = expr.exact_bounds()
    assert lo <= exact <= hi
    assert hi - lo <= 1e-12*sum(abs(F(x))**2 for x in a)
    with pytest.raises(ArithmeticError):
        fc.IV(np.array([1.0]), np.array([2.0])).recip()


def test_exact_admissibility_is_unchanged_in_stage_one():
    mesh = c.structured_mesh(2); v, z, _ = c.solve_fields(mesh, potential_degree=2)
    bad = list(z); bad[0] += F(1, 2**60)
    with pytest.raises(ValueError, match='balance'):
        fc.certificate_fast(mesh, v, bad, potential_degree=2)
    _, bc, _, _ = c.potential_layout(mesh, 2); w = v.copy(); w[next(i for i in bc if i >= len(mesh.points))] = 2.0**-1074
    with pytest.raises(ValueError, match='Dirichlet'):
        fc.certificate_fast(mesh, w, z, potential_degree=2)
    box = c.structured_mesh(2); e = c.box_electrodes(box); v, z, _ = c.solve_fields(box, 0, electrodes=e)
    with pytest.raises(ValueError, match='unit right'):
        fc.certificate_fast(box, v, [q*F(3, 4) for q in z], 0, electrodes=e)
    bad = list(z); bad[e['wall'][0]] = F(1, 100)
    with pytest.raises(ValueError, match='insulating'):
        fc.certificate_fast(box, v, bad, 0, electrodes=e)
    w = v.copy(); w[box.faces[e['right'][0]][0]] = 1 - 2.0**-53
    with pytest.raises(ValueError, match='0/1'):
        fc.certificate_fast(box, w, z, 0, electrodes=e)
    with pytest.raises(ValueError):
        fc.certificate_fast(dataclasses.replace(box), v, z, 0, electrodes=e)
    w = v.copy(); w[-1] = np.nan
    with pytest.raises(ValueError):
        fc.certificate_fast(box, w, z, 0, electrodes=e)
    with pytest.raises(ValueError):
        fc.certificate_fast(box, v, z, 0, [1]*(len(box.tets) - 1), e)


def test_two_stage_decision_uses_exact_only_when_undecided():
    mesh = c.structured_mesh(3); v, z, _ = c.solve_fields(mesh, potential_degree=2)
    ex = c.certificate(mesh, v, z, potential_degree=2)
    cert, ok, stage = fc.certify_decision(mesh, v, z, lambda lo, hi: (hi - lo)/lo <= .5, potential_degree=2)
    assert ok and stage == 'BINARY64'
    cert, ok, stage = fc.certify_decision(mesh, v, z, lambda lo, hi: (hi - lo)/lo <= 1e-6, potential_degree=2)
    assert not ok and stage == 'BINARY64'
    cert, ok, stage = fc.certify_decision(mesh, v, z, lambda lo, hi: lo >= ex['lower'], potential_degree=2)
    assert ok and stage == 'EXACT_FRACTION' and cert['lower'] == ex['lower']
    box = c.structured_mesh(2, (2, 1, 1)); e = c.box_electrodes(box); v, z, _ = c.solve_fields(box, 0, 1, e, 1)
    cert, ok, stage = fc.certify_decision(box, v, z, lambda lo, hi: hi <= 2.0000000000000004, 0, 1, e, 1)
    assert ok and stage == 'EXACT_FRACTION'


def _transport(mesh, A, src, z):
    A_, det, _ = fc._mat(A)
    return c.affine_mesh(mesh, A_), tuple(c.rat(x)*(det if src else 1) for x in z)


@pytest.mark.parametrize('A', [[[F(5, 4), F(1, 3), 0], [0, F(7, 8), F(1, 5)], [F(1, 7), 0, F(3, 2)]],
                               [[F(3, 4), 0, 0], [0, F(5, 4), 0], [0, 0, F(1, 2)]],
                               [[1, F(1, 64), 0], [0, 1, 0], [0, 0, 1]]])
@pytest.mark.parametrize('kind', ['poisson', 'resistance'])
def test_affine_variant_bounds_equal_direct_certificate_of_transported_fields(A, kind):
    mesh = c.structured_mesh(2, l_prism=True); src, e = (1, None) if kind == 'poisson' else (0, c.box_electrodes(mesh))
    v, z, _ = c.solve_fields(mesh, src, 1, e, 2)
    ex = c.certificate(mesh, v, z, src, 1, e, 2, with_energy_moments=True)
    img, zt = _transport(mesh, A, src, z)
    direct = c.certificate(img, v, zt, src, 1, e, 2)
    fam = fc.affine_variant_bounds(ex, A)
    if kind == 'resistance':
        assert (fam['lower'], fam['upper']) == (direct['lower'], direct['upper'])
    else:   # Poisson: same flux bound; the potential is additionally rescaled optimally (F^2/E >= 2F-E)
        assert fam['upper'] == direct['upper'] and fam['lower'] >= direct['lower']
    fam_fast = fc.affine_variant_bounds(fc.certificate_fast(mesh, v, z, src, 1, e, 2, with_energy_moments=True), A)
    assert fam_fast['lower'] <= fam['lower'] and fam['upper'] <= fam_fast['upper']
    fresh_v, fresh_z, _ = c.solve_fields(img, src, 1, e, 2)
    fresh = fc.certificate_fast(img, fresh_v, fresh_z, src, 1, e, 2)
    assert max(fresh['lower'], fam['lower']) <= min(fresh['upper'], fam['upper'])
    ok, d2, f2 = fc.direct_transport_check(mesh, v, z, ex, A, src, 1, e, 2)
    assert ok


def test_shear_point_bounds_agree_with_committed_shear_family_endpoint():
    mesh = c.structured_mesh(2, (2, 1, 1)); e = c.box_electrodes(mesh); B = F(1, 64)
    v, z, _ = c.solve_fields(mesh, 0, 1, e, 2); ex = c.certificate(mesh, v, z, 0, 1, e, 2, with_energy_moments=True)
    fam = g.field_shear_family_response(ex, g.verified_shear_family(mesh, B))
    pts = [fc.affine_variant_bounds(ex, [[1, b, 0], [0, 1, 0], [0, 0, 1]]) for b in (-B, -B/2, 0, B/3, B)]
    assert all(fam['lower'] <= p['lower'] and p['upper'] <= fam['upper'] for p in pts)
    assert min(p['lower'] for p in pts) == fam['lower'] and max(p['upper'] for p in pts) == fam['upper']


def test_stretch_family_with_facit_ellipsoid_and_box_certificate():
    ref = c.bisect_edges(octahedron(), c.longest_edges(octahedron(), range(8)))
    ref = c.bisect_edges(ref, c.longest_edges(ref, range(len(ref.tets))))
    enc = g.homothety_to_ellipsoid(ref, (1, 1, 1)); v, z, _ = c.solve_fields(ref, 1, 1, None, 2)
    base = fc.certificate_fast(ref, v, z, 1, 1, None, 2, with_energy_moments=True)
    for axes in [(1, F(3, 4), F(1, 2)), (F(1, 2), F(1, 2), F(1, 2)), (F(7, 8), 1, F(5, 8))]:
        fam = fc.affine_variant_bounds(base, [[axes[0], 0, 0], [0, axes[1], 0], [0, 0, axes[2]]])
        a, b, cc = axes; core = F(8, 15)*a*b*cc/(2*(1/a**2 + 1/b**2 + 1/cc**2))
        lo, hi = F(fam['lower'])/enc.rmax**5, F(fam['upper'])/enc.rmin**5
        assert lo <= core*F(314159265358979, 10**14) and core*F(314159265358980, 10**14) <= hi
    box = [(F(3, 4), F(1)), (F(1, 2), F(7, 8)), (F(5, 8), F(5, 4))]
    whole = fc.stretch_box_bounds(base, box)
    rng = np.random.default_rng(3)
    for _ in range(20):
        s = [F(int(rng.integers(0, 64)), 64)*(hi - lo) + lo for lo, hi in box]
        p = fc.affine_variant_bounds(base, [[s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]]])
        assert whole['lower'] <= p['lower'] and p['upper'] <= whole['upper']
    with pytest.raises(ValueError):
        fc.stretch_box_bounds(base, [(F(1), F(1, 2)), (1, 1), (1, 1)])


def test_family_inputs_are_typed_and_bound():
    mesh = c.structured_mesh(2); v, z, _ = c.solve_fields(mesh)
    plain = fc.certificate_fast(mesh, v, z)
    with pytest.raises(ValueError, match='moments'):
        fc.affine_variant_bounds(plain, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mom = fc.certificate_fast(mesh, v, z, with_energy_moments=True)
    with pytest.raises(ValueError, match='determinant'):
        fc.affine_variant_bounds(mom, [[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    var = fc.certificate_fast(mesh, v, z, 1, [1 + (i % 2) for i in range(len(mesh.tets))], with_energy_moments=True)
    with pytest.raises(ValueError, match='constant-coefficient'):
        fc.affine_variant_bounds(var, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    bad = dict(mom); bad['energy_moments'] = dict(mom['energy_moments'], arithmetic='CALLER')
    with pytest.raises(ValueError, match='unknown moment'):
        fc.affine_variant_bounds(bad, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_certificate_library_admission_and_exact_keys():
    mesh = c.structured_mesh(2, (2, 1, 1)); e = c.box_electrodes(mesh); v, z, _ = c.solve_fields(mesh, 0, 1, e, 2)
    cert = fc.certificate_fast(mesh, v, z, 0, 1, e, 2, with_energy_moments=True)
    lib = fc.CertificateLibrary(); box = [(F(7, 8), F(9, 8))]*3
    key = lib.add(mesh, cert, e, 1, 0, 'diag_stretch', box)
    k2, ent = lib.lookup(mesh, cert, e, 1, 0, 'diag_stretch', (1, F(9, 8), F(7, 8)))
    assert k2 == key and ent is not None
    assert lib.lookup(mesh, cert, e, 1, 0, 'diag_stretch', (1, F(9, 8) + F(1, 2**60), 1)) == (None, None)
    assert lib.lookup(mesh, cert, e, 2, 0, 'diag_stretch', (1, 1, 1)) == (None, None)
    other = c.structured_mesh(2, (2, 1, 1))
    assert lib.lookup(c.affine_mesh(other, [[1, 0, 0], [0, 1, 0], [0, 0, F(3, 2)]]), cert, e, 1, 0, 'diag_stretch', (1, 1, 1)) == (None, None)
    dz = lambda s: fc.affine_design([[s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]]], diagonal=True)
    for s in [(F(7, 8), 1, 1), (1, F(9, 8), 1)]:
        lib.record_check(key, s, True, dz(s))
    lib.record_check(key, (F(7, 8), 1, 1), True, dz((F(7, 8), 1, 1)))          # same variant twice is one green
    lib.record_needed(key)
    assert not lib.admitted(key)
    lib.record_check(key, (1, 1, F(9, 8)), True, dz((1, 1, F(9, 8))))
    assert lib.admitted(key)
    lib.record_check(key, (1, 1, F(7, 8)), False)
    assert not lib.admitted(key) and lib.lookup(mesh, cert, e, 1, 0, 'diag_stretch', (1, 1, 1)) == (None, None)
    with pytest.raises(ValueError):
        lib.add(c.structured_mesh(3), cert, e, 1, 0, 'diag_stretch', box)


# --------------------------------------------- integer exact kernels (identity with Fraction code)
def _reference_bisect(mesh, edges):
    p, ts = list(mesh.points), list(mesh.tets)
    for a, b in sorted(set(tuple(sorted(e)) for e in edges)):
        incident = [i for i, t in enumerate(ts) if a in t and b in t]
        if not incident:
            continue
        mid = tuple((x + y)/2 for x, y in zip(p[a], p[b])); m = len(p); p.append(mid)
        sel = set(incident); new = [t for i, t in enumerate(ts) if i not in sel]
        for i in incident:
            t = ts[i]; new.append(tuple(m if v == a else v for v in t)); new.append(tuple(m if v == b else v for v in t))
        ts = new
    return p, ts


def _reference_repair(mesh, proposed, f, electrodes):
    z = [c.rat(x) for x in proposed]
    if electrodes is None:
        root_face = next(i for i, inc in enumerate(mesh.incidences) if len(inc) == 1)
    else:
        for i in electrodes['wall']:
            z[i] = F(0)
        z[electrodes['right'][0]] += 1 - sum(z[i] for i in electrodes['right']); root_face = electrodes['left'][0]
    root = mesh.incidences[root_face][0][0]; adj = [[] for _ in mesh.tets]
    for fi, inc in enumerate(mesh.incidences):
        if len(inc) == 2:
            a, b = inc[0][0], inc[1][0]; adj[a].append((b, fi)); adj[b].append((a, fi))
    parent, order = {root: (None, root_face)}, [root]
    for t in order:
        for ch, face in adj[t]:
            if ch not in parent:
                parent[ch] = (t, face); order.append(ch)
    defect = [-f*V - sum(s*z[fi] for fi, s in row) for V, row in zip(mesh.volumes, mesh.tet_faces)]
    for t in reversed(order):
        pt, face = parent[t]; sign = next(s for ti, _, s in mesh.incidences[face] if ti == t)
        ch = defect[t]/sign; z[face] += ch
        if pt is not None:
            defect[pt] -= next(s for ti, _, s in mesh.incidences[face] if ti == pt)*ch
    return tuple(z)


def _same(m1, m2):
    return all(getattr(m1, k) == getattr(m2, k) for k in ('points', 'tets', 'volumes', 'faces', 'incidences', 'tet_faces', 'boundary_nodes', 'sha256'))


def test_integer_construction_bisection_and_repair_equal_fraction_reference():
    L = c.structured_mesh(2, l_prism=True)
    X, D = c._integer_coordinates(L.points)
    assert D > 0 and all(F(x, D) == y for p, q in zip(X, L.points) for x, y in zip(p, q))
    edges = c.longest_edges(L, range(0, len(L.tets), 3))
    M = c.bisect_edges(L, edges); p, ts = _reference_bisect(L, edges)
    assert list(M.tets) == [c.oriented(M.points, t) for t in ts] and list(M.points) == p
    odd = c.affine_mesh(M, [[F(5, 4), F(1, 3), 0], [0, F(7, 8), 0], [F(1, 7), 0, F(3, 2)]], (F(1, 3), 0, 2))
    assert _same(odd, c._construct_fraction(odd.points, odd.tets, odd.provenance))
    assert _same(M, c._construct_fraction(M.points, M.tets, M.provenance))
    rng = np.random.default_rng(5)
    for mesh, f, e in ((odd, 1, None), (odd, 0, c.box_electrodes(M)), (M, F(3, 7), None)):
        prop = rng.standard_normal(len(mesh.faces))*1e-2
        assert c.repair_flux(mesh, prop, f, e) == _reference_repair(mesh, prop, c.rat(f), e)
        assert c.repair_flux(mesh, [F(x).limit_denominator(1000) for x in prop], f, e) == _reference_repair(mesh, [F(x).limit_denominator(1000) for x in prop], c.rat(f), e)
    with pytest.raises(ValueError, match='zero volume'):
        c._construct([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [(0, 1, 2, 3)], 'x')
    with pytest.raises(ValueError, match='same side'):
        c._construct([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (F(1, 4), F(1, 4), F(1, 4))], [(0, 1, 2, 3), (0, 1, 2, 4)], 'x')
    with pytest.raises(ValueError, match='non-finite'):
        c.repair_flux(M, np.full(len(M.faces), np.inf))


def test_forged_certificate_dict_is_caught_by_the_independent_admission_check():
    mesh = c.structured_mesh(2, l_prism=True); e = c.box_electrodes(mesh); v, z, _ = c.solve_fields(mesh, 0, 1, e, 2)
    cert = fc.certificate_fast(mesh, v, z, 0, 1, e, 2, with_energy_moments=True)
    A = [[F(9, 8), 0, 0], [0, F(7, 8), 0], [0, 0, 1]]
    ok, direct, fam = fc.direct_transport_check(mesh, v, z, cert, A, 0, 1, e, 2)
    assert ok and fam['lower'] <= direct['upper'] and direct['lower'] <= fam['upper']
    forged = dict(cert); mom = dict(cert['energy_moments'])
    mom['dual_mid'] = [[x*.5 for x in row] for row in mom['dual_mid']]; forged['energy_moments'] = mom
    with pytest.raises(ArithmeticError, match='ordered'):          # gross forgery fails closed
        fc.direct_transport_check(mesh, v, z, forged, A, 0, 1, e, 2)
    mom = dict(mom); mom['dual_mid'] = [[x*.995*2 for x in row] for row in mom['dual_mid']]; forged['energy_moments'] = mom
    ok, direct, fam = fc.direct_transport_check(mesh, v, z, forged, A, 0, 1, e, 2)
    assert not ok and fam['upper'] < direct['upper']                # subtle forgery is a red check
    lib = fc.CertificateLibrary(); key = lib.add(mesh, forged, e, 1, 0, 'diag_stretch', [(F(7, 8), F(9, 8))]*3)
    for s in [(1, 1, F(9, 8)), (F(9, 8), 1, 1), (1, F(7, 8), 1)]:
        lib.record_check(key, s, True)
    lib.record_needed(key); lib.record_check(key, (F(9, 8), F(7, 8), 1), ok)
    assert not lib.admitted(key)


@pytest.mark.parametrize('layout', ['series', 'parallel'])
def test_material_family_bounds_match_direct_certificate_and_exact_layered_resistance(layout):
    mesh = c.structured_mesh(2, (2, 1, 1)); e = c.box_electrodes(mesh)
    cent = [tuple(sum(mesh.points[i][d] for i in t)/4 for d in range(3)) for t in mesh.tets]
    labels = [('a' if (x < 1 if layout == 'series' else y < F(1, 2)) else 'b') for x, y, z in cent]
    v, z, _ = c.solve_fields(mesh, 0, 1, e, 1)
    base = fc.certificate_fast(mesh, v, z, 0, 1, e, 1, region_labels=labels)
    for kb in (F(1, 1000), F(1, 3), F(1), F(7), F(10**6)):
        fam = fc.material_variant_bounds(base, {'a': 1, 'b': kb})
        coeff = [kb if lab == 'b' else F(1) for lab in labels]
        direct = c.certificate(mesh, v, z, 0, coeff, e, 1)
        assert fam['lower'] <= direct['lower'] and direct['upper'] <= fam['upper']
        assert fam['upper'] - fam['lower'] <= (direct['upper'] - direct['lower'])*(1 + 1e-9) + 1e-12*direct['upper']
        exact = 1 + 1/kb if layout == 'series' else 4/(1 + kb)
        assert F(fam['lower']) <= exact <= F(fam['upper'])
    A = [[F(5, 4), 0, 0], [0, 1, F(1, 8)], [0, 0, F(3, 4)]]
    img = c.affine_mesh(mesh, A); fam = fc.material_variant_bounds(base, {'a': 1, 'b': 3}, A)
    direct = c.certificate(img, v, z, 0, [3 if lab == 'b' else 1 for lab in labels], e, 1)
    assert fam['lower'] <= direct['lower'] and direct['upper'] <= fam['upper']
    with pytest.raises(ValueError, match='exactly the certified regions'):
        fc.material_variant_bounds(base, {'a': 1})
    with pytest.raises(ValueError, match='region moments'):
        fc.material_variant_bounds(fc.certificate_fast(mesh, v, z, 0, 1, e, 1), {'a': 1, 'b': 1})


def test_poisson_family_rescales_the_transported_potential():
    ref = c.bisect_edges(octahedron(), c.longest_edges(octahedron(), range(8)))
    v, z, _ = c.solve_fields(ref, 1, 1, None, 2)
    ex = c.certificate(ref, v, z, 1, 1, None, 2, with_energy_moments=True)
    s = F(3, 2); fam = fc.affine_variant_bounds(ex, [[s, 0, 0], [0, s, 0], [0, 0, s]])
    # u_s(y) = s^2 u(y/s): uniform scaling must keep the relative width (up to endpoint rounding)
    E = F(ex['energy_moments']['primal'][0][0]) + F(ex['energy_moments']['primal'][1][1]) + F(ex['energy_moments']['primal'][2][2])
    L = F(ex['energy_moments']['load_exact'])
    assert F(fam['lower']) <= s**5*L*L/E <= F(fam['lower'])*(1 + F(1, 2**50))
    assert abs(fam['relative_width_upper'] - (ex['upper'] - L*L/E)/(L*L/E)) < 1e-12
    img, zt = _transport(ref, [[s, 0, 0], [0, s, 0], [0, 0, s]], 1, z)
    direct = c.certificate(img, v*float(s*s), zt, 1, 1, None, 2)
    assert F(fam['lower']) >= F(direct['lower'])*(1 - F(1, 2**50)) and fam['upper'] == direct['upper']


def test_material_family_poisson_direct_check_and_rescaling():
    mesh = c.structured_mesh(2); labels = ['a' if sum(mesh.points[i][0] for i in t) < 2 else 'b' for t in mesh.tets]
    v, z, _ = c.solve_fields(mesh, 1, 1, None, 2)
    base = fc.certificate_fast(mesh, v, z, 1, 1, None, 2, region_labels=labels)
    ok, direct, fam = fc.direct_material_check(mesh, v, z, base, {'a': 1, 'b': 4}, labels, 1, None, 2)
    assert ok
    scaled = fc.material_variant_bounds(base, {'a': 1, 'b': 4})
    assert scaled['lower'] >= fam['lower'] and scaled['upper'] == fam['upper']
    with pytest.raises(ValueError, match='labels differ'):
        fc.direct_material_check(mesh, v, z, base, {'a': 1, 'b': 4}, labels[::-1], 1, None, 2)


# ------------------------------------------------------------- independent review additions
def test_iv_radius_covers_each_rounding_and_each_reduction():
    # one product rounds away 2**-60 and the difference cancels to 0: only the per-operation radius covers it
    x = fc.IV(np.array([1 + 2.0**-30])); y = fc.IV(np.array([1 + 2.0**-29]))
    lo, hi = (x.sq() - y).sum(0).exact_bounds()
    assert lo <= F(2)**-60 <= hi
    # exact summands whose float sum loses 2**-60: only the reduction term covers it
    lo, hi = fc.IV(np.array([1.0, 2.0**-60, -1.0])).sum(0).exact_bounds()
    assert lo <= F(2)**-60 <= hi
    # a non-dyadic rational input carries its conversion radius
    iv, _ = fc._from_rationals([F(1, 3)])
    lo, hi = fc.IV(iv.m[0], iv.r[0]).exact_bounds()
    assert lo <= F(1, 3) <= hi and lo < hi


@pytest.mark.parametrize('scale_exp', [300, 400])
def test_stage_one_range_failures_fall_back_to_the_exact_certificate(scale_exp):
    # unit-current fluxes on a mesh scaled by 2**300 underflow in binary64 (q ~ 2**-600); volumes at 2**400 overflow
    base = c.structured_mesh(2, l_prism=True); e = c.box_electrodes(base); s = F(2)**scale_exp
    mesh = c.affine_mesh(base, [[s, 0, 0], [0, s, 0], [0, 0, s]])
    v, z, _ = c.solve_fields(base, 0, 1, e, 2)
    ex = c.certificate(mesh, v, z, 0, 1, e, 2)
    cert, ok, stage = fc.certify_decision(mesh, v, z, lambda lo, hi: True, 0, 1, e, 2)
    assert ok and cert['lower'] <= ex['lower'] and ex['upper'] <= cert['upper']


def _swap_yz(cert):
    import copy
    forged = copy.deepcopy(cert); m = forged['energy_moments']; P = [0, 2, 1]
    for name in ('primal_mid', 'primal_rad', 'dual_mid', 'dual_rad'):
        M = m[name]; m[name] = [[M[P[i]][P[j]] for j in range(3)] for i in range(3)]
    return forged


def test_library_does_not_admit_checks_that_cannot_identify_the_moments():
    mesh = c.structured_mesh(2, l_prism=True); e = c.box_electrodes(mesh); v, z, _ = c.solve_fields(mesh, 0, 1, e, 2)
    cert = fc.certificate_fast(mesh, v, z, 0, 1, e, 2, with_energy_moments=True)
    forged = _swap_yz(cert); box = [(F(3, 4), F(5, 4))]*3
    D = lambda s: [[s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]]]
    lib = fc.CertificateLibrary(); key = lib.add(mesh, forged, e, 1, 0, 'diag_stretch', box)
    for cc in (F(7, 8), F(9, 8), F(5, 4)):            # isotropic checks cannot see the axis swap
        ok, _, _ = fc.direct_transport_check(mesh, v, z, forged, D((cc, cc, cc)), 0, 1, e, 2)
        assert ok
        lib.record_check(key, (cc, cc, cc), ok, fc.affine_design(D((cc, cc, cc)), diagonal=True))
    lib.record_needed(key)
    assert not lib.admitted(key)
    aniso = (1, F(5, 4), F(3, 4))
    assert not lib.admitted(key, fc.affine_design(D(aniso), diagonal=True))
    assert lib.admitted(key, fc.affine_design(D((1, 1, 1)), diagonal=True))   # in the span: pinned by the checks
    fam, honest = fc.affine_variant_bounds(forged, D(aniso)), fc.affine_variant_bounds(cert, D(aniso))
    assert fam['upper'] < honest['lower']            # the forged answer would exclude the truth
    ok, _, _ = fc.direct_transport_check(mesh, v, z, forged, D(aniso), 0, 1, e, 2)
    assert not ok                                    # an identifying check is red
    lib2 = fc.CertificateLibrary(); k2 = lib2.add(mesh, cert, e, 1, 0, 'diag_stretch', box)
    for s in [(F(7, 8), 1, F(5, 4)), (1, F(9, 8), F(3, 4)), (F(5, 4), F(3, 4), 1)]:
        ok, _, _ = fc.direct_transport_check(mesh, v, z, cert, D(s), 0, 1, e, 2)
        lib2.record_check(k2, s, ok, fc.affine_design(D(s), diagonal=True))
    lib2.record_needed(k2)
    assert lib2.admitted(k2) and lib2.admitted(k2, fc.affine_design(D(aniso), diagonal=True))


def test_library_red_closure_is_durable_and_keys_are_not_replaced():
    mesh = c.structured_mesh(2, (2, 1, 1)); e = c.box_electrodes(mesh); v, z, _ = c.solve_fields(mesh, 0, 1, e, 2)
    cert = fc.certificate_fast(mesh, v, z, 0, 1, e, 2, with_energy_moments=True); box = [(F(7, 8), F(9, 8))]*3
    lib = fc.CertificateLibrary(); key = lib.add(mesh, cert, e, 1, 0, 'diag_stretch', box)
    lib.record_check(key, (1, 1, 1), False)
    with pytest.raises(ValueError, match='never replaced'):
        lib.add(mesh, _swap_yz(cert), e, 1, 0, 'diag_stretch', box)
    assert lib.entries[key]['red'] == 1 and lib.entries[key]['cert'] is cert


def test_poisson_direct_check_compares_load_and_energy_not_only_the_lower_bound():
    mesh = c.structured_mesh(2); v, z, _ = c.solve_fields(mesh, 1, 1, None, 2)
    cert = fc.certificate_fast(mesh, v, z, 1, 1, None, 2, with_energy_moments=True)
    A = [[F(9, 8), 0, 0], [0, 1, 0], [0, 0, F(7, 8)]]
    ok, _, _ = fc.direct_transport_check(mesh, v, z, cert, A, 1, 1, None, 2)
    assert ok
    # shift load by d and the energy by 2 d det A at this map: 2 det L - E(A) is unchanged at A
    import copy
    forged = copy.deepcopy(cert); m = forged['energy_moments']; det = F(63, 64); M00 = 1/F(81, 64)
    d = F(m['load_mid'])/8; m['load_mid'] = float(F(m['load_mid']) + d)
    m['primal_mid'][0][0] = float(F(m['primal_mid'][0][0]) + 2*d/M00)
    fam_true, fam_forged = (fc.affine_variant_bounds(x, A, rescale_potential=False) for x in (cert, forged))
    assert abs(fam_forged['lower'] - fam_true['lower']) <= 1e-12*abs(fam_true['lower'])
    ok, _, _ = fc.direct_transport_check(mesh, v, z, forged, A, 1, 1, None, 2)
    assert not ok


def test_reciprocal_radius_denominator_cannot_silently_overflow():
    # Exact dyadic perturbation of a large finite binary64 input. The old
    # reciprocal lost its input-radius term and its upper bound missed 1/x.
    from fractions import Fraction as F
    import math
    import pytest
    from field_engine.experimental.guaranteed_fastcert3d import _from_rationals, UndecidedEnclosure
    # Use the actual failing binary64 value, explicitly and reproducibly.
    m=1.623850303300515e200
    value=F(m)-F(499,1000)*F(math.ulp(m))
    iv,_=_from_rationals((value,))
    with pytest.raises(UndecidedEnclosure,match='denominator'):
        iv[0].recip()
