"""Scientific invariants, invalid-premise refusal, and independent integral replay."""
from dataclasses import replace
from fractions import Fraction as F
import unittest
import numpy as np
import pytest
from field_engine.experimental.cut_front import (evaluate,make_slit,assemble,solve_fields,exact_moments,
                        energy_bounds,release_bounds,Moments,rat,region_ids)


class CertificateTests(unittest.TestCase):
    def test_intact_external_exact_and_perturbed_fields(self):
        m,v,p,_,c,_=evaluate(32,F(0))
        self.assertLessEqual(c['lower'],2);self.assertGreaterEqual(c['upper'],2)
        self.assertLessEqual(c['resistance_lower'],.25);self.assertGreaterEqual(c['resistance_upper'],.25)
        rng=np.random.default_rng(20261001)
        for k in range(3):
            vv=v.copy();pp=p.copy()
            fixed=set(m.top)|set(m.bottom);fixedpsi=set(m.insulating_left)|set(m.right)
            for i in range(len(v)):
                if i not in fixed:vv[i]=rng.integers(-5,6)/8
                if i not in fixedpsi:pp[i]=rng.integers(-5,6)/8
            c=energy_bounds(exact_moments(m,vv,pp))
            self.assertLessEqual(c['lower'],2);self.assertGreaterEqual(c['upper'],2)

    def test_volume_and_two_crack_traces(self):
        for theta in (F(0),F(1,10),F(1,2),F(9,10)):
            m,v,p,_,c,_=evaluate(32,2+F(1,8)*theta)
            self.assertEqual(m.a,2+F(1,8)*theta)
            self.assertTrue((m.a,F(0)) in m.xy)
            self.assertEqual(m.area,4)
            pairs={}
            for i,xy in enumerate(m.xy):pairs.setdefault(xy,[]).append(i)
            self.assertTrue(any(len(ids)==2 and abs(v[ids[0]]-v[ids[1]])>.1 for ids in pairs.values()))
            self.assertTrue(m.insulating_left)
            self.assertTrue(all(p[i]==0 for i in m.insulating_left))

    def test_invalid_stream_refused(self):
        m,v,p,*_=evaluate(32,F(17,8))
        p=p.copy();p[m.insulating_left[-1]]=.125
        with self.assertRaises(ValueError):exact_moments(m,v,p)

    def test_forged_mesh_and_moments_refused(self):
        m,v,p,mom,_,_=evaluate(32,F(17,8))
        with self.assertRaises(ValueError):exact_moments(replace(m),v,p)
        with self.assertRaises(ValueError):energy_bounds(Moments(m,mom.primal,mom.dual))

    def test_scopes_and_nested_lengths_checked(self):
        c1=evaluate(32,F(2))[-2];c2=evaluate(32,F(17,8))[-2]
        release_bounds(c1,c2,F(1,8))
        with self.assertRaises(ValueError):release_bounds(c1,c2,F(1,16))
        c3=evaluate(32,F(17,8),(F(1),F(1),F(2)))[-2]
        with self.assertRaises(ValueError):release_bounds(c1,c3,F(1,8))
        c4=evaluate(32,F(17,8),yc=F(1,8))[-2]
        with self.assertRaises(ValueError):release_bounds(c1,c4,F(1,8))

    def test_psd_single_constant_mode_and_aggregation(self):
        for theta in (F(1,10**4),1-F(1,10**4),F(1,10**12),1-F(1,10**12)):
            m=make_slit(32,2+F(1,8)*theta)
            K=assemble(m);evals=np.linalg.eigvalsh(K.toarray())
            self.assertLess(abs(evals[0]),1e-12);self.assertGreater(evals[1],1e-4)
            self.assertLess(np.max(abs(K@np.ones(K.shape[0]))),1e-12)
            v,p,_=solve_fields(m);energy_bounds(exact_moments(m,v,p))

    def test_separate_exact_triangle_replay(self):
        m,v,p,mom,_,_=evaluate(32,F(81,40),(F(1,2),F(1),F(2)))
        r=region_ids(m);pr=[F(0)]*3;du=[F(0)]*3;cross=F(0)
        def derivative(t,values):
            z=[m.xy[i] for i in t];b=[z[1][j]-z[0][j] for j in range(2)];c=[z[2][j]-z[0][j] for j in range(2)]
            det=b[0]*c[1]-b[1]*c[0];d=[rat(values[t[k]])-rat(values[t[0]]) for k in (1,2)]
            return ((d[0]*c[1]-d[1]*b[1])/det,(b[0]*d[1]-c[0]*d[0])/det),det/2
        for t,reg in zip(m.triangles,r):
            gv,ar=derivative(t,v);gp,_=derivative(t,p);q=(-gp[1],gp[0])
            pr[reg]+=ar*sum(x*x for x in gv);du[reg]+=ar*sum(x*x for x in q)
            cross+=ar*sum(x*y for x,y in zip(gv,q))
        self.assertEqual(tuple(pr),mom.primal);self.assertEqual(tuple(du),mom.dual);self.assertEqual(cross,1)

    def test_unsupported_geometry_refused(self):
        for a in (4,-1,float('nan')):
            with self.assertRaises(ValueError):make_slit(32,a)
        with self.assertRaises(ValueError):make_slit(32,2,yc=F(1,31))
        with self.assertRaises(ValueError):make_slit(33,2)
        with self.assertRaises(ValueError):evaluate(32,F(2)+F(1,10**200))


def test_intact_layered_strip_exact_solution_and_displacement_scaling():
    # Three layers are springs in series: S=sum(h_i/mu_i)=19/32.
    # C=L/S=128/19; Pi=Delta²*C/2. Stream psi=x/L is exactly feasible.
    m = make_slit(32, 0)
    shear = (F(1), F(2), F(4))
    compliance = F(19, 32)

    def v_at(y):
        lengths = (min(y+F(1, 2), F(3, 8)),
                   min(max(y+F(1, 8), F(0)), F(1, 4)),
                   max(y-F(1, 8), F(0)))
        return sum(h/k for h, k in zip(lengths, shear))/compliance

    v = tuple(v_at(y) for x, y in m.xy)
    psi = tuple(x/4 for x, y in m.xy)
    moments = exact_moments(m, v, psi)
    for delta in (F(1), F(3, 2)):
        c = energy_bounds(moments, shear, delta)
        expected = delta**2*F(64, 19)
        assert F.from_float(c['lower']) <= expected <= F.from_float(c['upper'])
        assert c['upper']-c['lower'] < 1e-14
        assert c['reaction_lower'] == pytest.approx(float(delta*F(128, 19)), rel=1e-14)
        assert c['resistance_lower'] == pytest.approx(float(F(19, 128)), rel=1e-14)


@pytest.mark.parametrize('a', [F(81, 40), F(2)+F(1, 8*10**8)])
def test_production_assembly_matches_separate_affine_FE_oracle(a):
    m = make_slit(32, a)
    n = max(m.groups)+1
    reference = np.zeros((n, n))
    checked = 0
    for triangle in m.triangles:
        xy = [m.xy[i] for i in triangle]
        # Exact affine interpolation inversion, independent of production grad().
        x0, y0 = xy[0]
        x1, y1 = xy[1]
        x2, y2 = xy[2]
        det = (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
        gradients = ((y1-y2, x2-x1), (y2-y0, x0-x2), (y0-y1, x1-x0))
        grouped = {}
        for i, g in zip(triangle, gradients):
            group = m.groups[i]
            old = grouped.get(group, (F(0), F(0)))
            grouped[group] = (old[0]+g[0]/det, old[1]+g[1]/det)
        for i, gi in grouped.items():
            for j, gj in grouped.items():
                reference[i, j] += float(det/2*(gi[0]*gj[0]+gi[1]*gj[1]))
        checked += 1
    assert checked == len(m.triangles) and checked > 0
    np.testing.assert_allclose(assemble(m).toarray(), reference, atol=1e-10, rtol=1e-12)
    v, psi, cost = solve_fields(m)
    vi, pi, _ = solve_fields(m, independent=True)
    np.testing.assert_allclose(v, vi, atol=1e-10, rtol=0)
    np.testing.assert_allclose(psi, pi, atol=1e-10, rtol=0)
    assert cost['free_dofs'] > 0


def test_release_composes_the_two_energy_budgets_with_correct_units():
    left = evaluate(32, F(2))[-2]
    right = evaluate(32, F(17, 8))[-2]
    result = release_bounds(left, right, F(1, 8))
    expected_hi = (F.from_float(left['upper'])-F.from_float(right['lower']))*8
    expected_lo = max(F(0), (F.from_float(left['lower'])-F.from_float(right['upper']))*8)
    assert F.from_float(result['lower']) <= expected_lo
    assert F.from_float(result['upper']) >= expected_hi
    assert result['lower'] == pytest.approx(float(expected_lo), abs=1e-14)
    assert result['upper'] == pytest.approx(float(expected_hi), rel=1e-14)
    assert result['quantity'] == 'mean_energy_release_over_finite_step'
    assert result['unit'] == 'J/m^2'
    with pytest.raises(ValueError):
        release_bounds(right, left, F(1, 8))


@pytest.mark.parametrize("shear", [(F(1), F(1), F(1)), (F(1), F(2), F(4))])
def test_release_width_uses_material_and_displacement_stationary_reference(shear):
    # S=sum(h_r/mu_r); far-field energy per unit advance is Delta^2/(2*S).
    # This normalizes the interval width; it is not the finite-crack truth.
    bases = [evaluate(32, a) for a in (F(2), F(17, 8))]
    widths = []
    for factor, delta in ((F(1), F(1)), (F(3), F(2))):
        mu = tuple(factor*k for k in shear)
        certificates = [energy_bounds(b[3], mu, delta) for b in bases]
        result = release_bounds(*certificates, F(1, 8))
        compliance = sum(h/k for h, k in zip((F(3, 8), F(1, 4), F(3, 8)), mu))
        reference = delta**2/(2*compliance)
        expected_width = (result['upper']-result['lower'])/float(reference)
        assert result['relative_width_at_Gss'] == pytest.approx(expected_width, rel=1e-14)
        widths.append(result['relative_width_at_Gss'])
    assert widths[0] == pytest.approx(widths[1], rel=1e-13)
    assert F(result['stationary_Gss_exact']) == reference


@pytest.mark.parametrize('gap', [F(1, 80000), F(1, 800000000)])
def test_terminal_ligament_aggregation_conflict_and_unaggregated_certificate(gap):
    a = 4-gap
    with pytest.raises(ValueError, match='aggregation conflicts with Dirichlet data'):
        evaluate(32, a)
    m, v, psi, moments, certificate, cost = evaluate(32, a, aggregate=False)
    assert m.aggregation == 'NONE'
    assert 0 < certificate['lower'] <= certificate['upper'] < 2
