"""Continuum identities, adversarial premises, material transfer and reserve."""
from fractions import Fraction as F
from dataclasses import FrozenInstanceError
import pytest
from field_engine.experimental import guaranteed_elasticity3d as g


def fields():
    q = g.polynomial({(2,0,0):1,(3,0,0):-2,(4,0,0):1})
    qy = g.polynomial({(0,2,0):1,(0,3,0):-2,(0,4,0):1})
    qz = g.polynomial({(0,0,2):1,(0,0,3):-2,(0,0,4):1})
    psi=g.product(g.product(q,qy),qz)
    v=(g.derivative(psi,1),g.scale(g.derivative(psi,0),-1),())
    e=g.strain(v)
    s=tuple(tuple(g.scale(e[i][j],2) for j in range(3)) for i in range(3))
    f=tuple(g.scale(p,-1) for p in g.divergence(s))
    return v,s,f


@pytest.fixture(scope='module')
def data():
    v,s,f=fields()
    return v,s,f,g.ElasticCertificate(v,s,f)


def test_external_compatible_continuum_identity(data):
    v,s,f,c=data
    nu=F(49,100); E=2*(1+nu)  # mu=1, divergence exactly zero
    result=c.exact(E,nu)
    true=2*g.norm_squared(g.strain(v))
    assert F(result['lower'])<=true<=F(result['upper'])
    assert result['error_squared_upper']==0
    assert F(result['potential_lower'])<=-true/2<=F(result['potential_upper'])


@pytest.mark.parametrize('nu',['0.3','0.45','0.49','0.4999','0.499999'])
def test_fast_encloses_exact_and_remains_robust(data,nu):
    c=data[-1];nu=F(nu);E=2*(1+nu)
    fast=c.fast(E,nu);exact=c.exact(E,nu)
    assert fast['lower']<=exact['lower']<=exact['upper']<=fast['upper']
    assert fast['error_squared_upper']<1e-20
    assert c.decision(E,nu)['decision']=='ACCEPT'


def test_nonzero_divergence_has_physical_incompressible_cost(data):
    v,s,f,_=data
    b=g.polynomial({(1,1,1):1,(2,1,1):-1,(1,2,1):-1,(1,1,2):-1,
                    (2,2,1):1,(2,1,2):1,(1,2,2):1,(2,2,2):-1})
    perturbed=(g.add(v[0],g.scale(b,F(1,1000))),v[1],v[2])
    c=g.ElasticCertificate(perturbed,s,f)
    a=c.exact(3,F(49,100))['error_squared_upper']
    z=c.exact(3,F(499999,1000000))['error_squared_upper']
    assert z>1000*a  # no mislabeled uniform robustness


def test_symmetry_refuses_even_tiny_defect(data):
    v,s,f,_=data;s=[list(row) for row in s]
    s[0][1]=g.add(s[0][1],g.polynomial({(0,0,0):F(1,10**60)}))
    with pytest.raises(ValueError,match='symmetric'):g.ElasticCertificate(v,s,f)


def test_equilibrium_refuses_even_tiny_residual(data):
    v,s,f,_=data;f=list(f)
    f[0]=g.add(f[0],g.polynomial({(0,0,0):F(1,10**60)}))
    with pytest.raises(ValueError,match='equilibrated'):g.ElasticCertificate(v,s,f)


def test_trace_refuses_even_tiny_nonzero_value(data):
    v,s,f,_=data;v=list(v)
    v[2]=g.polynomial({(0,0,0):F(1,10**60)})
    with pytest.raises(ValueError,match='trace'):g.ElasticCertificate(v,s,f)


def test_mixed_stress_repair_is_exact_and_symmetric(data):
    v,_,f,_=data
    pressure=g.polynomial({(1,0,1):F(1,7)})
    s=g.reconstruct_stress(f,v,pressure,F(3,10))
    assert all(s[i][j]==s[j][i] for i in range(3) for j in range(3))
    assert all(not g.add(g.divergence(s)[i],f[i]) for i in range(3))
    g.ElasticCertificate(v,s,f)


def test_continuous_box_and_reach_contain_interior_solution(data):
    v,s,f,_=data;mu0,_=g.moduli(F(91,100),F(47,100))
    c=g.ElasticCertificate(tuple(g.scale(p,1/mu0) for p in v),s,f)
    box=c.material_box(tuple(map(F,('0.819','1.001'))),tuple(map(F,('0.45','0.49'))),tuple(map(F,('0.91','0.47'))))
    M=g.norm_squared(g.strain(v))
    for E,nu in [('0.819','0.45'),('1.001','0.49'),('0.89123','0.46789')]:
        mu,_=g.moduli(F(E),F(nu));true=2*M/mu
        assert F(box['lower'])<=true<=F(box['upper'])
    assert box['continuous_box'] and box['instance_statistic']['exact_zero_divergence']


@pytest.mark.parametrize('E,nu',[(0,F('.49')),(-1,F('.3')),(1,F('.5')),(1,F(-1))])
def test_invalid_material(data,E,nu):
    with pytest.raises(ValueError):data[-1].fast(E,nu)


def test_overflow_reserve_keeps_a_finite_true_energy(data):
    # Every reported endpoint is finite: displacement/stress rescaling keeps
    # the true certificate finite while both moduli overflow in binary64.
    v,s,f,c=data
    # Zero fields avoid a genuinely unrepresentable nonzero energy.
    z=g.ElasticCertificate(((),(),()),(((),(),()),)*3,((),(),()))
    r=z.fast(F(10**400),F(49,100))
    assert r['arithmetic']=='EXACT_FRACTION' and r['upper']==0


def test_exact_reserve_for_width_boundary(data):
    r=data[-1].decision(F(298,100),F(49,100),relative_width=0)
    assert r['fallback'] and r['arithmetic']=='EXACT_FRACTION' and r['decision']=='ACCEPT'


def test_mutation_does_not_change_internal_fields(data):
    v,s,f,_=data;mutable=[dict(p) for p in v]
    c=g.ElasticCertificate(mutable,s,f);old=c.exact(3,F('.49'))
    mutable[0][(0,0,0)]=1
    assert c.exact(3,F('.49'))==old
    with pytest.raises(FrozenInstanceError):c._moments=(0,)*5


@pytest.mark.parametrize('kwargs',[{'domain':'kirsch_hole','observable':'max_stress'},
    {'domain':'boussinesq_halfspace'},{'domain':'hertz','contact':True},
    {'domain':'cook','boundary':'mixed'},{'nu':F('.5')}])
def test_unsupported_requests_refused(kwargs):
    assert g.admit_request(**kwargs)['status']=='OSAKER'


def test_bad_partition_or_polynomial_rejected():
    with pytest.raises(ValueError):g.polynomial({(-1,0,0):1})
    with pytest.raises(ValueError):g.polynomial({(0,0,0):float('nan')})
    with pytest.raises(ValueError):g.ElasticCertificate(((),),(),((),))


def test_invalid_boxes_refused(data):
    for E,nu in [(('1','0.5'),('.45','.49')),(('.8','1'),('.45','.5'))]:
        with pytest.raises(ValueError):data[-1].material_box(tuple(map(F,E)),tuple(map(F,nu)))


def test_volumetric_manufactured_solution_matches_independent_energy():
    # v=(x(1-x)y(1-y)z(1-z),0,0). Each squared partial derivative
    # integrates to 1/2700; 2mu||eps||^2+lambda(div v)^2=(4mu+lambda)/2700.
    factors=[g.polynomial({tuple(i if j==axis else 0 for j in range(3)):co
                          for i,co in [(1,1),(2,-1)]}) for axis in range(3)]
    b=g.product(g.product(factors[0],factors[1]),factors[2])
    v=(b,(),()); e=g.strain(v); divergence=g.trace(e)
    for nu in (F(3,10),F(49,100),F(499999,1000000)):
        mu=F(1);lam=2*mu*nu/(1-2*nu);E=2*mu*(1+nu)
        s=tuple(tuple(g.add(g.scale(e[i][j],2*mu),
                           g.scale(divergence,lam) if i==j else ())
                      for j in range(3)) for i in range(3))
        f=tuple(g.scale(a,-1) for a in g.divergence(s))
        c=g.ElasticCertificate(v,s,f);truth=(4*mu+lam)/2700
        assert c._moments[1]>0 and c._moments[3]>0
        for answer in (c.exact(E,nu),c.fast(E,nu)):
            assert F(answer['lower'])<=truth<=F(answer['upper'])
        assert c.exact(E,nu)['error_squared_upper']==0


def test_fixed_E_box_encloses_extreme_nu_without_reach(data):
    v,s,f,c=data
    box=c.material_box((F(1),F(1)),(F(3,10),F(49,100)))
    # Direct monomial integration of the specified curl field.
    C0=F(22,3472875)
    for nu in (F(3,10),F(49,100)):
        truth=2*(1+nu)*C0
        assert F(box['lower'])<=truth<=F(box['upper'])


def test_nonfinite_material_queries_are_rejected(data):
    for bad in (float('nan'),float('inf'),-float('inf')):
        for E,nu in ((bad,F(49,100)),(1,bad)):
            with pytest.raises(ValueError):data[-1].fast(E,nu)
