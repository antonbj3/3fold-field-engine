"""Moment projection, force ledger and explicit infeasible-wall regression."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/field_engine'))
import lbm_domare_v1 as LB

A=np.vstack((np.ones(19),LB.E.T))

def w(sd,project=True):
    return LB.forbered_sdf_vagg(sd,sd<=0,positivitetsbevarande=True,impulsbevarande=project)

@pytest.mark.parametrize('seed',range(60))
def test_projection_matches_independent_constrained_oracle(seed):
    rng=np.random.default_rng(seed);anchor=rng.uniform(.01,1,19);anchor/=anchor.sum()
    v=rng.normal(0,.5,19);v-=A.T@np.linalg.solve(A@A.T,A@v);h=anchor+v
    p=LB.projektera_populationer(h)
    oracle=minimize(lambda x:.5*np.sum((x-h)**2),anchor,jac=lambda x:x-h,
                    bounds=[(0,None)]*19,constraints={'type':'eq','fun':lambda x:A@(x-h),'jac':lambda x:A},
                    method='SLSQP',options={'ftol':1e-13,'maxiter':200})
    assert oracle.success,oracle.message
    assert np.all(p>=0)
    np.testing.assert_allclose(A@p,A@h,atol=2e-13,rtol=0)
    assert abs(np.sum((p-h)**2)-np.sum((oracle.x-h)**2))<1e-9

@pytest.mark.parametrize('vertex',range(19))
def test_hull_vertices_and_kinetic_nullspace(vertex):
    anchor=np.zeros(19);anchor[vertex]=1
    v=np.zeros(19);v[0]=-2;v[1]=v[2]=1
    h=anchor+v
    p=LB.projektera_populationer(h)
    assert np.min(p)>=0
    np.testing.assert_allclose(A@p,A@h,rtol=0,atol=2e-13)

@pytest.mark.parametrize('j',[[1.01,0,0],[.8,.8,.8],[-1.01,0,0],[0,0,-1.01]])
def test_outside_hull_is_flagged(j):
    b=np.r_[1.,j];h=A.T@np.linalg.solve(A@A.T,b)
    with pytest.raises(LB.PopulationProjectionError) as e:LB.projektera_populationer(h)
    assert e.value.reason=='infeasible'


def test_zero_mass_and_invalid_dtype():
    v=np.zeros(19);v[0]=-2;v[1]=v[2]=1
    assert np.array_equal(LB.projektera_populationer(v),np.zeros(19))
    with pytest.raises(ValueError,match='float64'):LB.projektera_populationer(v.astype(np.float32))


def test_active_wall_keeps_target_moments_and_force():
    z=np.arange(10);sd=np.minimum(z-.75,8.25-z)[None,None,:].astype(float);s=sd<=0
    rho=np.ones(sd.shape);rho[0,0,2]=100;f=LB.feq3d(rho,np.zeros(sd.shape+(3,)));a=np.zeros(sd.shape+(3,))
    d={};dp={};dh={}
    H=LB.lbm_step(f,s,1,a,[],sdf_vagg=LB.forbered_sdf_vagg(sd,s),diagnostics=dh,wall_force=True)
    p=LB.lbm_step(f,s,1,a,[],sdf_vagg=w(sd),diagnostics=d,wall_force=True)
    positive=LB.lbm_step(f,s,1,a,[],sdf_vagg=w(sd,False),diagnostics=dp,wall_force=True)
    assert H[~s].min()<0 and p[~s].min()>=0 and d['wall_projected_nodes']==1
    np.testing.assert_allclose(p[~s].sum(-1),H[~s].sum(-1),atol=2e-13)
    np.testing.assert_allclose(p[~s]@LB.E,H[~s]@LB.E,atol=2e-13)
    np.testing.assert_allclose(d['wall_force_last'],dh['wall_force_last'],atol=2e-13)
    assert np.linalg.norm(np.array(dp['wall_force_last'])-dh['wall_force_last'])>7
    assert np.linalg.norm(np.array(d['wall_link_force_last'])-d['wall_force_last'])>3
    np.testing.assert_allclose(d['wall_force_last'],np.array(d['wall_link_force_last'])+d['wall_redistribution_force_last'],atol=2e-13)


def test_admissible_input_can_have_infeasible_wall_target_without_mutation():
    z=np.arange(6);sd=np.minimum(z-.25,4.75-z)[None,None,:].astype(float);s=sd<=0
    f=LB.feq3d(np.ones(sd.shape)*1e-5,np.zeros(sd.shape+(3,)))
    f[0,0,1]=0;f[0,0,1,5]=100
    before=f.copy()
    with pytest.raises(LB.PopulationProjectionError) as e:
        LB.lbm_step(f,s,.51,np.zeros(sd.shape+(3,)),[],sdf_vagg=w(sd),positivitet=True)
    assert e.value.reason=='infeasible' and e.value.node==1
    assert np.array_equal(f,before)


def test_inactive_steps_are_byteidentical_and_diagnostics_stay_optional():
    sd=np.minimum(np.arange(10)-.75,8.25-np.arange(10))[None,None,:].astype(float);s=sd<=0
    a=np.zeros(sd.shape+(3,));a[...,0]=np.where(s,0,1e-5)
    f=LB.feq3d(np.ones(sd.shape),np.zeros(sd.shape+(3,)));g=f.copy();d={}
    for _ in range(100):
        f=LB.lbm_step(f,s,.8,a,[],sdf_vagg=w(sd,False),positivitet=True)
        g=LB.lbm_step(g,s,.8,a,[],sdf_vagg=w(sd),positivitet=True,diagnostics=d)
        assert f.tobytes()==g.tobytes()
    assert d=={}


def test_flags_and_backend_contract(monkeypatch):
    sd=np.ones((2,2,2));s=sd<=0
    with pytest.raises(ValueError,match='mass and positivity'):LB.forbered_sdf_vagg(sd,s,impulsbevarande=True)
    f=LB.feq3d(np.ones(sd.shape),np.zeros(sd.shape+(3,))).astype(np.float32)
    with pytest.raises(ValueError,match='float64'):LB.lbm_step(f,s,1,np.zeros(sd.shape+(3,)),[],sdf_vagg=w(sd))
    monkeypatch.setenv('FIELD_ENGINE_LBM_BACKEND','native')
    with pytest.raises(ValueError,match='numpy'):LB.run_lbm(sd.shape,s,1,np.zeros(sd.shape+(3,)),[],1,sdf_vagg=w(sd))


def test_reviewer_shock_deterministic_with_force_balance():
    N=20;X=np.indices((N,N,N)).astype(float)
    sd=np.sqrt(((X-np.array([9.7,9.4,10.1])[:,None,None,None])**2).sum(0))-5.3;s=sd<=0
    f0=LB.feq3d(np.where(X[0]<9.7,3.,1.),np.zeros(sd.shape+(3,)));a=np.zeros(sd.shape+(3,));hashes=[]
    for _ in range(2):
        f=f0.copy();d={};wall=w(sd)
        for step in range(300):
            j0=f[~s].sum(0,dtype=np.longdouble)@LB.E
            f=LB.lbm_step(f,s,.51,a,[],sdf_vagg=wall,positivitet=True,diagnostics=d,wall_force=True)
            j1=f[~s].sum(0,dtype=np.longdouble)@LB.E
            assert np.max(np.abs(j1-j0+d['wall_force_last']))<2e-11
            assert f[~s].min()>=0
        assert d['wall_projected_nodes']>0 and d['max_wall_moment_error']<2e-13
        assert abs(f[~s].sum()/f0[~s].sum()-1)<1e-10
        hashes.append(hashlib.sha256(f.tobytes()).hexdigest())
    assert hashes[0]==hashes[1]


def test_force_request_has_explicit_contract_and_run_passthrough():
    s=np.zeros((1,1,3),bool);f=LB.feq3d(np.ones(s.shape),np.zeros(s.shape+(3,)));a=np.zeros(s.shape+(3,))
    with pytest.raises(ValueError,match='prepared SDF wall and diagnostics'):LB.lbm_step(f,s,1,a,[],wall_force=True,diagnostics={})
    sd=np.ones(s.shape);d={}
    v,_=LB.run_lbm(sd.shape,s,1,a,[],2,sdf_vagg=w(sd),diagnostics=d,wall_force=True)
    assert d['wall_force_last']==[0.,0.,0.] and np.array_equal(v,f)


def test_active_fortran_input_and_component_inventory():
    sd=np.minimum(np.arange(10)-.75,8.25-np.arange(10))[None,None,:].astype(float);s=sd<=0
    rho=np.ones(sd.shape);rho[0,0,2]=100
    f=np.asfortranarray(LB.feq3d(rho,np.zeros(sd.shape+(3,))))
    p=LB.lbm_step(f,s,1,np.zeros(sd.shape+(3,)),[],sdf_vagg=w(sd),positivitet=True)
    assert p[~s].min()>=0 and abs(p[~s].sum()/f[~s].sum()-1)<2e-13


def test_finite_input_with_overflowed_moments_is_explicitly_flagged():
    h=np.ones(19)*1e308
    with pytest.raises(LB.PopulationProjectionError) as e:LB.projektera_populationer(h)
    assert e.value.reason=='nonfinite_moments'


# A conservative wall target from a tau=.501, density-ratio-1000 sphere shock (step 3). It lies deep
# inside the moment hull, yet a pseudoinverse Newton step stalls on its rank-deficient support.
SHOCK_TARGET=np.array([-34.513923018171305,0.04390198159097843,0.05555555555555555,0.06215317392463436,
    0.07326836850567955,0.0663362338167238,0.06877087191710865,0.025375343790177352,0.027777777777777776,
    51.86490914824208,0.027777777777777776,0.026961521344161528,0.027777777777777776,51.7731284082505,
    49.49860433090741,0.03303935092208469,0.03397620928231816,0.034399258645588585,0.031961090484260185])

def test_degenerate_support_target_from_shock_converges():
    p=LB.projektera_populationer(SHOCK_TARGET)
    assert np.all(p>=0)
    np.testing.assert_allclose(A@p,A@SHOCK_TARGET,rtol=0,atol=2e-13*SHOCK_TARGET.sum())

@pytest.mark.parametrize('seed',range(4))
def test_wide_feasible_targets_never_report_solver_failure(seed):
    rng=np.random.default_rng(seed)
    for _ in range(300):
        anchor=rng.uniform(0,1,19);anchor/=anchor.sum()
        v=rng.normal(0,10,19);v-=A.T@np.linalg.solve(A@A.T,A@v);h=anchor+v
        p=LB.projektera_populationer(h)
        assert np.all(p>=0)
        np.testing.assert_allclose(A@p,A@h,rtol=0,atol=2e-13)

def test_strong_shock_runs_with_moment_wall():
    N=20;X=np.indices((N,N,N)).astype(float)
    sd=np.sqrt(((X-np.array([9.7,9.4,10.1])[:,None,None,None])**2).sum(0))-5.3;s=sd<=0
    f=LB.feq3d(np.where(X[0]<9.7,1000.,1.),np.zeros(sd.shape+(3,)));a=np.zeros(sd.shape+(3,))
    m0=f[~s].sum();d={};wall=w(sd)
    for _ in range(40):
        f=LB.lbm_step(f,s,.501,a,[],sdf_vagg=wall,positivitet=True,diagnostics=d)
        assert f[~s].min()>=0
    assert d['wall_projected_nodes']>0 and abs(f[~s].sum()/m0-1)<1e-12
