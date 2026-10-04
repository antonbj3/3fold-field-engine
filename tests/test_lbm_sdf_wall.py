"""Regression for direction-correct Bouzidi reflection and FACIT Poiseuille walls."""
import sys,os
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/field_engine'))
import lbm_domare_v1 as LB


def plane(theta,n=8):
    z=np.arange(n+2);return np.minimum(z-(1-theta),n+theta-z)[None,None,:].astype(float)


def steady(sd,tau,tol=1e-9):
    solid=sd<=0;sh=sd.shape;a=np.zeros(sh+(3,));a[...,0]=np.where(solid,0,1e-6)
    wall=LB.forbered_sdf_vagg(sd,solid)
    f=LB.feq3d(np.ones(sh),np.zeros(sh+(3,)));last=None
    for s in range(15000):
        f=LB.lbm_step(f,solid,tau,a,[],sdf_vagg=wall)
        if s%100==0:
            rho=f.sum(-1);u=(f@LB.E+.5*rho[...,None]*a)/rho[...,None]
            if last is not None and np.max(abs(u[~solid]-last))<tol*np.max(abs(u[~solid])):break
            last=u[~solid].copy()
    else:raise AssertionError('steady state not reached')
    rho=f.sum(-1);u=(f@LB.E+.5*rho[...,None]*a)/rho[...,None]
    return u,rho,wall


@pytest.mark.parametrize('theta,tau',[
    pytest.param(t,tau,marks=pytest.mark.xfail(strict=True,raises=AssertionError,reason='PREREG H4: coarse forced BGK wall slip exceeds 0.08 lu'))
    if t==.25 and tau==1.2 else (t,tau)
    for t in [.25,.5,.75] for tau in [.6,.8,1.2]
])
def test_facit_plane_wall_at_declared_sdf_position(tau,theta):
    sd=plane(theta);u,rho,wall=steady(sd,tau)
    fit=np.polyfit(np.arange(1,9),u[0,0,1:-1,0],2);roots=np.sort(np.roots(fit).real)
    assert abs(roots[0]-(1-theta))<.08
    assert abs(roots[1]-(8+theta))<.08
    assert -2*fit[0]/(1e-6/((tau-.5)/3))==pytest.approx(1,abs=1e-6)
    assert abs(rho[sd>0].mean()-1)<1e-6
    assert wall['stats']['fallback_links']==0


def test_low_theta_uses_outgoing_population_in_both_cells():
    sd=plane(.25);solid=sd<=0;sh=sd.shape
    wall=LB.forbered_sdf_vagg(sd,solid)
    f=LB.feq3d(np.ones(sh),np.zeros(sh+(3,)))
    # Nonequilibrium perturbation makes incoming/outgoing values different at the back node.
    f[0,0,2,6]+=.002;f[0,0,2,5]-=.002
    a=np.zeros(sh+(3,));tau=.8
    rho=f.sum(-1);u=(f@LB.E)/rho[...,None];u[solid]=0
    post=f-(f-LB.feq3d(rho,u))/tau
    after=LB.lbm_step(f,solid,tau,a,[],sdf_vagg=wall)
    # At z=1 the wall is toward -z, q=6, incoming opposite q=5.
    expected=.5*post[0,0,1,6]+.5*post[0,0,2,6]
    wrong=.5*post[0,0,1,6]+.5*post[0,0,2,5]
    assert abs(expected-wrong)>1e-6
    assert after[0,0,1,5]==pytest.approx(expected,abs=1e-15)


def test_halfway_reflection_independent_of_solid_reservoir():
    sd=plane(.5);solid=sd<=0;sh=sd.shape
    wall=LB.forbered_sdf_vagg(sd,solid);a=np.zeros(sh+(3,))
    f=LB.feq3d(np.ones(sh),np.zeros(sh+(3,)));f2=f.copy();f2[solid]*=1.1
    o1=LB.lbm_step(f,solid,.8,a,[],sdf_vagg=wall)
    o2=LB.lbm_step(f2,solid,.8,a,[],sdf_vagg=wall)
    assert np.array_equal(o1[~solid],o2[~solid])


def test_empty_wall_matches_legacy_streaming_bitwise():
    sd=np.ones((2,3,4));solid=sd<=0;a=np.zeros(sd.shape+(3,));a[...,0]=1e-6
    f=LB.feq3d(np.ones(sd.shape),np.zeros(sd.shape+(3,)))
    wall=LB.forbered_sdf_vagg(sd,solid)
    assert np.array_equal(LB.lbm_step(f,solid,.8,a,[]),LB.lbm_step(f,solid,.8,a,[],sdf_vagg=wall))


def test_stale_mask_and_backend_are_refused(monkeypatch):
    sd=plane(.25);solid=sd<=0;wall=LB.forbered_sdf_vagg(sd,solid);f=LB.feq3d(np.ones(sd.shape),np.zeros(sd.shape+(3,)));a=np.zeros(sd.shape+(3,))
    changed=solid.copy();changed[...,3]=True
    with pytest.raises(ValueError,match='stale'):LB.lbm_step(f,changed,.8,a,[],sdf_vagg=wall)
    monkeypatch.setenv('FIELD_ENGINE_LBM_BACKEND','native')
    with pytest.raises(ValueError,match='numpy'):LB.run_lbm(sd.shape,solid,.8,a,[],1,sdf_vagg=wall)


def test_thin_gap_declares_halfway_fallback():
    sd=np.array([-1.,.25,-1.])[None,None,:]
    wall=LB.forbered_sdf_vagg(sd,sd<=0)
    assert wall['stats']['fallback_links']>0
    u,rho,_=steady(sd,.8)
    assert np.isfinite(u).all() and np.isfinite(rho).all()


def test_sdf_wall_sign_mismatch_refused():
    sd=plane(.5)
    with pytest.raises(ValueError,match='disagree'):LB.forbered_sdf_vagg(sd,np.zeros(sd.shape,bool))


def test_reflection_on_fortran_order_populations():
    sd=plane(.25);solid=sd<=0;wall=LB.forbered_sdf_vagg(sd,solid)
    f=LB.feq3d(np.ones(sd.shape),np.zeros(sd.shape+(3,)))
    a=np.zeros(sd.shape+(3,));a[...,0]=np.where(solid,0,1e-6)
    f_c=LB.lbm_step(f,solid,.8,a,[],sdf_vagg=wall)
    f_f=LB.lbm_step(np.asfortranarray(f),solid,.8,a,[],sdf_vagg=wall)
    assert np.allclose(f_c,f_f,rtol=0,atol=2e-16)
