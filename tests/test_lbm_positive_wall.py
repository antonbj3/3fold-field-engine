"""Conservative positivity corridor: adversarial states and inactive-path identity."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/field_engine'))
import lbm_domare_v1 as LB


def components(solid):
    from test_lbm_mass_wall import component_ids
    return component_ids(solid)


def wall(sd, mode='positive'):
    solid=sd<=0
    if mode=='halfway':sd=np.where(solid,-1.,1.)
    return LB.forbered_sdf_vagg(sd,solid,massbevarande=mode!='bouzidi',
                              positivitetsbevarande=mode=='positive')


def test_large_density_gradient_is_positive_and_conservative():
    z=np.arange(10);sd=np.minimum(z-.75,8.25-z)[None,None,:].astype(float)
    solid=sd<=0;rho=np.ones(sd.shape);rho[0,0,2]=100
    f=LB.feq3d(rho,np.zeros(sd.shape+(3,)));a=np.zeros(sd.shape+(3,));diag={}
    old=LB.lbm_step(f,solid,1,a,[],sdf_vagg=wall(sd,'rest'))
    new=LB.lbm_step(f,solid,1,a,[],sdf_vagg=wall(sd),diagnostics=diag)
    assert old[~solid].min() < -7
    assert new[~solid].min()>=0
    assert abs(new[~solid].sum()/f[~solid].sum()-1)<2e-13
    assert diag['wall_limited_nodes']>0


@pytest.mark.parametrize('seed',range(100))
def test_random_wall_each_component(seed):
    rng=np.random.default_rng(seed);sd=rng.uniform(-1,1,(3,4,5));solid=sd<=0
    rho=10**rng.uniform(-3,3,sd.shape)
    f=rng.lognormal(0,1,sd.shape+(19,))
    f=(f+f[...,LB.OPP])/2  # zero momentum, arbitrary positive nonequilibrium
    f*=rho[...,None]/f.sum(-1)[...,None]
    # tau>1 gives positive f* without invoking the new collision limiter.
    new=LB.lbm_step(f,solid,1.7,np.zeros(sd.shape+(3,)),[],sdf_vagg=wall(sd))
    assert new[~solid].min()>=0
    for ids in components(solid):
        assert abs(new.reshape(-1,19)[ids].sum()/f.reshape(-1,19)[ids].sum()-1)<2e-13


def test_positive_nonequilibrium_collision_keeps_mass_and_momentum():
    rng=np.random.default_rng(770);f=rng.lognormal(0,1.5,(3,4,5,19))
    sd=np.ones(f.shape[:-1]);solid=sd<=0;a=np.zeros(sd.shape+(3,));diag={}
    v=LB.lbm_step(f,solid,.51,a,[],positivitet=True,diagnostics=diag)
    assert v.min()>=0 and diag['collision_limited_nodes']>0
    post=np.empty_like(v)
    for q in range(19):post[...,q]=np.roll(v[...,q],tuple(-LB.Ei[q]),axis=(0,1,2))
    np.testing.assert_allclose(post.sum(-1),f.sum(-1),rtol=2e-13,atol=0)
    np.testing.assert_allclose(post@LB.E,f@LB.E,rtol=2e-13,atol=2e-13*f.sum(-1).max())
    assert abs(v.sum()/f.sum()-1)<2e-13
    np.testing.assert_allclose(v.sum((0,1,2))@LB.E,f.sum((0,1,2))@LB.E,rtol=2e-13,atol=2e-13*f.sum())


def test_active_forced_collision_is_rejected():
    f=np.ones((1,1,1,19));f[...,1]=100;solid=np.zeros((1,1,1),bool)
    a=np.ones((1,1,1,3))*1e-6
    with pytest.raises(ValueError,match='zero local force'):
        LB.lbm_step(f,solid,.51,a,[],positivitet=True)


def test_invalid_positive_mode_input_is_rejected():
    f=np.ones((1,1,1,19));f[...,1]=-1
    with pytest.raises(ValueError,match='finite nonnegative'):
        LB.lbm_step(f,np.zeros((1,1,1),bool),.8,np.zeros((1,1,1,3)),[],positivitet=True)


def test_wall_rejects_negative_collision_and_nonconservative_flag():
    sd=np.ones((2,2,2));s=sd<=0;f=np.ones(sd.shape+(19,));f[...,1]=100
    with pytest.raises(ValueError,match='post-collision'):
        LB.lbm_step(f,s,.51,np.zeros(sd.shape+(3,)),[],sdf_vagg=wall(sd))
    with pytest.raises(ValueError,match='massbevarande'):
        LB.forbered_sdf_vagg(sd,s,massbevarande=False,positivitetsbevarande=True)


def test_isolated_nodes_fallback_and_fortran_layout():
    sd=np.full((5,5,10),-1.);sd[2,2,2]=.1;sd[2,2,7]=.7;s=sd<=0
    f=np.asfortranarray(LB.feq3d(np.ones(sd.shape),np.zeros(sd.shape+(3,))))
    f[2,2,2]*=3;w=wall(sd);v=LB.lbm_step(f,s,.51,np.zeros(sd.shape+(3,)),[],sdf_vagg=w,positivitet=True)
    assert w['stats']['fallback_links']>0 and v[~s].min()>=0
    for ids in components(s):
        assert abs(v.reshape(-1,19)[ids].sum()/f.reshape(-1,19)[ids].sum()-1)<2e-13


def test_open_plane_declares_mass_exchange():
    sd=np.ones((3,3,3));s=sd<=0;f=LB.feq3d(np.ones(sd.shape),np.zeros(sd.shape+(3,)))
    bc=dict(idx=(0,slice(None),slice(None)),adj=(1,slice(None),slice(None)),rho=1.1)
    closed=LB.lbm_step(f,s,.8,np.zeros(sd.shape+(3,)),[],sdf_vagg=wall(sd),positivitet=True)
    opened=LB.lbm_step(f,s,.8,np.zeros(sd.shape+(3,)),[bc],sdf_vagg=wall(sd),positivitet=True)
    assert opened.min()>=0
    assert opened.sum()-f.sum()==pytest.approx((opened[0]-closed[0]).sum(),abs=2e-13)


def test_negative_open_plane_equilibrium_is_rejected():
    f=np.ones((3,1,1,19));f[...,1]=100;s=np.zeros((3,1,1),bool)
    bc=dict(idx=(0,0,0),adj=(1,0,0),rho=1.0)
    with pytest.raises(ValueError,match='boundary equilibrium'):
        LB.lbm_step(f,s,.8,np.zeros(s.shape+(3,)),[bc],positivitet=True)


def test_low_mach_steps_have_identical_bytes_and_no_limiting():
    sd=np.minimum(np.arange(10)-.75,8.25-np.arange(10))[None,None,:].astype(float);s=sd<=0
    a=np.zeros(sd.shape+(3,));a[...,0]=np.where(s,0,1e-5)
    f=LB.feq3d(np.ones(sd.shape),np.zeros(sd.shape+(3,)));g=f.copy();diag={}
    for _ in range(100):
        f=LB.lbm_step(f,s,.8,a,[],sdf_vagg=wall(sd,'rest'))
        g=LB.lbm_step(g,s,.8,a,[],sdf_vagg=wall(sd),positivitet=True,diagnostics=diag)
        assert f.tobytes()==g.tobytes()
    assert diag=={}


def test_positive_collision_backend_guard(monkeypatch):
    monkeypatch.setenv('FIELD_ENGINE_LBM_BACKEND','native')
    with pytest.raises(ValueError,match='numpy'):
        LB.run_lbm((1,1,1),np.zeros((1,1,1),bool),.8,np.zeros((1,1,1,3)),[],1,positivitet=True)


def test_reviewers_tau051_ratio3_shock_positive_conservative_and_deterministic():
    X=np.indices((20,20,20)).astype(float)
    sd=np.sqrt(((X-np.array([9.7,9.4,10.1])[:,None,None,None])**2).sum(0))-5.3
    s=sd<=0;rho=np.where(X[0]<9.7,3.,1.);a=np.zeros(sd.shape+(3,))
    f0=LB.feq3d(rho,np.zeros(sd.shape+(3,)));m0=f0[~s].sum();hashes=[]
    for _ in range(2):
        f=f0.copy();diag={};w=wall(sd)
        for _ in range(300):
            f=LB.lbm_step(f,s,.51,a,[],sdf_vagg=w,positivitet=True,diagnostics=diag)
            assert f[~s].min()>=0
            assert abs(f[~s].sum()/m0-1)<1e-10
        assert diag['wall_limited_nodes']>0 and diag['collision_limited_nodes']>0
        hashes.append(hashlib.sha256(f.tobytes()).hexdigest())
    assert hashes[0]==hashes[1]


@pytest.mark.parametrize('f0scale', [0.0, 1e-30, 1e-12, 1e-4])
def test_tiny_rest_reserve_keeps_wall_limiter_nonnegative(f0scale):
    # The 64-eps margin is relative to f*_0; blend rounding is relative to f*_q. Review counterexample.
    for seed in range(40):
        rng=np.random.default_rng(seed);sd=rng.uniform(-1,1,(4,5,6));s=sd<=0
        f=rng.lognormal(0,1,sd.shape+(19,));f=(f+f[...,LB.OPP])/2
        f[...,0]=f0scale*f[...,1:].sum(-1)
        new=LB.lbm_step(f,s,1e300,np.zeros(sd.shape+(3,)),[],sdf_vagg=wall(sd))
        assert new[~s].min()>=0
        for ids in components(s):
            assert abs(new.reshape(-1,19)[ids].sum()/f.reshape(-1,19)[ids].sum()-1)<2e-13


@pytest.mark.parametrize('case', [('sphere', .51, 30., 40), ('channel', .501, 1000., 60)])
def test_strong_density_ratio_stays_positive_in_floating_point(case):
    # Review counterexamples: rest f_0 ~ -1e-16 after wall limiting (sphere, ratio>=10) and a subnormal
    # collision blend rounding to -5e-324 at |u|~0.9 (channel). Both must stay >=0 and conservative.
    geom,tau,ratio,steps=case
    if geom=='sphere':
        X=np.indices((20,20,20)).astype(float)
        sd=np.sqrt(((X-np.array([9.7,9.4,10.1])[:,None,None,None])**2).sum(0))-5.3;hi=X[0]<9.7
    else:
        X=np.indices((40,3,14)).astype(float);sd=np.minimum(X[2]-.83,12.17-X[2]);hi=X[0]<19.5
    s=sd<=0;a=np.zeros(sd.shape+(3,));f=LB.feq3d(np.where(hi,ratio,1.),np.zeros(sd.shape+(3,)))
    m0=f[~s].sum();w=wall(sd);diag={}
    for _ in range(steps):
        f=LB.lbm_step(f,s,tau,a,[],sdf_vagg=w,positivitet=True,diagnostics=diag)
        assert f[~s].min()>=0
        assert abs(f[~s].sum()/m0-1)<1e-12
    assert diag.get('wall_halfway_nodes',0)+diag.get('collision_frozen_nodes',0)>0
