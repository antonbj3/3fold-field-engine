"""Closed-component conservation and geometric convergence of the production SDF wall.

Long tests intentionally execute at least 40000 production steps. They do not renormalise f.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/field_engine'))
import lbm_domare_v1 as LB


def pipe(R,phase=(.13,.13)):
    n=2*R+4;Y,Z=np.meshgrid(np.arange(n)-(n-1)/2-phase[0],np.arange(n)-(n-1)/2-phase[1],indexing='ij')
    return (R-np.hypot(Y,Z))[None],Y,Z


def initial(sd,g=0):
    solid=sd<=0;a=np.zeros(sd.shape+(3,));a[...,0]=np.where(solid,0,g)
    f=LB.feq3d(np.ones(sd.shape),np.zeros(sd.shape+(3,)))
    return solid,a,f


def velocity(f,a):
    return (f@LB.E)/f.sum(-1)[...,None]+.5*a


def component_ids(solid):
    # D3Q19 adjacency, including periodic wrapping and diagonal directions.
    ids=np.flatnonzero(~solid);parent=np.arange(solid.size)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    grid=np.arange(solid.size).reshape(solid.shape)
    for direction in LB.Ei[1:]:
        neighbor=np.roll(grid,tuple(-direction),axis=(0,1,2)).ravel()
        for i in ids:
            j=neighbor[i]
            if not solid.ravel()[j]:parent[root(j)]=root(i)
    return [ids[np.array([root(i) for i in ids])==r] for r in sorted({root(i) for i in ids})]


@pytest.mark.parametrize('tau',[.6,.8,1.2])
@pytest.mark.parametrize('shape',[(2,5,6),(4,7,8)])
def test_each_disconnected_component_conserves_mass_and_wall_keeps_momentum(tau,shape):
    rng=np.random.default_rng(180);sd=rng.uniform(-1,1,shape)
    # Add two isolated fluid points in a predominantly solid domain.
    sd[0,:,:]=-1;sd[-1,:,:]=-1
    solid,a,f=initial(sd);a[~solid]=rng.uniform(-1e-4,1e-4,(np.count_nonzero(~solid),3))
    f*=rng.uniform(.98,1.02,f.shape)
    wall=LB.forbered_sdf_vagg(sd,solid)
    oldwall=LB.forbered_sdf_vagg(sd,solid,massbevarande=False)
    corrected=LB.lbm_step(f,solid,tau,a,[],sdf_vagg=wall)
    uncorrected=LB.lbm_step(f,solid,tau,a,[],sdf_vagg=oldwall)
    assert np.array_equal(corrected[~solid,1:],uncorrected[~solid,1:])
    assert np.array_equal(corrected[~solid]@LB.E,uncorrected[~solid]@LB.E)
    for ids in component_ids(solid):
        before=f.reshape(-1,19)[ids].sum();after=corrected.reshape(-1,19)[ids].sum()
        assert abs(after/before-1)<2e-13


def test_isolated_components_and_thin_gap_fallback_are_closed_individually():
    sd=np.full((5,5,10),-1.);sd[2,2,2]=.1;sd[2,2,7]=.7
    solid,a,f=initial(sd);f[2,2,2]*=1.03;f[2,2,7]*=.99
    wall=LB.forbered_sdf_vagg(sd,solid)
    after=LB.lbm_step(f,solid,.8,a,[],sdf_vagg=wall)
    assert wall['stats']['fallback_links']>0
    assert len(component_ids(solid))==2
    for ids in component_ids(solid):
        assert after.reshape(-1,19)[ids].sum()==pytest.approx(f.reshape(-1,19)[ids].sum(),rel=2e-13)


def test_defect_equals_outgoing_minus_reflected_at_each_wall_node():
    sd,_,_=pipe(4);solid,a,f=initial(sd,1e-5)
    old=LB.forbered_sdf_vagg(sd,solid,massbevarande=False)
    new=LB.forbered_sdf_vagg(sd,solid)
    boundary_id=next(link[0][0] for link in old['links'] if len(link[0]))
    f.reshape(-1,19)[boundary_id]*=1.01
    rho=f.sum(-1);u=velocity(f,a);u[solid]=0
    cu=u@LB.E.T;ea=a@LB.E.T;ua=(u*a).sum(-1,keepdims=True)
    post=f-(f-LB.feq3d(rho,u))/.8+(1-1/(2*.8))*LB.W*rho[...,None]*(3*(ea-ua)+9*cu*ea)
    uncorrected=LB.lbm_step(f,solid,.8,a,[],sdf_vagg=old)
    corrected=LB.lbm_step(f,solid,.8,a,[],sdf_vagg=new)
    delta=np.zeros(solid.size)
    for q in range(1,19):
        ids=old['links'][q][0]
        delta[ids]+=post.reshape(-1,19)[ids,q]-uncorrected.reshape(-1,19)[ids,LB.OPP[q]]
    np.testing.assert_allclose((corrected-uncorrected)[...,0].ravel(),delta,rtol=0,atol=2e-16)
    assert abs(uncorrected[~solid].sum()-f[~solid].sum())>1e-6
    assert abs(corrected[~solid].sum()/f[~solid].sum()-1)<2e-13


def test_open_density_plane_has_only_declared_mass_exchange():
    sd,_,_=pipe(4);sd=np.repeat(sd,3,axis=0);solid,a,f=initial(sd,1e-5)
    wall=LB.forbered_sdf_vagg(sd,solid)
    bc=dict(idx=(0,slice(None),slice(None)),adj=(1,slice(None),slice(None)),rho=1.003)
    closed=LB.lbm_step(f,solid,.8,a,[],sdf_vagg=wall)
    opened=LB.lbm_step(f,solid,.8,a,[bc],sdf_vagg=wall)
    u=velocity(f,a);u[solid]=0;target=LB.feq3d(np.full(sd.shape[1:],1.003),u[1])
    np.testing.assert_array_equal(opened[0],target)
    exchange=(target[~solid[0]]-closed[0][~solid[0]]).sum()
    assert opened[~solid].sum()-f[~solid].sum()==pytest.approx(exchange,abs=2e-13)


def test_prepared_wall_without_mode_key_defaults_to_conservation():
    sd,_,_=pipe(4);solid,a,f=initial(sd)
    wall=LB.forbered_sdf_vagg(sd,solid);wall.pop('massbevarande')
    idx=next(link[0][0] for link in wall['links'] if len(link[0]))
    f.reshape(-1,19)[idx]*=1.01
    new=LB.lbm_step(f,solid,.8,a,[],sdf_vagg=wall)
    assert abs(new[~solid].sum()/f[~solid].sum()-1)<2e-13


def test_empty_wall_with_nonequilibrium_and_spatial_force_is_bitwise_legacy():
    sd=np.ones((3,4,5));solid,a,f=initial(sd);rng=np.random.default_rng(2026)
    a[:]=rng.uniform(-1e-4,1e-4,a.shape);f*=rng.uniform(.99,1.01,f.shape)
    wall=LB.forbered_sdf_vagg(sd,solid)
    legacy=LB.lbm_step(f,solid,.8,a,[])
    corrected=LB.lbm_step(f,solid,.8,a,[],sdf_vagg=wall)
    np.testing.assert_array_equal(corrected,legacy)


def test_wall_free_abc_converges_second_order():
    errors=[]
    for N in (8,16):
        k=2*np.pi/N;U=.01*8/N;X,Y,Z=np.meshgrid(*[np.arange(N)]*3,indexing='ij')
        exact=np.stack([U*np.sin(k*Z)+.6*U*np.cos(k*Y),.8*U*np.sin(k*X)+U*np.cos(k*Z),.6*U*np.sin(k*Y)+.8*U*np.cos(k*X)],-1)
        sd=np.ones((N,N,N));solid,a,f=initial(sd);a[:]=.1*k*k*exact
        wall=LB.forbered_sdf_vagg(sd,solid);last=None
        for s in range(1,5001):
            f=LB.lbm_step(f,solid,.8,a,[],sdf_vagg=wall)
            if s%100==0:
                u=velocity(f,a)
                if last is not None and np.max(abs(u-last))<1e-9*np.max(abs(u)):break
                last=u.copy()
        else:raise AssertionError('ABC steady state not reached')
        errors.append(np.linalg.norm(u-exact)/np.linalg.norm(exact))
    assert np.log2(errors[0]/errors[1])>=1.8


@pytest.mark.parametrize('tau',[.6,.8,1.2])
@pytest.mark.parametrize('U',[.02,.05])
def test_forced_pipe_preserves_mass_for_40000_steps(tau,U):
    sd,_,_=pipe(8);g=4*((tau-.5)/3)*U/64;solid,a,f=initial(sd,g)
    wall=LB.forbered_sdf_vagg(sd,solid);m0=f[~solid].sum();history=[];minpop=1.
    for s in range(1,40001):
        f=LB.lbm_step(f,solid,tau,a,[],sdf_vagg=wall)
        if s%1000==0:
            history.append([s,float(f[~solid].sum()/m0-1)])
            minpop=min(minpop,float(f[~solid].min()))
    h=np.array(history);slope=np.polyfit(h[20:,0],h[20:,1],1)[0]
    print('MASS_RESULT '+json.dumps(dict(tau=tau,U=U,steps=40000,history=history,minpop=minpop,slope=float(slope))))
    assert np.isfinite(f).all() and minpop>0
    assert np.abs(h[:,1]).max()<1e-10
    assert abs(slope)<5e-15


def steady(sd,tau,g):
    solid,a,f=initial(sd,g);wall=LB.forbered_sdf_vagg(sd,solid);last=None
    for s in range(1,100001):
        f=LB.lbm_step(f,solid,tau,a,[],sdf_vagg=wall)
        if s%100==0:
            u=velocity(f,a)[~solid]
            if last is not None and np.max(abs(u-last))<1e-9*np.max(abs(u)):
                return velocity(f,a),s
            last=u.copy()
    raise AssertionError('steady state not reached')


@pytest.mark.parametrize('tau',[.6,.8,1.2])
@pytest.mark.parametrize('theta',[.25,.5,.75])
def test_plane_poiseuille_converges_second_order(tau,theta):
    errors=[]
    for n in (8,16,32):
        z=np.arange(n+2);sd=np.minimum(z-(1-theta),n+theta-z)[None,None,:].astype(float)
        H=n-1+2*theta;nu=(tau-.5)/3;g=8*nu*(.02*8/n)/H**2
        u,steps=steady(sd,tau,g);z=z[1:-1];exact=g/(2*nu)*(z-(1-theta))*(n+theta-z)
        errors.append(float(np.linalg.norm(u[0,0,1:-1,0]-exact)/np.linalg.norm(exact)))
    p=-np.polyfit(np.log([8,16,32]),np.log(errors),1)[0]
    print('PLANE_RESULT '+json.dumps(dict(tau=tau,theta=theta,errors=errors,order=float(p))))
    assert p>=1.8 and errors[-1]<.01


@pytest.mark.parametrize('phase',[(.13,.13),(.37,.21)])
def test_pipe_poiseuille_converges_second_order(phase):
    errors=[]
    for R in (8,16,32):
        sd,Y,Z=pipe(R,phase);nu=.1;g=4*nu*(.02*8/R)/R**2
        u,steps=steady(sd,.8,g);fluid=sd[0]>0;exact=g/(4*nu)*(R**2-Y**2-Z**2)
        errors.append(float(np.linalg.norm(u[0,...,0][fluid]-exact[fluid])/np.linalg.norm(exact[fluid])))
    p=-np.polyfit(np.log([8,16,32]),np.log(errors),1)[0]
    print('PIPE_RESULT '+json.dumps(dict(phase=phase,errors=errors,order=float(p))))
    assert p>=1.8 and errors[-1]<.01
