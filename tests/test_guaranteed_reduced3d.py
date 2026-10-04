"""End-to-end certificates, exact Fraction oracle and invalid input controls."""
from fractions import Fraction as F
from itertools import combinations
from dataclasses import replace
import numpy as np
import pytest
from field_engine.experimental import guaranteed_scalar3d as c
from field_engine.experimental import guaranteed_reduced3d as r
from field_engine.experimental import guaranteed_draft3d as draft


def diag(a):return [[a[0],0,0],[0,a[1],0],[0,0,a[2]]]


def poisson_basis(degree=2):
    m=c.structured_mesh(2);basis=r.ReducedCertificate(m,1,degree=degree)
    for axes in ((F(1),F(1),F(1)),(F(1,2),F(1),F(1))):
        img=c.affine_mesh(m,diag(axes));v,z,_=c.solve_fields(img,1,1,None,degree)
        basis.add(v,[x/(axes[0]*axes[1]*axes[2]) for x in z])
    return m,basis


def exact_reconstructed(basis,cert,A,conductivity=None):
    m=c.affine_mesh(basis.mesh,A);_,det,_=r.fc._mat(A)
    v=basis.reconstruct(cert['alpha'],'potential');z=basis.reconstruct(cert['beta'],'flux')
    if basis.electrodes is None:
        z=tuple(x*det for x in z)
        ex=c.certificate(m,v,z,basis.source,1,None,basis.degree,with_energy_moments=True)
        load=F(ex['energy_moments']['load_exact']);energy=sum(F(ex['energy_moments']['primal'][i][i]) for i in range(3))
        v=tuple(x*load/energy for x in v);coeff=1
    else:
        coeff=[conductivity[x] for x in basis.labels] if conductivity else 1
    return c.certificate(m,v,z,basis.source,coeff,basis.electrodes,basis.degree)


@pytest.mark.parametrize('degree',[1,2])
@pytest.mark.parametrize('A',[
    diag((F(3,4),F(5,4),F(1))),
    [[F(1),F(1,3),0],[0,F(1),F(-1,4)],[0,0,F(1)]]])
def test_cross_moments_enclose_fraction_under_affine_maps(degree,A):
    _,basis=poisson_basis(degree)
    cert=basis.bounds(A,alpha=[F(3,8),F(5,8)],beta=[F(1,4),F(3,4)])
    ex=exact_reconstructed(basis,cert,A)
    assert 0<cert['lower']<=ex['lower']<=ex['upper']<=cert['upper']
    assert abs(cert['upper']-ex['upper'])<1e-9*ex['upper']
    assert cert['physical_geometry_error']=='UNKNOWN'


@pytest.mark.parametrize('family',['series','parallel'])
@pytest.mark.parametrize('k',[F(1,1000),F(1),F(1000)])
def test_material_cross_terms_recover_exact_layered_resistance(family,k):
    m=c.structured_mesh(2,(2,1,1));e=c.box_electrodes(m)
    centers=[tuple(sum(m.points[i][d] for i in t)/4 for d in range(3)) for t in m.tets]
    labels=['a' if (x<1 if family=='series' else y<F(1,2)) else 'b' for x,y,z in centers]
    basis=r.ReducedCertificate(m,0,e,2,labels)
    for anchor in (F(1,10000),F(10000)):
        v,z,_=c.solve_fields(m,0,[anchor if x=='b' else F(1) for x in labels],e,2);basis.add(v,z)
    cert=basis.bounds(conductivity={'a':1,'b':k})
    ex=exact_reconstructed(basis,cert,diag((1,1,1)),{'a':1,'b':k})
    true=1+1/k if family=='series' else 4/(1+k)
    assert F(cert['lower'])<=true<=F(cert['upper'])
    assert cert['lower']<=ex['lower']<=ex['upper']<=cert['upper']
    assert cert['relative_width_upper']<1e-10
    if k==1:
        one=basis.bounds(conductivity={'a':1,'b':k},alpha=[1,0],beta=[1,0])
        assert one['relative_width_upper']>0.1


def test_reduced_minimum_is_better_than_every_single_field():
    _,basis=poisson_basis()
    A=diag((F(3,4),1,1));mixed=basis.bounds(A)
    vertices=[basis.bounds(A,alpha=a,beta=a) for a in ([1,0],[0,1])]
    assert mixed['lower']>max(v['lower'] for v in vertices)
    assert mixed['upper']<min(v['upper'] for v in vertices)
    ex=exact_reconstructed(basis,mixed,A)
    assert mixed['lower']<=ex['lower']<=ex['upper']<=mixed['upper']


def test_exact_simplex_survives_rounding_and_singular_duplicate_fields():
    m,basis=poisson_basis();basis.add(*basis.fields[0],normalize_load=False)
    cert=basis.bounds();assert cert['lower']>0
    for w in (cert['alpha'],cert['beta'],r.simplex_dyadic([1e-300,0.4,0.6])):
        assert sum(w)==1 and min(w)>=0
    ex=exact_reconstructed(basis,cert,diag((1,1,1)))
    assert cert['lower']<=ex['lower']<=ex['upper']<=cert['upper']


def test_exact_integer_load_matches_fraction_expression():
    m,basis=poisson_basis();cert=basis.bounds();assert cert['lower']>0
    v=basis.fields[0][0];dofs=c.potential_layout(m,2)[0]
    oracle=sum((V*(-sum((v[i] for i in row[:4]),F(0))/20+sum((v[i] for i in row[4:]),F(0))/5) for V,row in zip(m.volumes,dofs)),F(0))
    assert r.exact_load(m,v)==oracle==1


@pytest.mark.parametrize('bad',['trace','balance','weights','coefficient','map','context','mesh','labels'])
def test_invalid_hypotheses_are_rejected_after_valid_certificate(bad):
    m,basis=poisson_basis();cert=basis.bounds();assert cert['lower']>0
    v,z=basis.fields[0]
    if bad=='trace':
        v=list(v);v[m.boundary_nodes[0]]=F(1,2**60)
        with pytest.raises(ValueError):basis.add(v,z)
    elif bad=='balance':
        z=list(z);z[0]+=F(1,2**60)
        with pytest.raises(ValueError):basis.add(v,z)
    elif bad=='weights':
        with pytest.raises(ValueError):basis.bounds(alpha=[F(1),F(1,2**60)])
        with pytest.raises(ValueError):basis.bounds(beta=[F(2),F(-1)])
    elif bad=='coefficient':
        with pytest.raises(ValueError):basis.bounds(conductivity={None:F(0)})
    elif bad=='map':
        with pytest.raises(ValueError):basis.bounds(diag((-1,1,1)))
    elif bad=='labels':
        with pytest.raises(ValueError):r.ReducedCertificate(m,region_labels=[float('nan')]+[None]*(len(m.tets)-1))
    elif bad=='context':
        with pytest.raises(AttributeError):basis.source=0
        with pytest.raises(TypeError):basis._G[0,0,0,0,0]=(0,0)
    else:
        with pytest.raises(ValueError):r.ReducedCertificate(replace(m))


@pytest.mark.parametrize('mode',['warm','dual_only','primal_only'])
def test_iterative_drafts_repair_balance_and_match_direct_certificate(mode):
    m=c.structured_mesh(2);v0,z0,_=c.solve_fields(m,1,1,None,2)
    if mode=='warm':
        v,z,cost=draft.solve_fields_warm(m,1,1,None,2)
        state=cost.pop('state');v,z,cost=draft.solve_fields_warm(m,1,1,None,2,state)
    elif mode=='dual_only':
        z,cost=draft.solve_dual_warm(m,1,1,None);v=v0
    else:
        v,cost=draft.solve_primal_warm(m,1,1,None,2,v0);z=z0
    ex=c.certificate(m,v,z,1,1,None,2);direct=c.certificate(m,v0,z0,1,1,None,2)
    assert ex['flux_load_identity_exact'] and ex['source_balance_exact']
    assert abs(ex['upper']-direct['upper'])<1e-7*direct['upper']
    assert abs(ex['lower']-direct['lower'])<1e-7*direct['lower']
    assert cost['assembly_s']>=0


def test_resistance_combines_material_and_nondiagonal_affine_metric():
    m=c.structured_mesh(2,(2,1,1));e=c.box_electrodes(m)
    labels=['a' if sum(m.points[i][0] for i in t)/4<1 else 'b' for t in m.tets]
    basis=r.ReducedCertificate(m,0,e,2,labels)
    for k in (F(1,4),F(4)):
        v,z,_=c.solve_fields(m,0,[k if x=='b' else F(1) for x in labels],e,2);basis.add(v,z)
    A=[[F(3,2),F(1,3),0],[0,F(2,3),F(-1,4)],[0,0,F(5,4)]]
    k={'a':F(3,2),'b':F(7,3)}
    cert=basis.bounds(A,conductivity=k,alpha=[F(2,5),F(3,5)],beta=[F(7,11),F(4,11)])
    ex=exact_reconstructed(basis,cert,A,k)
    assert 0<cert['lower']<=ex['lower']<=ex['upper']<=cert['upper']
    assert abs(cert['upper']-ex['upper'])<1e-9*ex['upper']


def test_cross_moments_preserve_exact_far_translation():
    m,basis=poisson_basis();translated=c.affine_mesh(m,diag((1,1,1)),(F(10**30),F(-10**30),F(10**30)))
    shifted=r.ReducedCertificate(translated,1,degree=2)
    for v,z in basis.fields:shifted.add(v,z,normalize_load=False)
    cert=basis.bounds(alpha=[F(1,2),F(1,2)],beta=[F(1,2),F(1,2)])
    other=shifted.bounds(alpha=cert['alpha'],beta=cert['beta'])
    assert cert['lower']>0 and cert['lower']==other['lower'] and cert['upper']==other['upper']
    ex=exact_reconstructed(shifted,other,diag((1,1,1)))
    assert other['lower']<=ex['lower']<=ex['upper']<=other['upper']


@pytest.mark.parametrize('degree',[1,2])
@pytest.mark.parametrize('nbases',[2,3])
def test_material_batch_encloses_fraction_for_every_reconstructed_query(degree,nbases):
    m,basis=poisson_basis(degree)
    # Three fields also exercises the general QP proposal path in batch mode.
    if nbases==3:basis.add(*basis.fields[0],normalize_load=False)
    ks=[F(1,13),F(1),F(17,3)]
    answers=basis.bounds_material_batch([{None:k} for k in ks])
    assert len(answers)==len(ks)
    for k,cert in zip(ks,answers):
        v=basis.reconstruct(cert['alpha'],'potential');z=basis.reconstruct(cert['beta'],'flux')
        first=c.certificate(m,v,z,1,k,None,degree,with_energy_moments=True)
        E=sum(F(first['energy_moments']['primal'][i][i]) for i in range(3));L=F(first['energy_moments']['load_exact'])
        exact=c.certificate(m,tuple(x*L/E for x in v),z,1,k,None,degree)
        assert 0<cert['lower']<=exact['lower']<=exact['upper']<=cert['upper']
        assert sum(cert['alpha'])==sum(cert['beta'])==1
    assert answers[0]['lower']>answers[1]['lower']>answers[2]['lower']
    with pytest.raises(ValueError):basis.bounds_material_batch([{None:0}])
