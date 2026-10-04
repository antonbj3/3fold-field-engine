"""Adversaries for geometry proof admission and actual Poisson consumer transport."""
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as Q
import math
import numpy as np
import pytest
from field_engine.experimental.geometry_contract import (
    AxisSlabs,BoxScale,SphereOffset,MeshPair,SDFGrid,Unverified,verify_geometry,
    assess,revalidate,verified,sqrt_bounds,conditional_normal_from_sdf,
    conditional_tube_measures,rat)
from field_engine.experimental.guaranteed_scalar import Mesh
from field_engine.experimental.guaranteed_geometry2d import poisson_with_geometry


def mesh(n=2,L=(1,1)):
    xy=[(Q(i,n)*L[0],Q(j,n)*L[1]) for j in range(n+1) for i in range(n+1)]
    ts=[]
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i;ts.extend(((a,a+1,a+n+2),(a,a+n+2,a+n+1)))
    return Mesh.admit(xy,ts)


def planar_pair(reverse=False):
    p=((0,0,0),(1,0,0),(0,1,0));f=((0,1,2),)
    return MeshPair(p,p,f,((0,2,1),) if reverse else f)


@pytest.mark.parametrize('e',[BoxScale((1,1)),BoxScale((1,1,1)),SphereOffset(1,1,1)])
def test_positive_identity_model(e):
    c=verify_geometry(e,pitch=Q(1,4))
    a=assess(c,evidence=e,distance=0,distance_unit='model',normal_chord=0,area_relative=0,volume_relative=0)
    assert a['status']=='PASS'
    assert assess(c,evidence=e,require_physical=True)['status']=='UNKNOWN'
    assert assess(c,evidence=e,require_contact=True)['status']=='UNKNOWN'


def test_hidden_disconnect_has_small_hausdorff_large_area_and_subgrid_features():
    eps=Q(1,2**30);e=AxisSlabs(((0,eps/2),(eps,1)))
    c=verify_geometry(e,pitch=Q(1,4))
    a=assess(c,evidence=e,distance=Q(1,10000),distance_unit='model',area_relative=Q(1,1000),volume_relative=Q(1,1000))
    assert a['checks']['distance']==a['checks']['volume']=='PASS'
    assert a['checks']['topology']==a['checks']['area']=='FAIL'
    assert c.topology.solid_betti==(2,0,0) and c.topology.electrode_connected is False
    assert c.minimum_void_channel.lower==eps/2
    assert c.feature_resolution=='SUBGRID_FEATURE'


def test_mesh_small_distance_does_not_imply_normal_or_area_accuracy():
    d=Q(1,2**30)
    p=((0,0,0),(2*d,0,0),(0,2*d,0));q=((0,0,0),(2*d,0,d),(0,2*d,0));f=((0,1,2),)
    e=MeshPair(p,q,f,f);c=verify_geometry(e,pitch=Q(1,4))
    a=assess(c,evidence=e,distance=Q(1,10000),distance_unit='model',normal_chord=Q(1,1000),area_relative=Q(1,1000),require_topology=False,require_resolution=False)
    assert a['checks']=={'distance':'PASS','normal':'FAIL','area':'FAIL'}


def test_oriented_normal_reversal_is_not_zero_geometry_error():
    e=planar_pair(reverse=True);c=verify_geometry(e,pitch=1)
    assert c.distance.upper==0 and c.area_relative_error.upper==0
    assert c.normal_chord.upper==2
    assert c.normal_angle_rad.upper>=math.pi


def test_open_surface_volume_topology_unknown_even_identity():
    e=planar_pair();c=verify_geometry(e,pitch=1)
    assert c.volume_measure.status==c.topology.status=='UNKNOWN'
    assert c.normal_chord.upper==c.distance.upper==0


@pytest.mark.parametrize('representation',['MESH','BREP','SDF','SCAN'])
def test_unsupported_input_flags_never_certify(representation):
    e=Unverified(representation,'fake-hash');c=verify_geometry(e,pitch=1)
    assert assess(c,evidence=e,distance=0,distance_unit='model',normal_chord=0,area_relative=0,volume_relative=0)['status']=='UNKNOWN'


def test_replaced_contract_and_stale_evidence_are_rejected():
    e=BoxScale((1,1),Q(999,1000),Q(1001,1000));c=verify_geometry(e,pitch=Q(1,4))
    forged=replace(c,distance=verified(0,0,'model','fake'))
    with pytest.raises(ValueError):assess(forged,evidence=e,distance=0)
    with pytest.raises(ValueError):revalidate(c,replace(e,scale_upper=2))
    with pytest.raises(ValueError):poisson_with_geometry(mesh(),e,forged)


@pytest.mark.parametrize('x',[True,np.bool_(False),math.nan,math.inf,'0.1',Decimal('0.1')])
def test_scalar_admission_never_silently_changes_inputs(x):
    with pytest.raises((TypeError,ValueError)):
        verify_geometry(BoxScale((1,1),x,1),pitch=1)


@pytest.mark.parametrize('e',[BoxScale((1,-1)),BoxScale((1,1),-1,1),SphereOffset(1,0,1),
                             AxisSlabs(((0,Q(1,2)),(Q(1,3),1)))])
def test_invalid_primitive_proofs(e):
    with pytest.raises(ValueError):verify_geometry(e,pitch=1)


def test_tiny_square_roots_and_huge_normals_enclosed_without_zero_division():
    for x in (Q(0),Q(1,2**1000),Q(2**1000),Q(2),Q(9,4)):
        a,b=sqrt_bounds(x)
        assert a*a<=x<=b*b and (a>0 or x==0)
    tiny=Q(1,2**600);p=((0,0,0),(tiny,0,0),(0,tiny,0))
    e=MeshPair(p,p,((0,1,2),),((0,2,1),))
    assert verify_geometry(e,pitch=1).normal_chord.upper==2


def test_trilinear_sdf_zero_field_is_not_a_solid_and_no_geometry_terms_inferred():
    ax=(Q(0),Q(1));e=SDFGrid((ax,ax,ax),(0,)*8,AxisSlabs(((0,1),)))
    c=verify_geometry(e,pitch=1)
    assert c.volume_measure.lower==c.volume_measure.upper==0
    assert c.volume_relative_error.upper==1
    assert c.normal_chord.status==c.area_relative_error.status==c.topology.status=='UNKNOWN'


def test_sdf_checks_every_node_and_pays_interpolation():
    ax=(0,Q(1,2),1);vals=tuple(-min(x,1-x,y,1-y,z,1-z) for x in ax for y in ax for z in ax)
    e=SDFGrid((ax,ax,ax),vals,AxisSlabs(((0,1),)))
    c=verify_geometry(e,pitch=Q(1,2))
    assert c.distance.upper>=math.sqrt(3)/4
    bad=list(vals);bad[13]+=Q(1,8)
    d=verify_geometry(replace(e,samples=tuple(bad)),pitch=Q(1,2))
    assert d.distance.upper==c.distance.upper+Q(1,8)
    with pytest.raises(ValueError):verify_geometry(replace(e,unit='mm'),pitch=Q(1,2))


def test_sdf_partial_coverage_is_rejected():
    ax=(Q(1,4),Q(1,2))
    with pytest.raises(ValueError):verify_geometry(SDFGrid((ax,ax,ax),(0,)*8,AxisSlabs(((0,1),))),pitch=1)


def test_conditional_normal_and_tube_theorems_are_not_certificates():
    eps=Q(1,10000);n=conditional_normal_from_sdf(eps,4,Q(1,2))
    assert n['status']=='CONDITIONAL_ONLY' and n['normal_chord_upper']**2>=8*eps
    t=conditional_tube_measures(eps,1,Q(1),4)
    assert t['area_ratio_lower']==(1-eps)**2
    assert t['area_ratio_upper']==(1+eps)**2
    assert t['status']=='CONDITIONAL_ONLY'
    with pytest.raises(ValueError):conditional_tube_measures(1,1,Q(1),4)
    with pytest.raises(ValueError):conditional_tube_measures(eps,1,0,4)


def test_actual_poisson_consumer_has_nonzero_geometry_term_and_identity_zero():
    e=BoxScale((1,1),Q(9999,10000),Q(10001,10000));c=verify_geometry(e,pitch=Q(1,4))
    out=poisson_with_geometry(mesh(),e,c)
    assert out['status']=='VERIFIED_MODEL_INTERVAL' and out['geometry_endpoint_addition_upper']>0
    assert out['lower']<out['reference_lower'] and out['upper']>out['reference_upper']
    e0=BoxScale((1,1));c0=verify_geometry(e0,pitch=Q(1,4))
    assert poisson_with_geometry(mesh(),e0,c0)['geometry_endpoint_addition_upper']==0


def test_poisson_consumer_refuses_unknown_geometry_before_any_solve(monkeypatch):
    from field_engine.experimental import guaranteed_geometry2d as consumer
    def forbidden(*args,**kwargs):raise AssertionError('solve must not run')
    monkeypatch.setattr(consumer.gs,'solve_poisson_fields',forbidden)
    e=Unverified('SDF');c=verify_geometry(e,pitch=1)
    out=poisson_with_geometry(None,e,c,expected_pitch=1)
    assert out['status']=='UNKNOWN' and out['solve_executed'] is False


def test_poisson_rejects_wrong_reference_units_and_replaced_mesh():
    e=BoxScale((1,1));c=verify_geometry(e,pitch=Q(1,4));m=mesh()
    with pytest.raises(ValueError):poisson_with_geometry(mesh(L=(2,1)),e,c)
    with pytest.raises(ValueError):poisson_with_geometry(m,e,c,expected_unit='mm')
    with pytest.raises(ValueError):poisson_with_geometry(m,e,c,expected_pitch=1)
    with pytest.raises(ValueError):poisson_with_geometry(replace(m),e,c)


def test_poisson_checks_entire_geometry_budget_before_solve(monkeypatch):
    from field_engine.experimental import guaranteed_geometry2d as consumer
    def forbidden(*args,**kwargs):raise AssertionError('solve must not run')
    monkeypatch.setattr(consumer.gs,'solve_poisson_fields',forbidden)
    e=BoxScale((1,1),Q(9,10),Q(11,10));c=verify_geometry(e,pitch=Q(1,4))
    assert poisson_with_geometry(mesh(),e,c)['status']=='BUDGET_FAIL'


# Reviewer regressions (SOL_FALT_GEOKONTRAKT review F1-F3).
@pytest.mark.parametrize('representation',['MESH','BREP','SDF','SCAN'])
def test_assessment_without_any_check_is_unknown_not_pass(representation):
    e=Unverified(representation);c=verify_geometry(e,pitch=1)
    assert assess(c,evidence=e,require_topology=False,require_resolution=False)['status']=='UNKNOWN'
    e=BoxScale((1,1));c=verify_geometry(e,pitch=Q(1,4))
    assert assess(c,evidence=e,require_topology=False,require_resolution=False)['status']=='UNKNOWN'


def test_distance_budget_must_carry_a_matching_unit():
    p=((0,0,0),(1,0,0),(0,1,0));q=tuple((x,y,z+Q(3,10000)) for x,y,z in p);f=((0,1,2),)
    e=MeshPair(p,q,f,f,unit='m');c=verify_geometry(e,pitch=Q(1,4))
    kw=dict(require_topology=False,require_resolution=False)
    with pytest.raises(ValueError):assess(c,evidence=e,distance=Q(1,1000),**kw)
    assert assess(c,evidence=e,distance=Q(1,1000),distance_unit='mm',**kw)['status']=='FAIL'
    assert assess(c,evidence=e,distance=Q(1,1000),distance_unit='m',**kw)['status']=='PASS'
    assert assess(c,evidence=e,distance=1,distance_unit='mm',**kw)['status']=='PASS'
    with pytest.raises(ValueError):assess(c,evidence=e,distance=1,distance_unit='model',**kw)
    u=replace(e,unit='source_unit_undocumented');cu=verify_geometry(u,pitch=Q(1,4))
    with pytest.raises(ValueError):assess(cu,evidence=u,distance=1,distance_unit='mm',**kw)


def test_signed_field_band_is_not_admitted_as_boundary_hausdorff():
    # Sub-pitch void channel x in (12/5,13/5) between SDF nodes 2 and 3: the trilinear
    # zero set is the cube surface, 2 away from the channel face, while the signed band is sqrt(3)/2.
    ref=AxisSlabs(((0,Q(12,5)),(Q(13,5),4)),(4,4));ax=tuple(Q(i) for i in range(5))
    def sd(x,y,z):
        inside=[(a,b) for a,b in ref.intervals if a<=x<=b]
        if inside and 0<=y<=4 and 0<=z<=4:
            a,b=inside[0];return -min(x-a,b-x,y,4-y,z,4-z)
        raise AssertionError('grid nodes lie in the solid or on its boundary')
    e=SDFGrid((ax,ax,ax),tuple(sd(*x) for x in __import__('itertools').product(ax,repeat=3)),ref)
    c=verify_geometry(e,pitch=1);kw=dict(require_topology=False,require_resolution=False,distance_unit='model')
    assert c.distance.upper<1
    assert assess(c,evidence=e,distance=1,**kw)['status']=='UNKNOWN'
    assert assess(c,evidence=e,distance=1,distance_metric='SIGNED_FIELD_LINF_ON_COVERED_BOX',**kw)['status']=='PASS'
