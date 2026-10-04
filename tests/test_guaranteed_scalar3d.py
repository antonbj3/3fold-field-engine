"""Continuum obligations, invalid inputs and consumer regressions."""
from fractions import Fraction as F
from itertools import combinations
import math
import numpy as np
import pytest
import dataclasses
from decimal import Decimal
from field_engine.experimental import guaranteed_scalar3d as c
from field_engine.experimental import guaranteed_geometry3d as g


def octahedron():
    p=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    faces=[(i,j,k) for i in (0,1) for j in (2,3) for k in (4,5)]
    return c.convex_star_mesh(p,faces)


@pytest.mark.parametrize('degree',[1,2])
@pytest.mark.parametrize('sigma',[F(1,2),F(1),F(3)])
def test_finite_3d_electrodes_exact_affine_resistance(degree,sigma):
    mesh=c.structured_mesh(2,(2,1,1));e=c.box_electrodes(mesh)
    v,z,_=c.solve_fields(mesh,0,sigma,e,degree);cert=c.certificate(mesh,v,z,0,sigma,e,degree)
    expected=2/sigma
    assert c.rat(cert['lower'])<=expected<=c.rat(cert['upper'])
    assert cert['relative_width_upper']<1e-12
    assert cert['gap_upper']<1e-22
    assert c.raw_error_upper(cert,0)>=float(expected)


@pytest.mark.parametrize('degree',[1,2])
def test_bad_primal_still_valid_compliance_on_sphere(degree):
    mesh=octahedron();enc=g.homothety_to_ellipsoid(mesh,(1,1,1));dofs,bc,_,nv=c.potential_layout(mesh,degree)
    v=np.zeros(nv);free=sorted(set(range(nv))-set(bc));v[free]=100
    z=c.repair_flux(mesh,np.zeros(len(mesh.faces)))
    cert=c.certificate(mesh,v,z,potential_degree=degree);phys=g.ellipsoid_compliance(cert,enc)
    facit=4*math.pi/45
    assert phys['lower']<=facit<=phys['upper']
    assert phys['absolute_error_upper']>=abs(phys['estimate']-facit)


def test_exact_balance_required_and_tree_repair_not_tolerance():
    mesh=c.structured_mesh(2);v,z,_=c.solve_fields(mesh);bad=list(z);bad[0]+=F(1,2**60)
    with pytest.raises(ValueError,match='balance'):c.certificate(mesh,v,bad)
    repaired=c.repair_flux(mesh,bad);cert=c.certificate(mesh,v,repaired)
    assert cert['source_balance_exact'] and cert['flux_load_identity_exact']


def test_boundary_p2_edge_trace_is_checked():
    mesh=c.structured_mesh(2);v,z,_=c.solve_fields(mesh,potential_degree=2)
    _,bc,_,_=c.potential_layout(mesh,2);edge=next(i for i in bc if i>=len(mesh.points));v[edge]=.1
    with pytest.raises(ValueError,match='Dirichlet'):c.certificate(mesh,v,z,potential_degree=2)


def test_unit_current_and_wall_flux_are_required():
    mesh=c.structured_mesh(2);e=c.box_electrodes(mesh);v,z,_=c.solve_fields(mesh,0,electrodes=e)
    with pytest.raises(ValueError,match='unit right'):c.certificate(mesh,v,[q*F(3,4) for q in z],0,electrodes=e)
    bad=list(z);bad[e['wall'][0]]=F(1,100)
    with pytest.raises(ValueError,match='insulating'):c.certificate(mesh,v,bad,0,electrodes=e)
    v[mesh.faces[e['left'][0]][0]]=1
    with pytest.raises(ValueError,match='electrode traces'):c.certificate(mesh,v,z,0,electrodes=e)


@pytest.mark.parametrize('kind',['potential','flux','coefficient'])
def test_nonfinite_inputs_refused(kind):
    mesh=c.structured_mesh(2);v,z,_=c.solve_fields(mesh);kw={}
    if kind=='potential':v[0]=np.nan
    elif kind=='flux':z=list(z);z[0]=np.inf
    else:kw['coefficient']=np.nan
    with pytest.raises(ValueError,match='finite'):c.certificate(mesh,v,z,**kw)


@pytest.mark.parametrize('degree',[1,2])
def test_material_contrast_has_valid_numeric_bound_and_no_scaling_claim(degree):
    mesh=octahedron();k=[1,10**4]*4;v,z,_=c.solve_fields(mesh,coefficient=k,potential_degree=degree)
    cert=c.certificate(mesh,v,z,coefficient=k,potential_degree=degree)
    assert cert['lower']<=cert['upper'] and not cert['constant_coefficient']
    with pytest.raises(ValueError,match='constant-coefficient'):g.ellipsoid_compliance(cert,g.homothety_to_ellipsoid(mesh,(1,1,1)))


def test_refinement_preserves_exact_geometry_volume_and_face_admission():
    mesh=c.structured_mesh(1,l_prism=True);ref=c.bisect_edges(mesh,c.longest_edges(mesh,range(len(mesh.tets))))
    assert sum(mesh.volumes)==sum(ref.volumes)==3
    assert all(V>0 for V in ref.volumes)
    assert all(len(inc) in (1,2) for inc in ref.incidences)
    assert ref.sha256!=mesh.sha256


def test_geometric_claim_needs_verified_object_for_exact_mesh():
    mesh=octahedron();v,z,_=c.solve_fields(mesh);cert=c.certificate(mesh,v,z)
    with pytest.raises(ValueError,match='global signed-distance'):g.uncertain_convex_compliance(cert,mesh,0,{'scope':'GLOBAL_SIGNED_DISTANCE_ON_COVERED_BOX','median':0})
    with pytest.raises(ValueError,match='verified homothety'):g.ellipsoid_compliance(cert,{'rmin_lower':1,'rmax_upper':1})
    other=c.bisect_edges(mesh,c.longest_edges(mesh,[0]));ov,oz,_=c.solve_fields(other);othercert=c.certificate(other,ov,oz)
    with pytest.raises(ValueError,match='exact certificate mesh'):g.ellipsoid_compliance(othercert,g.homothety_to_ellipsoid(mesh,(1,1,1)))


def test_all_node_verifier_pays_interpolation_and_mesh_binding():
    mesh=octahedron();axes=[[-2,0,2]]*3;sd=np.zeros((3,3,3))
    proof=g.verify_grid_sdf(mesh,axes,sd);s=proof.summary
    assert s['verified_nodes']==27 and s['distance_bound_upper']>=s['nodal_error_upper']+s['interpolation_error_upper']-1e-14
    v,z,_=c.solve_fields(mesh);cert=c.certificate(mesh,v,z)
    out=g.uncertain_convex_compliance(cert,mesh,s['distance_bound_upper'],proof)
    assert out['lower']==0 and out['physical_acquisition_error']=='UNKNOWN'
    with pytest.raises(ValueError,match='global signed-distance'):g.uncertain_convex_compliance(cert,mesh,0,proof)
    other=octahedron()
    with pytest.raises(ValueError,match='global signed-distance'):g.uncertain_convex_compliance(cert,other,s['distance_bound_upper'],proof)
    summary=proof.summary;summary['distance_bound_upper']=0
    assert proof.summary['distance_bound_upper']>0
    original_hash=proof.summary['sdf_model_sha256'];sd[0,0,0]=1
    assert proof.summary['sdf_model_sha256']==original_hash
    assert g.verify_grid_sdf(mesh,axes,sd).summary['sdf_model_sha256']!=original_hash


def test_triangle_distance_handles_projection_edge_and_vertex():
    a,b,d=(F(0),F(0),F(0)),(F(1),F(0),F(0)),(F(0),F(1),F(0))
    assert g.triangle_distance2((F(1,4),F(1,4),F(2)),a,b,d)==4
    assert g.triangle_distance2((F(1),F(1),F(0)),a,b,d)==F(1,2)
    assert g.triangle_distance2((F(-1),F(-1),F(0)),a,b,d)==2


def test_sqrt_bounds_cover_extreme_scales_and_exact_squares():
    for x in [F(1,3),F(1,10**200),F(10**200),F(9,16),F(0)]:
        a,b=g.sqrt_bounds(x);assert a*a<=x<=b*b
        if x:assert a>0
    assert g.sqrt_bounds(F(9,16))==(F(3,4),F(3,4))


def test_unknown_resistance_geometry_and_box_face_contract_are_distinct():
    r=g.uncertain_box_resistance((2,1,1),F(1,1024))
    assert r['lower']<2<r['upper'] and r['relative_width_upper']>.005
    mesh=octahedron();v,z,_=c.solve_fields(mesh);cert=c.certificate(mesh,v,z)
    with pytest.raises(ValueError,match='only constant-coefficient'):g._transport_compliance({**cert,'quantity':'effective_resistance'},1,1,{'explicit':True})


def test_unsupported_mesh_and_open_convex_boundary_refused():
    mesh=octahedron();bad=c.Mesh3D(mesh.points,mesh.tets,mesh.volumes,mesh.faces,mesh.incidences,mesh.tet_faces,mesh.boundary_nodes,mesh.provenance,mesh.sha256)
    with pytest.raises(ValueError,match='constructive'):c.certificate(bad,[0]*len(mesh.points),[0]*len(mesh.faces))
    triangles=g.boundary_triangles(mesh)
    with pytest.raises(ValueError):c.convex_star_mesh(mesh.points[:-1],triangles[:-1])


def test_invalid_shape_coefficient_and_electrode_partition_refused():
    mesh=c.structured_mesh(2);v,z,_=c.solve_fields(mesh)
    with pytest.raises(ValueError,match='shape'):c.certificate(mesh,v[:-1],z)
    with pytest.raises(ValueError,match='positive'):c.certificate(mesh,v,z,coefficient=0)
    e=c.box_electrodes(mesh);e['wall']=e['wall'][:-1]
    with pytest.raises(ValueError,match='partition'):c.certificate(mesh,v,z,0,electrodes=e)


def test_verified_continuous_shear_family_has_separate_geometry_and_decision():
    mesh=c.structured_mesh(2,(2,1,1));e=c.box_electrodes(mesh);v,z,_=c.solve_fields(mesh,0,electrodes=e,potential_degree=2)
    cert=c.certificate(mesh,v,z,0,electrodes=e,potential_degree=2)
    proof=g.verified_shear_family(mesh,F(1,4096));out=g.shear_family_response(cert,proof)
    assert 1.999<=out['lower']<=2<=out['upper']<=2.001
    assert out['relative_width_upper']<.005 and out['geometry_error_upper_separate']>0
    assert out['physical_acquisition_error']=='UNKNOWN'
    for beta in [-F(1,4096),F(1,4096)]:
        image=c.affine_mesh(mesh,[[1,beta,0],[0,1,0],[0,0,1]])
        iv,iz,_=c.solve_fields(image,0,electrodes=e,potential_degree=2);ic=c.certificate(image,iv,iz,0,electrodes=e,potential_degree=2)
        assert out['lower']<=ic['lower']<=ic['upper']<=out['upper']


def test_unverified_map_wrong_mesh_and_noninvertible_affine_refused():
    mesh=c.structured_mesh(2);v,z,_=c.solve_fields(mesh);cert=c.certificate(mesh,v,z)
    with pytest.raises(ValueError,match='verified shear'):g.shear_family_response(cert,{'kappa_min':1,'kappa_max':1})
    with pytest.raises(ValueError,match='verified shear'):g.shear_family_response(cert,g.verified_shear_family(c.structured_mesh(1),F(1,100)))
    with pytest.raises(ValueError,match='positive affine'):c.affine_mesh(mesh,[[1,0,0],[0,0,0],[0,0,1]])
    with pytest.raises(ValueError,match='distinct'):g.verified_shear_family(mesh,1)


def test_exact_rational_rotation_preserves_resistance_under_affine_admission():
    mesh=c.structured_mesh(1,(2,1,1));e=c.box_electrodes(mesh)
    image=c.affine_mesh(mesh,[[F(3,5),F(-4,5),0],[F(4,5),F(3,5),0],[0,0,1]])
    v,z,_=c.solve_fields(image,0,electrodes=e,potential_degree=2);cert=c.certificate(image,v,z,0,electrodes=e,potential_degree=2)
    assert cert['lower']<=2<=cert['upper'] and cert['relative_width_upper']<1e-12


@pytest.mark.parametrize('degree',[1,2])
def test_transported_energy_moments_close_whole_shear_family(degree):
    mesh=c.structured_mesh(2,(2,1,1));e=c.box_electrodes(mesh);v,z,_=c.solve_fields(mesh,0,electrodes=e,potential_degree=degree)
    cert=c.certificate(mesh,v,z,0,electrodes=e,potential_degree=degree,with_energy_moments=True)
    family=g.verified_shear_family(mesh,F(1,4096));out=g.field_shear_family_response(cert,family)
    assert out['relative_width_upper']<1e-6 and 1.999<out['lower']<=2<=out['upper']<2.001
    assert out['relative_width_upper']<g.shear_family_response(cert,family)['relative_width_upper']/1000
    assert out['consumer_geometry_change_upper']>=2-out['lower']
    identity=g.field_shear_family_response(cert,g.verified_shear_family(mesh,0))
    assert identity['consumer_geometry_change_upper']==0


def test_energy_moments_are_required_and_have_valid_shape():
    mesh=c.structured_mesh(2);v,z,_=c.solve_fields(mesh);cert=c.certificate(mesh,v,z)
    family=g.verified_shear_family(mesh,F(1,100))
    with pytest.raises(ValueError,match='energy moments'):g.field_shear_family_response(cert,family)
    cert=c.certificate(mesh,v,z,with_energy_moments=True);cert['energy_moments']['primal']=[[1]]
    with pytest.raises(ValueError,match='3x3'):g.field_shear_family_response(cert,family)


def test_source_poisson_transport_on_nonbox_shape_uses_same_continuum_fields():
    mesh=octahedron();v,z,_=c.solve_fields(mesh,potential_degree=2)
    cert=c.certificate(mesh,v,z,potential_degree=2,with_energy_moments=True)
    family=g.verified_shear_family(mesh,F(1,16));phys=g.field_shear_family_response(cert,family)
    for beta in [-F(1,16),F(1,16)]:
        image=c.affine_mesh(mesh,[[1,beta,0],[0,1,0],[0,0,1]])
        iv,iz,_=c.solve_fields(image,potential_degree=2);ic=c.certificate(image,iv,iz,potential_degree=2)
        assert ic['lower']<=phys['upper'] and ic['upper']>=phys['lower']


def test_modified_admitted_mesh_copy_is_refused_before_using_cached_volume():
    mesh=c.structured_mesh(2);v,z,_=c.solve_fields(mesh)
    forged=dataclasses.replace(mesh,points=tuple(tuple(16*x for x in p) for p in mesh.points))
    with pytest.raises(ValueError,match='constructive'):c.certificate(forged,[0]*len(mesh.points),z)
    with pytest.raises(ValueError,match='constructive'):c.certificate(dataclasses.replace(mesh),v,z)


def test_modified_verified_geometry_objects_are_not_admitted():
    mesh=octahedron();v,z,_=c.solve_fields(mesh);cert=c.certificate(mesh,v,z)
    enc=g.homothety_to_ellipsoid(mesh,(1,1,1))
    with pytest.raises(ValueError,match='verified'):g.ellipsoid_compliance(cert,dataclasses.replace(enc,rmin=F(100)))
    family=g.verified_shear_family(mesh,F(1,4096))
    with pytest.raises(ValueError,match='verified'):g.shear_family_response(cert,dataclasses.replace(family,kappa_min=F(10),kappa_max=F(10)))
    proof=g.verify_grid_sdf(mesh,[[-2,0,2]]*3,np.zeros((3,3,3)))
    with pytest.raises(ValueError,match='global signed-distance'):g.uncertain_convex_compliance(cert,mesh,0,dataclasses.replace(proof,distance_bound=F(0)))


@pytest.mark.parametrize('value',[Decimal('0.1'),np.longdouble(1)+np.finfo(np.longdouble).eps,'0.5',True])
def test_nonexact_or_nonnumeric_values_do_not_silently_change_inputs(value):
    with pytest.raises((ValueError,TypeError)):c.rat(value)
    assert c.rat(2**54+1)==F(2**54+1)
    assert c.rat(np.int64(2**54+1))==F(2**54+1)
    assert c.rat(Decimal('1.25'))==F(5,4)


def test_unordered_raw_readout_interval_refused():
    with pytest.raises(ValueError,match='ordered'):c.raw_error_upper({'lower':2,'upper':1},1.5)


def test_fractional_electrode_identifiers_do_not_change_terminal_identity():
    mesh=c.structured_mesh(2);e=c.box_electrodes(mesh);v,z,_=c.solve_fields(mesh,0,electrodes=e)
    e['left']=tuple(F(i)+F(1,2) for i in e['left'])
    with pytest.raises(ValueError,match='integer electrode-face'):c.certificate(mesh,v,z,0,electrodes=e)


def test_homothety_refuses_replaced_or_hand_built_mesh():
    # Review F1: the enclosure is bound to mesh.sha256; a replaced copy keeps
    # the hash with other points and gave [2.9e-5,6.8e-4] around true 0.279.
    import types
    mesh=octahedron();big=tuple(tuple(4*x for x in p) for p in mesh.points)
    with pytest.raises(ValueError,match='constructive'):g.homothety_to_ellipsoid(dataclasses.replace(mesh,points=big),(1,1,1))
    fake=types.SimpleNamespace(points=big,tets=mesh.tets,sha256=mesh.sha256)
    with pytest.raises(ValueError,match='constructive'):g.homothety_to_ellipsoid(fake,(1,1,1))
    v,z,_=c.solve_fields(mesh,potential_degree=2);cert=c.certificate(mesh,v,z,potential_degree=2)
    out=g.ellipsoid_compliance(cert,g.homothety_to_ellipsoid(mesh,(1,1,1)))
    assert out['lower']<=4*math.pi/45<=out['upper']


def test_sdf_evidence_is_checked_against_the_consumer_array():
    # Review F2: the evidence holds a hash; the array used later must match it.
    mesh=octahedron();ax=np.linspace(-1.5,1.5,5);X,Y,Z=np.meshgrid(ax,ax,ax,indexing='ij')
    sd=(np.abs(X)+np.abs(Y)+np.abs(Z)-1)/math.sqrt(3);proof=g.verify_grid_sdf(mesh,[ax]*3,sd)
    assert proof.matches([ax]*3,sd)
    v,z,_=c.solve_fields(mesh);cert=c.certificate(mesh,v,z);bound=proof.summary['distance_bound_upper']
    g.uncertain_convex_compliance(cert,mesh,bound,proof,[ax]*3,sd)
    sd[2,2,2]=5.0
    assert not proof.matches([ax]*3,sd)
    with pytest.raises(ValueError,match='differs'):g.uncertain_convex_compliance(cert,mesh,bound,proof,[ax]*3,sd)
    with pytest.raises(ValueError,match='differs'):g.uncertain_convex_compliance(cert,mesh,bound,proof,[ax]*3,None)


def test_box_resistance_premise_is_labelled_caller_asserted():
    assert g.uncertain_box_resistance((2,1,1),F(1,1024))['geometry_premise']=='CALLER_ASSERTED_NOT_VERIFIED'
