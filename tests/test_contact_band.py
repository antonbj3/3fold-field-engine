import numpy as np
import pytest
from field_engine.contact_band_v1 import contact_band,projected_sigma,region_with_uncertainty


def band(gap,**kwargs):
    gap=np.asarray(gap,dtype=float)
    defaults=dict(scan_sigma_mm=0.,sdf_radius_mm=0.,z=3.,completion_sign=np.where(gap<0,-1,1),evidence_id='analytically declared test geometry')
    defaults.update(kwargs)
    return contact_band(gap,**defaults)


def test_convex_gap_bias_cannot_hide_contact():
    # Analytic plane/cylinder tangent: uncorrected chord has 0.01 mm gap.
    assert band([.01],contact_distance_mm=.005).classes.tolist()==[1]
    corrected=band([.01],tess_bias_mm=.01,contact_distance_mm=.005)
    assert corrected.classes.tolist()==[-1]


def test_concave_bias_cannot_create_penetration():
    assert band([-.02],tess_bias_mm=-.02).classes.tolist()==[0]


def test_completion_unknown_overrides_large_numerical_margin():
    b=band([100.,-100.],completion_sign=[0,0],scan_sigma_mm=1e-6)
    assert b.classes.tolist()==[0,0]
    assert b.guard_input()['completion_unknown']==[0,1]


def test_no_completion_evidence_cannot_become_void():
    assert band([5],completion_sign=None).classes.tolist()==[0]


def test_distance_interval_changes_contact_even_with_known_outside_sign():
    # Correct outside sign still allows another hidden surface 0.1 mm away.
    assert band([5],gap_bounds_mm=[[.1,5]],contact_distance_mm=.3).classes.tolist()==[0]


def test_quantile_treated_as_radius_cannot_be_sigma():
    # API requires explicit caller radius; max 7.8 um each side adds 15.6 um.
    assert band([.01],sdf_radius_mm=.0156).classes.tolist()==[0]


def test_equal_threshold_abstains():
    assert band([.3],contact_distance_mm=.3).classes.tolist()==[0]


def test_signed_scan_bias_is_separate_from_spread():
    b=band([.2],scan_bias_mm=.2,scan_sigma_mm=.01,z=2)
    assert b.sigma_tot_mm[0]==.01
    assert b.classes.tolist()==[0]


def test_shared_surface_error_cancels_in_difference():
    c=np.ones((2,2))
    assert projected_sigma(c,[1,-1])==0
    assert projected_sigma(c,[1,1])==2


def test_unknown_correlation_worst_case_is_not_independent_sigma():
    independent=projected_sigma(np.eye(2),[1,1])
    shared=projected_sigma(np.ones((2,2)),[1,1])
    assert independent<shared


@pytest.mark.parametrize('cov',[np.array([[1.,2.],[2.,1.]]),np.array([[1.,0.],[.1,1.]])])
def test_invalid_covariance_refused(cov):
    with pytest.raises(ValueError):projected_sigma(cov,[1,1])


@pytest.mark.parametrize('kwargs',[{'sdf_radius_mm':-.1},{'scan_sigma_mm':float('nan')},{'z':-1},{'completion_sign':[2]},{'gap_bounds_mm':[[1,0]]},{'evidence_id':''}])
def test_invalid_contract_refused(kwargs):
    with pytest.raises(ValueError):band([0.],**kwargs)


def test_geometry_guard_expands_both_actual_topologies():
    b=band([0,2,2]);assert region_with_uncertainty((0,1),b,n_nodes=3,contact_topology='node')==(0,3)
    assert region_with_uncertainty((0,1),b,n_nodes=4,contact_topology='edge')==(0,4)


def test_empty_band_preserves_consumer_region():
    b=band([-1,1]);assert region_with_uncertainty((0,1),b,n_nodes=2,contact_topology='node')==(0,1)


def test_node_edge_contract_cannot_be_confused():
    with pytest.raises(ValueError):region_with_uncertainty((0,1),band([0,2]),n_nodes=2,contact_topology='edge')


def test_clearly_separated_and_penetrating_contacts_are_decided():
    b=band([2.,-1.],scan_sigma_mm=.05,sdf_radius_mm=.002,contact_distance_mm=.3)
    assert b.classes.tolist()==[1,-1]


def test_known_sign_contradicting_distance_stays_indeterminate():
    b=band([5.,-5.],completion_sign=[-1,1],contact_distance_mm=.3)
    assert b.classes.tolist()==[0,0]
    assert b.guard_input()['sign_conflict']==[0,1]


def test_gap_outside_its_completion_interval_refused():
    with pytest.raises(ValueError):band([5.],gap_bounds_mm=[[10.,20.]])


def test_z_has_no_default_and_must_be_scalar():
    with pytest.raises(TypeError):
        contact_band([1.],scan_sigma_mm=0.,sdf_radius_mm=0.,completion_sign=[1],evidence_id='x')
    with pytest.raises(ValueError):band([1.,2.],z=[3.,3.])


def test_band_arrays_are_read_only():
    b=band([1.])
    with pytest.raises(ValueError):b.classes[0]=0


def test_projected_sigma_does_not_depend_on_length_unit():
    c=np.array([[1.,.5],[.5,1.]])*1e-2
    mm=projected_sigma(c,[1,1]);um=projected_sigma(c*1e6+np.array([[0,1e-9],[0,0]]),[1,1])
    assert abs(um/1e3-mm)<1e-12
    with pytest.raises(ValueError):projected_sigma(np.array([[1e-15,0],[0,-5e-15]]),[0,1])
    with pytest.raises(ValueError):projected_sigma(np.zeros((0,0)),[])


@pytest.mark.parametrize('region',[(.5,2.),(False,True)])
def test_region_bounds_must_be_integers(region):
    with pytest.raises(ValueError):region_with_uncertainty(region,band([2.,2.,2.]),n_nodes=3,contact_topology='node')
