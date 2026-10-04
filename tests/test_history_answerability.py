"""Regressions for erased modes and inappropriate nonlinear acceptance."""
import numpy as np
import pytest
from field_engine.experimental.history_answerability import diagnose, minrank, two_point


def test_wide_summary_keeps_complete_nullspace():
    # Reduced SVD has only one right row here; the missing fourth direction
    # must still be detected, before pseudoinverse reconstruction.
    r=diagnose([0,0,0,1],[[1,0,0,0]],[1,1,1,1],epsilon=.01,query_contract='linear')
    assert r['status']=='LINEAR_NUMERICAL_INSUFFICIENT'
    assert r['LB']==pytest.approx(1)
    assert np.max(abs(np.array([[1,0,0,0]])@r['witness']))<1e-8


def test_reachability_equality_removes_apparent_blind_mode():
    r=diagnose([1,1],np.zeros((0,2)),[1,1],epsilon=1e-8,R=[[1,1]],query_contract='linear')
    assert r['LB']<1e-8
    assert r['status']=='LINEAR_NUMERICAL_WITHIN_BUDGET'
    assert minrank([[1,1]],R=[[1,1]])['rank']==0
    assert minrank([[1,2,3]],R=[[1,2,3]])['rank']==0


def test_two_relaxation_moments_miss_a_third_mode():
    rates=np.array([1.,2.,3.]); S=np.vstack([np.ones(3),rates])
    r=diagnose(rates**2,S,[1,1,1],epsilon=.1,query_contract='linear')
    assert r['LB']>0.1
    assert r['dual_L1_upper']>=r['LB']-1e-9
    assert np.max(abs(S@r['witness']))<1e-8


def test_zero_tangent_cannot_accept_saturation():
    # f(x)=x^2 at0 has zero derivative, but x=0 and1 give answers0 and1.
    r=diagnose([0.],np.zeros((0,1)),[1.],epsilon=.1,query_contract='tangent')
    assert r['LB']==0.
    assert r['status']=='UNKNOWN_NONLINEAR_REMAINDER'
    assert r['outward_certified'] is False


def test_dependent_moments_preserve_constraint_quality():
    S=np.array([[1,1,0],[2,2,0],[1,1,0]])
    r=diagnose([1,1,1],S,[1,1,1],epsilon=.1,query_contract='linear')
    assert r['LB']==pytest.approx(1)
    assert r['numerical_constraint_rank']==1
    assert r['primal_scaled_residual']<1e-8


@pytest.mark.parametrize('radius',[0,-1,np.nan,np.inf])
def test_invalid_physical_uncertainty_contract_refused(radius):
    with pytest.raises(ValueError):
        diagnose([1],[[1]],[radius],epsilon=.1,query_contract='linear')


def test_query_contract_and_payload_equality_are_explicit():
    # A Jacobian row passed without a contract must not default to "linear".
    with pytest.raises(TypeError):
        diagnose([0.],np.zeros((0,1)),[1.],epsilon=.1)
    with pytest.raises(TypeError):
        two_point([0.],[1.],.1)
    assert two_point([0.],[1.],.1,payload_equal=True)['gate']=='IMPOSSIBLE'
    assert two_point([0.],[1.],.1,payload_equal=False)['gate']=='UNKNOWN'
    assert two_point([0.],[.2],.1,payload_equal=True)['gate']=='UNKNOWN'


def test_threshold_rank_flags_possible_underestimate():
    r=minrank(np.diag([1.,1e-12]))
    assert r['rank']==1 and r['rank_sweep']['1e-11']==1
    assert r['may_underestimate_exact_rank'] is True
