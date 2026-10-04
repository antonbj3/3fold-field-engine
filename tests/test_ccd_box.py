from fractions import Fraction as Q
import math
import numpy as np
import pytest
from field_engine.experimental.ccd_box import (
    Branch, BoxDecision, certify_boxes, support_separated, SAFE, COLLISION, UNKNOWN)

def pt(z=.2):
    return np.array([[[.25,.25,z],[0,0,0],[1,0,0],[0,1,0]]]*2,float)

def branch(x=None,r=None,lo=None,hi=None):
    return Branch.from_arrays(pt() if x is None else x,r,lo,hi)

def test_anisotropy_is_used():
    x=pt(.02);r=np.zeros_like(x);r[:,:,0]=.1
    assert certify_boxes([branch(x,r)],'PT',complete=True).status==SAFE

def test_all_uncertain_moving_faces_cross():
    x=pt();x[1,0,2]=-.2;x[1,1:,0]+=.01
    r=np.full_like(x,.001)
    d=certify_boxes([branch(x,r)],'PT',complete=True)
    assert d.status==COLLISION
    assert d.branches[0].method=='universal_orientation_crossing'

def test_crossing_outside_triangle_is_not_collision():
    x=pt();x[:,0,:2]=[2,2];x[1,0,2]=-.2
    assert certify_boxes([branch(x)],'PT',complete=True).status==SAFE

def test_mixed_motion_branches_stay_unknown():
    x=pt();x[1,0,2]=-.2
    d=certify_boxes([branch(),branch(x)],'PT',complete=True)
    assert d.status==UNKNOWN
    assert {b.status for b in d.branches}=={SAFE,COLLISION}

@pytest.mark.parametrize('complete,bs',[(False,[branch()]),(True,[]),(False,[])])
def test_incomplete_and_empty(complete,bs):
    d=certify_boxes(bs,'PT',complete=complete)
    assert d.status==UNKNOWN and d.safe_prefix==0

def test_every_branch_must_collide():
    x=pt();x[1,0,2]=-.2
    y=x.copy();y[:,0,0]=.3
    assert certify_boxes([branch(x),branch(y)],'PT',complete=True).status==COLLISION

def test_thick_contact_in_every_world():
    x=pt(.001);x[:,0,:2]=[0,0];r=np.full_like(x,.0001)
    lo=np.full((2,4),.002);hi=np.full((2,4),.003)
    d=certify_boxes([branch(x,r,lo,hi)],'PT',complete=True)
    assert d.status==COLLISION and d.branches[0].method=='universal_fixed_time_skin'

def test_skin_interval_straddles_contact():
    x=pt(.01);lo=np.zeros((2,4));hi=np.full((2,4),.02)
    d=certify_boxes([branch(x,lo=lo,hi=hi)],'PT',complete=True,max_cells=32)
    assert d.status==UNKNOWN and d.branches[0].method=='proved_mixed_worlds'

def test_interior_box_contact_refutes_corner_sampling():
    x=pt(0);r=np.zeros_like(x);r[:,0,:]=1
    d=certify_boxes([branch(x,r)],'PT',complete=True,max_cells=16)
    assert d.status==UNKNOWN
    # The interior world is an actual contact; every +/-1 point-box corner is free.
    assert certify_boxes([branch(x)],'PT',complete=True).status==COLLISION

def test_zero_budget_never_accepts():
    d=certify_boxes([branch()],'PT',complete=True,max_cells=0)
    assert d.status==UNKNOWN and d.safe_prefix==0

def test_local_simplex_refinement_changes_decision():
    x=pt(.2);x[:,2,0]=2;x[:,3,1]=2;x[:,0,:2]=[.1,.1]
    r=np.zeros_like(x);r[:,2,2]=1
    b=branch(x,r)
    assert not support_separated(b,'PT')
    d=certify_boxes([b],'PT',complete=True,max_cells=512)
    assert d.status==SAFE and d.safe_prefix==1 and d.branches[0].cells>1

def test_zero_box_reserve_retains_external_base():
    # Boundary hit at t=1/4 defeats strict projected containment and the
    # fixed-time probes 0,1/2,1, so this actually enters the exact reserve.
    x=pt(.25);x[:,0,:2]=0;x[1,0,2]=-.75
    d=certify_boxes([branch(x)],'PT',complete=True)
    assert d.status==COLLISION
    assert d.branches[0].method.startswith('reviewed_zero_box_reserve:')

def test_ee_skin_and_nonintersection():
    x=np.array([[[0,0,0],[1,0,0],[0,.1,0],[1,.1,0]]]*2,float)
    assert certify_boxes([branch(x)],'EE',complete=True).status==SAFE
    lo=np.full((2,4),.1)
    assert certify_boxes([branch(x,lo=lo,hi=lo)],'EE',complete=True).status==COLLISION

def test_input_is_copied_and_exact():
    x=pt();b=branch(x);x[:]=0
    assert b.centers[0][0][2]==Q(.2)
    assert certify_boxes([b],'PT',complete=True).status==SAFE

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf'),True])
def test_bad_coordinates_rejected(bad):
    x=pt().tolist();x[0][0][0]=bad
    with pytest.raises((ValueError,OverflowError)):branch(x)

def test_negative_radius_rejected():
    r=np.zeros((2,4,3));r[0,0,0]=-.1
    with pytest.raises(ValueError):branch(r=r)

def test_inverted_skin_rejected():
    with pytest.raises(ValueError):branch(lo=np.ones((2,4)),hi=np.zeros((2,4)))

@pytest.mark.parametrize('kwargs',[{'max_cells':-1},{'max_depth':-1},{'max_cells':1.5},{'complete':'yes'}])
def test_invalid_budget_and_coverage(kwargs):
    args=dict(complete=True);args.update(kwargs)
    with pytest.raises((ValueError,TypeError)):certify_boxes([branch()],'PT',**args)

def test_kind_and_shapes():
    with pytest.raises(ValueError):certify_boxes([branch()],'TT',complete=True)
    with pytest.raises(ValueError):Branch.from_arrays(np.zeros((2,4,2)))

def test_nonbinary_rational_not_silently_cast_in_reserve():
    x=pt(0).tolist();x[0][0][0]=Q(1,3);x[1][0][0]=Q(1,3)
    d=certify_boxes([branch(x)],'PT',complete=True,max_cells=1)
    assert d.status==COLLISION and d.branches[0].method=='universal_fixed_time_skin'

def test_prefix_rounds_down():
    d=BoxDecision(UNKNOWN,Q(1,3),())
    assert Q(d.prefix_float())<=Q(1,3)

def test_small_large_exact_coordinates():
    for scale in (2.**-500,2.**500):
        x=pt(.2)*scale;r=np.zeros_like(x);r[:,:,0]=scale*.1
        assert certify_boxes([branch(x,r)],'PT',complete=True).status==SAFE

def test_mixed_guard_matches_unguarded_and_bounds_cost():
    x=pt(.01);r=np.zeros_like(x);r[:,0,2]=.02
    b=branch(x,r)
    fast=certify_boxes([b],'PT',complete=True,max_cells=32)
    slow=certify_boxes([b],'PT',complete=True,max_cells=32,prove_mixed=False)
    assert fast.status==slow.status==UNKNOWN
    assert fast.branches[0].method=='proved_mixed_worlds'
    assert fast.branches[0].cells==1 and slow.branches[0].cells>1

def test_time_refinement_retains_a_positive_safe_prefix():
    x=pt(.2);x[1,0,2]=0
    r=np.zeros_like(x);r[1,0,2]=.05
    d=certify_boxes([branch(x,r)],'PT',complete=True,max_cells=128,max_depth=12,prove_mixed=False)
    assert d.status==UNKNOWN and 0<d.safe_prefix<=Q(4,5)

def test_reserve_decline_remains_unknown(monkeypatch):
    from field_engine.experimental import ccd_band
    x=pt(.25);x[:,0,:2]=0;x[1,0,2]=-.75
    calls=[]
    def decline(values,kind):
        calls.append((values,kind))
        return ccd_band.Decision(UNKNOWN,'refinement_budget')
    monkeypatch.setattr(ccd_band,'certify',decline)
    d=certify_boxes([branch(x)],'PT',complete=True)
    assert len(calls)==1
    assert calls[0][1]=='PT'
    assert d.status==UNKNOWN
    assert d.safe_prefix==0
    assert d.branches[0].method=='reviewed_zero_box_reserve:refinement_budget'

def test_collapsed_primitives_are_not_false_safe():
    # The point meets a fully collapsed face at 1/4; unsupported coplanar
    # motion must refuse, without a vacuous empty-family acceptance.
    x=np.zeros((2,4,3));x[0,0,2]=.25;x[1,0,2]=-.75
    d=certify_boxes([branch(x)],'PT',complete=True)
    assert d.status==UNKNOWN
    assert d.safe_prefix==0

def test_simplex_split_covers_both_halves():
    from field_engine.experimental.ccd_box import _split_domain,_root_domains
    parent=_root_domains('PT')[1]
    a,b=_split_domain(parent)
    assert len(set(a)&set(b))==2
    assert set(parent).issubset(set(a)|set(b))
    assert all(sum(v)==1 and min(v)>=0 for v in a+b)

@pytest.mark.parametrize('wrap',[np.int64,lambda x:Q(np.int64(x))])
def test_fixed_width_rationals_are_normalized_before_geometry(wrap):
    from field_engine.experimental.ccd_box import exact
    value=exact(wrap(2**62))
    assert type(value.numerator) is int
    assert value*4==Q(2**64)
    x=np.zeros((2,4,3),dtype=np.int64).tolist();x[0][0][2]=wrap(2**62);x[1][0][2]=wrap(2**62)
    b=branch(x)
    assert type(b.centers[0][0][2].numerator) is int
    assert certify_boxes([b],'PT',complete=True).status==SAFE
