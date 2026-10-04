"""Sign, indexing and scale contracts for shared sampled-SDF crossings."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/field_engine'))


def test_link_theta_and_back_node_in_both_sign_conventions():
    from sdf_rand_v1 import sdf_randlankar
    sd=np.array([2.,-.25,-1.25,2.])
    inside,links=sdf_randlankar(sd,[[1],[-1]])
    idx,theta,back,ok=links[1]
    assert idx.tolist()==[1] and back.tolist()==[2] and ok.tolist()==[True]
    assert theta[0]==pytest.approx(.25/2.25)
    inside2,links2=sdf_randlankar(-sd,[[1],[-1]],negativ_insida=False)
    assert np.array_equal(inside,inside2)
    assert np.array_equal(theta,links2[1][1])


def test_no_wrap_at_nonperiodic_edge():
    from sdf_rand_v1 import sdf_randlankar
    _,links=sdf_randlankar(np.array([-1.,-1.,1.]),[[-1]])
    assert len(links[0][0])==0
    _,links=sdf_randlankar(np.array([-1.,-1.,1.]),[[-1]],periodisk=True)
    assert links[0][0].tolist()==[0]


@pytest.mark.parametrize('scale',[1e-200,1.,1e200])
def test_fraction_is_scale_invariant_without_overflow(scale):
    from sdf_rand_v1 import sdf_randlankar
    _,links=sdf_randlankar(np.array([-1.,3.])*scale,[[1]])
    assert links[0][1][0]==pytest.approx(.25)


@pytest.mark.parametrize('direction',[[[.5]],[[1,0]],[[np.nan]]])
def test_bad_directions_are_refused(direction):
    from sdf_rand_v1 import sdf_randlankar
    with pytest.raises(ValueError):sdf_randlankar(np.array([-1.,1.]),direction)
