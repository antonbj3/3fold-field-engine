"""Regressions for nonempty dimensional input and complete integration bounds."""
from fractions import Fraction as Q
import pytest
from field_engine.experimental.homogeneous_boundary import full_zero_trace,integrate,_int_box

ROD={'kind':'rod','length':'10','width':'1','depth':'1'}
@pytest.mark.parametrize('field',[[],[{}],[{},{}],[{},{},{},{}],None])
def test_trace_helper_declines_wrong_component_inventory(field):
    with pytest.raises(ValueError,match='three polynomial'):
        full_zero_trace(field,ROD)

def test_full_integral_with_no_remaining_axes_is_valid():
    assert integrate({(0,0,0):Q(1)},ROD)==10
    assert integrate({(2,0,0):Q(1)},ROD)==Q(1000,3)

def test_unintegrated_nonzero_coordinate_is_rejected():
    with pytest.raises(ValueError,match='unintegrated'):
        _int_box({(1,0,0):Q(1)},{1:(Q(0),Q(1)),2:(Q(0),Q(1))})
