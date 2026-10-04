"""The final energy interval must include nonzero projected error energies."""
from fractions import Fraction as F
import pytest
from field_engine.experimental.guaranteed_energy_difference import energy_difference_interval


@pytest.mark.parametrize('args,truth',[
    # A0=A1=I; homogeneous gradients span e1. Residuals are e1 and e2,
    # so Q0=1,Q1=0. True cross=0, not inferred from equal marginal norms.
    ((0,0,1,1,0,1,1,1),F(-1,2)),
    # A0=I,A1=diag(2,1); same residual e1, Q0=1,Q1=1/2.
    ((0,0,1,1,1,1,2,1),F(-1,4)),
    ((F(7,3),F(2),1,1,0,1,1,F(1,1000000)),F(-1,6)*1000000),
])
def test_composition_encloses_independent_exact_projected_energies(args,truth):
    lo,hi=energy_difference_interval(*args)
    assert F(lo)<=truth<=F(hi)


@pytest.mark.parametrize('sign',(-1,1))
def test_outward_rounding_below_binary64_subnormal_range(sign):
    exact=F(sign,10**500)
    lo,hi=energy_difference_interval(exact,0,0,0,0,1,1,1)
    assert F(lo)<=exact<=F(hi)
