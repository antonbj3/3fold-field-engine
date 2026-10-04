from fractions import Fraction as F
import random
import pytest
from field_engine.experimental.guaranteed_energy_difference import (
    projector_difference_bounds,energy_difference_interval,sqrt_upper)


def test_exact_schur_projection_and_metric_change():
    rng=random.Random(61001)
    B=((F(1),F(0)),(F(0),F(1)),(F(1),F(1)))
    def err(A,r):
        K=[[sum(A[k]*B[k][i]*B[k][j] for k in range(3)) for j in range(2)] for i in range(2)]
        f=[sum(B[k][i]*r[k] for k in range(3)) for i in range(2)]
        a,b,c=K[0][0],K[0][1],K[1][1]
        return (c*f[0]**2-2*b*f[0]*f[1]+a*f[1]**2)/(a*c-b*b)
    for _ in range(240):
        A0=[F(rng.randint(1,9),3) for j in range(3)]
        ratios=[F(rng.randint(2,8),5) for j in range(3)]
        A1=[x*y for x,y in zip(A0,ratios)]
        r0=[F(rng.randint(-8,8),7) for j in range(3)]
        r1=[x+F(rng.randint(-3,3),100) for x in r0]
        R0=sum(x*x/a for x,a in zip(r0,A0));R1=sum(x*x/a for x,a in zip(r1,A0))
        C=sum(x*y/a for x,y,a in zip(r0,r1,A0))
        lo,hi=projector_difference_bounds(R0,R1,C,min(F(1),*ratios),max(F(1),*ratios))
        assert lo<=err(A0,r0)-err(A1,r1)<=hi


def test_common_error_exact_cancellation_and_composition():
    huge=F(10**40)
    assert projector_difference_bounds(huge,huge,huge,1,1)==(0,0)
    lo,hi=energy_difference_interval(F(7,3),F(2),huge,huge,huge,1,1,F(1,7))
    assert F(lo)<=F(7,3)<=F(hi)


@pytest.mark.parametrize('args',[(-1,1,0,1,1),(1,1,2,1,1),(1,1,0,0,1),(1,1,0,2,3)])
def test_reject_invalid_gram_or_metric(args):
    with pytest.raises(ValueError):projector_difference_bounds(*args)


def test_reject_zero_step_and_nonexact_input():
    with pytest.raises(ValueError):energy_difference_interval(1,1,1,1,1,1,1,0)
    with pytest.raises(TypeError):projector_difference_bounds('1',1,1,1,1)


def test_square_root_is_an_upper_bound():
    for x in (F(0),F(2),F(1,10**100),F(10**100)):
        assert sqrt_upper(x)**2>=x
