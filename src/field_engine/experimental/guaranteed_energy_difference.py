"""Algebraic common-projector bounds, conditional on the stated metric hypotheses.

This helper alone does not certify a mesh, PDE, field, or parameter box.
R0/R1/cross must be the exact Gram entries of equilibrated residuals in the
same reference metric, and m*A0 <= A1 <= M*A0 on the homogeneous trial space.
"""
from fractions import Fraction as F
from math import isqrt
from field_engine.experimental.guaranteed_scalar import rat


def sqrt_upper(x,bits=96):
    x=rat(x)
    if x<0:raise ValueError('negative square root')
    scale=1<<bits;k=isqrt(x.numerator*scale*scale//x.denominator)
    if k*k*x.denominator<x.numerator*scale*scale:k+=1
    return F(k,scale)


def projector_difference_bounds(r0_squared,r1_squared,cross,m,M):
    """Return exact rational bounds for Q0-Q1 (squared primal error difference).

    Valid for a common orthogonal projector and the declared metric ordering;
    coefficients are not inferred or verified against any external operator.
    """
    r0,r1,c,m,M=map(rat,(r0_squared,r1_squared,cross,m,M))
    if min(r0,r1)<0 or c*c>r0*r1:raise ValueError('Gram matrix must be positive semidefinite')
    if not 0<m<=1<=M:raise ValueError('require 0 < m <= 1 <= M')
    prod=sqrt_upper((r0+r1+2*c)*(r0+r1-2*c))
    lo=(r0-r1-prod)/2+(1-1/m)*r1
    hi=(r0-r1+prod)/2+(1-1/M)*r1
    return lo,hi


def energy_difference_interval(trial0,trial1,r0_squared,r1_squared,cross,m,M,step):
    """Outward interval for (E0-E1)/step, under the module's hypotheses.

    trial_i is J_i(v_i), including the 1/2 in the energy definition.
    E_i=trial_i-Q_i/2. No monotonicity clipping or physical-model admission
    is inferred from these scalar inputs. This is a conditional algebra API.
    """
    from field_engine.experimental.guaranteed_scalar import down,up
    trial0,trial1,step=map(rat,(trial0,trial1,step))
    if step<=0:raise ValueError('positive step required')
    qlo,qhi=projector_difference_bounds(r0_squared,r1_squared,cross,m,M)
    return down((trial0-trial1-qhi/2)/step),up((trial0-trial1-qlo/2)/step)
