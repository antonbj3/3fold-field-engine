"""Experimental elastic cartilage contact API.

SI inputs: R, t, delta [m], E [Pa], nu dimensionless; outputs F [N], a [m],
pmax [Pa], p(r) callable [Pa]. The sphere is rigid, the layer linear elastic,
isotropic, bonded below and frictionless above. No global 2% certificate is
claimed for a, pmax or the whole pressure curve.
"""
import numpy as np
from numpy.polynomial.chebyshev import chebvander

from .cartilage_contact import LayerContact


def cartilage_contact(R, t, E, nu, delta):
    """Return (F, a, p_max, p(r)) from the corrected Hankel kernel.

    The high quadrature is necessary at t/R=0.02, nu near 0.5. The returned
    pressure callable accepts scalar or array radii. Outside contact it is 0.
    Inputs are m, Pa, dimensionless nu, and m; outputs are N, m, Pa, Pa.
    Rigid sphere on a bonded isotropic linear elastic layer, frictionless top.
    Grid discrepancies are empirical; no continuous-parameter error enclosure
    or verified biphasic pressure curve is returned.
    """
    if not (R > 0 and t > 0 and E > 0 and 0 <= nu < .5 and 0 < delta < R):
        raise ValueError('require R,t,E>0, 0<=nu<0.5 and 0<delta<R')
    solver=LayerContact(R,t,E,nu,K=8,nxi=1800,nu_quad=220)
    result=solver.solve(delta)
    c=result['c'].copy()
    # Smooth-sphere edge coefficient is a numerical root, not a physical
    # flat-punch singularity. Remove its residual from p(r) evaluation.
    c[0]=0.0
    a=result['a']

    def p_of_r(radius):
        rr=np.asarray(radius,dtype=float)
        u=np.clip(rr/a,0,1)
        sq=np.sqrt(np.maximum(1-u*u,0.0))
        p=sq*(chebvander(u,solver.K-1)@c[1:])
        p=np.where((rr>=0)&(rr<=a),p,0.0)
        return p.item() if np.ndim(rr)==0 else p

    sample=p_of_r(np.linspace(0,a,401))
    pmax=float(np.max(sample))
    if not np.isfinite(result['force']) or result['force']<=0 or np.min(sample)<-1e-6*pmax:
        raise ArithmeticError('contact solve violates pressure positivity')
    return float(result['force']),float(a),pmax,p_of_r
