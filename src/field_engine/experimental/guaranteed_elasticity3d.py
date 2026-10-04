"""Experimental continuum certificates for polynomial fields on the unit cube.

Homogeneous Dirichlet data on ALL six faces, constant isotropic E, -1<nu<1/2,
and a polynomial body force are the entire admitted regime.  No mesh, curved
domain, point-stress, contact, or exact-incompressibility claim is made.

Every polynomial coefficient is rational (floats mean their exact dyadic
value).  Symmetry, div(sigma)+f=0, and the displacement trace are checked as
polynomial identities, BEFORE forming any internally owned moments.

For the continuum solution u and C=L(u)=a(u,u),
  max(0,2L(v)-a(v,v)) <= C <= int sigma:S:sigma,
  ||u-v||_a^2 <= a(v,v)-2L(v)+int sigma:S:sigma.
Potential energy has the reverse bounds [-D/2, a(v,v)/2-L(v)].
The deviatoric/volumetric split avoids cancellation in S near nu=1/2;
it does not cure a nonzero divergence defect in the primal.

Energy contractions reuse the reviewed guaranteed_fastcert3d interval
arithmetic. An exact Fraction reserve handles overflow and ambiguous
width decisions. This is a conventional Prager-Synge implementation,
not a new a posteriori theorem or a general stress-reconstruction FE.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math

from .guaranteed_scalar import rat, down, up
from .guaranteed_fastcert3d import IV, _from_rationals, UndecidedEnclosure


def polynomial(value):
    """Canonical immutable polynomial: iterable/dict of ((i,j,k),coef)."""
    items = value.items() if isinstance(value, dict) else value
    result = {}
    for powers, coef in items:
        powers = tuple(powers)
        if len(powers) != 3 or any(type(i) is not int or i < 0 for i in powers):
            raise ValueError('expected three nonnegative integer powers')
        c = rat(coef)
        result[powers] = result.get(powers, F(0)) + c
    return tuple(sorted((p, c) for p, c in result.items() if c))


def add(*polys):
    return polynomial(item for p in polys for item in p)


def scale(poly, factor):
    factor = rat(factor)
    return tuple((p, c*factor) for p, c in poly if c*factor)


def derivative(poly, axis):
    return polynomial((tuple(v-1 if j == axis else v for j, v in enumerate(p)),
                       c*p[axis]) for p, c in poly if p[axis])


def primitive(poly, axis):
    return polynomial((tuple(v+1 if j == axis else v for j, v in enumerate(p)),
                       c/(p[axis]+1)) for p, c in poly)


def product(p, q):
    return polynomial((tuple(a+b for a,b in zip(i,j)), c*d)
                      for i,c in p for j,d in q)


def integral(poly):
    return sum((c/F(math.prod(i+1 for i in p)) for p,c in poly), F(0))


def inner(p, q):
    return sum((c*d/F(math.prod(a+b+1 for a,b in zip(i,j)))
                for i,c in p for j,d in q), F(0))


def strain(v):
    return tuple(tuple(scale(add(derivative(v[i],j), derivative(v[j],i)), F(1,2))
                       for j in range(3)) for i in range(3))


def divergence(sigma):
    return tuple(add(*(derivative(sigma[i][j],j) for j in range(3))) for i in range(3))


def trace(tensor):
    return add(*(tensor[i][i] for i in range(3)))


def deviator(tensor):
    tr = scale(trace(tensor), F(-1,3))
    return tuple(tuple(add(tensor[i][j], tr) if i == j else tensor[i][j]
                       for j in range(3)) for i in range(3))


def norm_squared(tensor):
    return sum((inner(tensor[i][j],tensor[i][j]) for i in range(3) for j in range(3)), F(0))


def moduli(E, nu):
    E, nu = rat(E), rat(nu)
    if E <= 0 or not -1 < nu < F(1,2):
        raise ValueError('require finite E>0 and -1<nu<1/2')
    return E/(2*(1+nu)), E/(3*(1-2*nu))


def reconstruct_stress(body, displacement=None, pressure=(), mu=1):
    """Symmetric global polynomial repair of a proposed mixed stress.

    A diagonal antiderivative of each residual restores equilibrium exactly.
    This operation is admissible ONLY with all-Dirichlet data; it generally
    changes boundary tractions, and is neither a vertex-patch FE nor optimal.
    pressure is a tensile hydrostatic stress (compressive pressure has - sign).
    """
    f = tuple(polynomial(p) for p in body)
    if len(f) != 3:
        raise ValueError('three force components required')
    if displacement is None:
        s = tuple(tuple(() for _ in range(3)) for _ in range(3))
    else:
        v = tuple(polynomial(p) for p in displacement)
        if len(v) != 3:
            raise ValueError('three displacement components required')
        d = deviator(strain(v)); pr = polynomial(pressure)
        s = tuple(tuple(add(scale(d[i][j], 2*rat(mu)), pr if i == j else ())
                        for j in range(3)) for i in range(3))
    residual = tuple(add(f[i], divergence(s)[i]) for i in range(3))
    return tuple(tuple(add(s[i][j], scale(primitive(residual[i],i),-1))
                       if i == j else s[i][j] for j in range(3)) for i in range(3))


def _zero_trace(poly):
    for axis in range(3):
        for endpoint in (0,1):
            value = polynomial((tuple(0 if j == axis else v for j,v in enumerate(p)),
                                c*endpoint**p[axis]) for p,c in poly)
            if value:
                return False
    return True


def _interval_output(lower, upper, gap, mode):
    lo = max(F(0),lower); hi = upper
    if hi < lo or gap < 0:
        raise ArithmeticError('inconsistent certified moments')
    return {'lower': down(lo), 'upper': up(hi), 'error_squared_upper': up(gap),
            'potential_lower': down(-hi/2), 'arithmetic': mode,
            'point_stress': 'OSAKER_UNBOUNDED_FUNCTIONAL',
            'regime': 'UNIT_CUBE_ALL_HOMOGENEOUS_DIRICHLET_POLYNOMIAL_BODY'}


@dataclass(frozen=True, init=False)
class ElasticCertificate:
    """Validated fields with internally computed immutable exact moments."""
    displacement: tuple
    stress: tuple
    body: tuple
    _moments: tuple

    def __init__(self, displacement, stress, body):
        v = tuple(polynomial(p) for p in displacement)
        f = tuple(polynomial(p) for p in body)
        s = tuple(tuple(polynomial(p) for p in row) for row in stress)
        if len(v)!=3 or len(f)!=3 or len(s)!=3 or any(len(row)!=3 for row in s):
            raise ValueError('3D vectors and a 3x3 tensor required')
        if not all(_zero_trace(p) for p in v):
            raise ValueError('nonzero Dirichlet trace')
        if any(s[i][j] != s[j][i] for i in range(3) for j in range(3)):
            raise ValueError('stress is not exactly symmetric')
        if any(add(f[i],divergence(s)[i]) for i in range(3)):
            raise ValueError('stress is not exactly equilibrated')
        e = strain(v); tr_e = trace(e); tr_s = trace(s)
        moments = (norm_squared(deviator(e)), inner(tr_e,tr_e),
                   norm_squared(deviator(s)), inner(tr_s,tr_s),
                   sum((inner(f[i],v[i]) for i in range(3)), F(0)))
        for key,value in [('displacement',v),('stress',s),('body',f),('_moments',moments)]:
            object.__setattr__(self,key,value)

    def exact(self, E, nu):
        mu,kappa = moduli(E,nu)
        vd,vt,sd,st,L = self._moments
        a = 2*mu*vd+kappa*vt
        D = sd/(2*mu)+st/(9*kappa)
        output = _interval_output(2*L-a,D,a-2*L+D,'EXACT_FRACTION')
        output['potential_upper'] = up(a/2-L)
        return output

    def fast(self, E, nu):
        # Convert only small positive moduli, avoiding 1-2*nu cancellation.
        mu,kappa = moduli(E,nu)
        try:
            m,_ = _from_rationals(self._moments)
            mk,_ = _from_rationals((mu,kappa))
            a = 2*mk[0]*m[0]+mk[1]*m[1]
            D = m[2]/(2*mk[0])+m[3]/(9*mk[1])
            al,ah = a.exact_bounds(); dl,dh = D.exact_bounds()
            ll,lh = m[4].exact_bounds()
            output = _interval_output(2*ll-ah,dh,ah-2*ll+dh,'BINARY64_MIDRAD')
            output['potential_upper'] = up(ah/2-ll)
            return output
        except (ArithmeticError,OverflowError,ValueError):
            return self.exact(E,nu)

    def decision(self, E, nu, relative_width=F(1,50)):
        tolerance = rat(relative_width)
        if tolerance < 0:
            raise ValueError('nonnegative width required')
        result = self.fast(E,nu)
        lo,hi = F(result['lower']),F(result['upper'])
        if lo > 0 and hi-lo <= tolerance*lo:
            result['decision']='ACCEPT'; result['fallback']=False
            return result
        # Exact reserve: uncertainty of arithmetic is resolved, field error
        # is not. Do not mistake a width miss for nonexistence of a solution.
        result = self.exact(E,nu)
        mu,kappa = moduli(E,nu); vd,vt,sd,st,L = self._moments
        lo=max(F(0),2*L-2*mu*vd-kappa*vt); hi=sd/(2*mu)+st/(9*kappa)
        result['decision']='ACCEPT' if lo>0 and hi-lo<=tolerance*lo else 'OSAKER'
        result['fallback']=True
        return result

    def material_box(self, E_box, nu_box, reference=None):
        """Continuous enclosure, licensed by positive dev/trace moments.

        The separate extrema of mu and kappa are analytic monotone bounds;
        no assumption of corner extrema of energy/compliance is used.
        Optional reach uses the Loewner spectral-ratio theorem. Intersecting
        two sound intervals needs no classifier and cannot reduce coverage.
        """
        el,eh = map(rat,E_box); nl,nh = map(rat,nu_box)
        if el<=0 or eh<el or nh<nl or not -1<nl<=nh<F(1,2):
            raise ValueError('invalid material box')
        mul=el/(2*(1+nh)); muh=eh/(2*(1+nl))
        kl=el/(3*(1-2*nl)); kh=eh/(3*(1-2*nh))
        vd,vt,sd,st,L = self._moments
        lower=max(F(0),2*L-2*muh*vd-kh*vt)
        upper=sd/(2*mul)+st/(9*kl)
        witness=(lower,upper); reach=None
        if reference is not None:
            e0,n0=map(rat,reference); mu0,k0=moduli(e0,n0)
            alpha=min(mul/mu0,kl/k0); beta=max(muh/mu0,kh/k0)
            a0=2*mu0*vd+k0*vt; d0=sd/(2*mu0)+st/(9*k0)
            reach=(max(F(0),2*L-a0)/beta,d0/alpha)
            lower=max(lower,reach[0]); upper=min(upper,reach[1])
        return {'lower':down(lower),'upper':up(upper),
                'witness_interval':(down(witness[0]),up(witness[1])),
                'reach_interval':None if reach is None else (down(reach[0]),up(reach[1])),
                'license':'POSITIVE_DEVIATORIC_VOLUMETRIC_MOMENTS_AND_LOEWNER_REACH',
                'instance_statistic':{'exact_zero_divergence':vt==0,'exact_zero_stress_trace':st==0},
                'continuous_box':True,'point_stress':'OSAKER_UNBOUNDED_FUNCTIONAL'}


def admit_request(domain='unit_cube', boundary='all_homogeneous_dirichlet',
                  observable='compliance', contact=False, nu=F(49,100)):
    reasons=[]
    if domain!='unit_cube': reasons.append('UNSUPPORTED_GEOMETRY')
    if boundary!='all_homogeneous_dirichlet': reasons.append('UNSUPPORTED_TRACE_TRACTION')
    if observable not in ('compliance','energy_error'): reasons.append('UNBOUNDED_OR_UNSUPPORTED_OBSERVABLE')
    if contact: reasons.append('CONTACT_ADMISSIBILITY_REQUIRED')
    if not -1<rat(nu)<F(1,2): reasons.append('INCOMPRESSIBLE_OR_INVALID_MATERIAL')
    return {'status':'OSAKER' if reasons else 'ADMITTED_REQUIRES_EXACT_FIELD_CHECKS','reasons':reasons}
