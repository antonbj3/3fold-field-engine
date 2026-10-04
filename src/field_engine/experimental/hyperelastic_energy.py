"""Exact-rational certificates for explicitly scoped Neo-Hookean models.

No float enters the certificate arithmetic. Bounds concern the declared model,
not transverse modes, a general mesh, or measured material uncertainty.
"""
from fractions import Fraction as Q
from math import isqrt


def rational(x):
    if isinstance(x, bool) or not isinstance(x, (int, str, Q)):
        raise TypeError('use exact int, str or Fraction')
    return Q(x)


class I:
    def __init__(self, lo, hi=None):
        self.lo = rational(lo)
        self.hi = self.lo if hi is None else rational(hi)
        if self.lo > self.hi:
            raise ValueError('reversed interval')

    def __add__(self, b):
        b = as_i(b)
        return I(self.lo+b.lo, self.hi+b.hi)
    __radd__ = __add__

    def __neg__(self): return I(-self.hi, -self.lo)
    def __sub__(self, b): return self + -as_i(b)
    def __rsub__(self, b): return as_i(b) + -self

    def __mul__(self, b):
        b = as_i(b)
        values = [x*y for x in (self.lo,self.hi) for y in (b.lo,b.hi)]
        return I(min(values), max(values))
    __rmul__ = __mul__

    def __truediv__(self, b):
        b = as_i(b)
        if b.lo <= 0 <= b.hi: raise ZeroDivisionError('zero in denominator')
        return self * I(1/b.hi, 1/b.lo)
    def __rtruediv__(self, b): return as_i(b)/self

    def __pow__(self, n):
        if not isinstance(n, int): raise TypeError('integer powers only')
        if n < 0: return 1/(self**(-n))
        if n == 0: return I(1)
        if n % 2 == 1: return I(self.lo**n, self.hi**n)
        return I(0 if self.lo <= 0 <= self.hi else min(self.lo**n,self.hi**n),
                 max(self.lo**n,self.hi**n))

    def absmax(self): return max(abs(self.lo),abs(self.hi))
    def encoded(self): return [str(self.lo),str(self.hi)]


def as_i(x): return x if isinstance(x,I) else I(x)


def rounded(x,digits=30):
    """Outward rational decimal compression, never float rounding."""
    scale=10**digits
    low=(x.lo*scale).__floor__()
    high=(x.hi*scale).__ceil__()
    return I(Q(low,scale),Q(high,scale))


def root_point(x, n, digits=32):
    x = rational(x)
    if x < 0 or n not in (2,3): raise ValueError('positive square/cube root only')
    scale = 10**digits
    k = (x.numerator*scale**n)//x.denominator
    if n == 2: lo = isqrt(k)
    else:
        lo,hi = 0,1 << ((k.bit_length()+2)//3)
        while hi-lo > 1:
            mid=(lo+hi)//2
            if mid**3 <= k: lo=mid
            else: hi=mid
    lower=Q(lo,scale)
    return I(lower, lower if lower**n==x else Q(lo+1,scale))


def root(x,n):
    x=as_i(x)
    return I(root_point(x.lo,n).lo,root_point(x.hi,n).hi)


def log_point(x, terms=60):
    """atanh series with exact geometric remainder; argument reduction by 2."""
    x=rational(x)
    if x <= 0: raise ValueError('log outside positive domain')
    k=0
    while x>2: x/=2; k+=1
    while x<1: x*=2; k-=1
    def reduced(t):
        z=(t-1)/(t+1)
        s=sum((2*z**(2*j+1)/Q(2*j+1) for j in range(terms)),Q(0))
        rem=2*z**(2*terms+1)/(Q(2*terms+1)*(1-z*z))
        return I(s,s+rem)
    return rounded(reduced(x)+k*reduced(Q(2)))


def log(x):
    x=as_i(x)
    return I(log_point(x.lo).lo,log_point(x.hi).hi)


def stretch(stress):
    stress=as_i(stress)
    # Monotone inverse P(F)=F-1/F; endpoint evaluation avoids dependency.
    def one(t): return (I(t)+root(I(t*t+4),2))/2
    return I(one(stress.lo).lo,one(stress.hi).hi)


def bar_certificate(stretches, traction='5/6', body=1, *, scope='UNIAXIAL_CONTINUUM'):
    if scope!='UNIAXIAL_CONTINUUM':
        return {'status':'OSAKER','reason':'UNSUPPORTED_TRANSVERSE_OR_3D_MODES'}
    t,f=rational(traction),rational(body)
    fs=[rational(v) for v in stretches]
    if not fs or min(fs)<=0:
        return {'status':'OSAKER','reason':'NONPOSITIVE_STRETCH_OR_EMPTY_MESH'}
    n=len(fs);h=Q(1,n); eta2=Q(0); energy=I(0)
    for j,F in enumerate(fs):
        qmid=t+f*(1-Q(2*j+1,2*n))
        defect=F-1/F-qmid
        # Entire element integral, including unresolved load variation.
        eta2+=h*defect**2+f*f*h**3/12
        energy+=h*((I(F*F)-1)/2-log(F)-qmid*(F-1))
    bound=eta2/2  # W''=1+1/F^2 >= 1 throughout the positive cone.
    return {'status':'CERTIFIED','scope':scope,'n':n,'traction':str(t),'body':str(f),
            'stretches':[str(v) for v in fs],'eta_squared':str(eta2),'coercivity':'1',
            'energy_gap_upper':str(bound),'energy_at_fe':energy.encoded(),
            'minimum_energy':[str(energy.lo-bound),str(energy.hi)]}


def bar_dual_exact(traction='5/6',body=1):
    """Conventional complementary energy, analytically integrated equilibrium stress."""
    t,f=rational(traction),rational(body)
    def conj(q):
        F=stretch(q)
        return (F**2-1)/2+log(F)
    def primitive(q):
        F=stretch(q)
        return F**3/6-F-1/(2*F)+q*log(F)
    integral=conj(t) if f==0 else (primitive(t+f)-primitive(t))/f
    return I(t+f/2)-integral


def sphere_pressure(a):
    """A=1, B=2, mu=1; incompressible radial continuum, outer pressure zero."""
    a=as_i(a)
    if a.lo<=0: raise ValueError('positive radius required')
    b=root(a**3+7,3)
    return 4/b+8/b**4-2/a-1/(2*a**4)


def sphere_slope(a):
    a=as_i(a);b=root(a**3+7,3)
    return 2/a**2+2/a**5-(4/b**2+32/b**5)*a**2/b**2


def arch_residual(y, load):
    y=as_i(y)
    return (y**3-y/25)/(1+y**2)+load


def arch_stiffness(y):
    y=as_i(y)
    return 1-Q(26,25)*(1-y**2)/(1+y**2)**2


def scalar_certificate(model, center, radius, load, *, scope=None):
    """Existence by endpoint signs; positive curvature over the ENTIRE box.

    center is a numerical proposal only, interpreted as an exact rational.
    Energy gap follows from strong convexity; state error from monotonicity.
    """
    x,r,p=map(rational,(center,radius,load))
    allowed={'sphere':'SPHERICAL_INCOMPRESSIBLE_CONTINUUM',
             'arch':'TWO_BAR_CONSTRAINED_APEX'}
    if model not in allowed: raise ValueError('unknown model')
    if scope is not None and scope!=allowed[model]:
        return {'status':'OSAKER','reason':'UNSUPPORTED_MODES','scope':scope}
    if r<=0: raise ValueError('strictly positive radius required')
    box=I(x-r,x+r)
    out={'model':model,'scope':allowed[model],'center':str(x),'radius':str(r),
         'load':str(p),'box':box.encoded(),'status':'OSAKER'}
    try:
        if model=='sphere':
            if box.lo<=0: raise ValueError('nonpositive radius')
            g=lambda z:as_i(z)**2*(sphere_pressure(z)-p)
            curvature=2*box*(sphere_pressure(box)-p)+box**2*sphere_slope(box)
            # State bound uses pressure equation to avoid a^2 dependency.
            residual=(sphere_pressure(x)-p).absmax()
            state_slope=sphere_slope(box).lo
        else:
            g=lambda z:arch_residual(z,p)
            curvature=arch_stiffness(box)
            residual=g(x).absmax();state_slope=curvature.lo
        left,right=g(box.lo),g(box.hi)
        out.update(curvature=curvature.encoded(),endpoint_residuals=[left.encoded(),right.encoded()])
        if curvature.lo<=0 or state_slope<=0:
            out['reason']='NO_POSITIVE_STABILITY_MARGIN';return out
        if left.hi>0 or right.lo<0:
            out['reason']='EXISTENCE_NOT_BRACKETED';return out
        eta=g(x).absmax();gap=eta**2/(2*curvature.lo)
        out.update(status='CERTIFIED',residual_upper=str(eta),
                   energy_gap_upper=str(gap),state_error_upper=str(residual/state_slope),
                   reason='EXISTENCE_AND_POSITIVE_CURVATURE_ON_WHOLE_BOX')
    except (ValueError,ZeroDivisionError): out['reason']='OUTSIDE_ADMISSIBLE_DOMAIN'
    return out


def verify(answer):
    """Replay and bind every model input; a status flag is never trusted."""
    if not isinstance(answer, dict):
        return False
    try:
        if answer.get('scope')=='UNIAXIAL_CONTINUUM':
            fresh=bar_certificate(answer['stretches'],answer['traction'],answer['body'])
        else:
            fresh=scalar_certificate(answer['model'],answer['center'],answer['radius'],
                                     answer['load'],scope=answer['scope'])
        return fresh==answer
    except (KeyError,TypeError,ValueError,ZeroDivisionError): return False
