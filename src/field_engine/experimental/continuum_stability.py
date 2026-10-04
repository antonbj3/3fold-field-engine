"""Exact witness checks for one incompressible NH rectangular solid.

A negative admissible direction proves continuum instability. A positive
finite-dimensional direction never proves full-space stability. All
arithmetic uses Fraction; coordinates are reference X, Y, Z.
"""
from fractions import Fraction as Q
from itertools import product

MODEL='INCOMPRESSIBLE_NH_RECTANGULAR_CLAMPED_ENDS_V1'

def rational(x):
    if isinstance(x,bool) or not isinstance(x,(int,str,Q)):
        raise TypeError('exact int, rational string or Fraction required')
    return Q(x)

def clean(p): return {tuple(k):v for k,v in p.items() if v}
def add(a,b):
    p=dict(a)
    for k,v in b.items():p[k]=p.get(k,Q(0))+v
    return clean(p)
def scale(a,c): return clean({k:v*c for k,v in a.items()})
def derivative(a,axis):
    p={}
    for k,v in a.items():
        if k[axis]:
            key=list(k);key[axis]-=1;p[tuple(key)]=v*k[axis]
    return p

def multiply(a,b):
    p={}
    for k,v in a.items():
        for l,w in b.items():
            key=tuple(x+y for x,y in zip(k,l));p[key]=p.get(key,Q(0))+v*w
    return clean(p)

def integral(a,L,B,C):
    sides=((Q(0),L),(-B/2,B/2),(-C/2,C/2))
    total=Q(0)
    for k,v in a.items():
        for n,(lo,hi) in zip(k,sides):v*=(hi**(n+1)-lo**(n+1))/Q(n+1)
        total+=v
    return total

def trace(a,x):
    p={}
    for (i,j,k),v in a.items():p[(0,j,k)]=p.get((0,j,k),Q(0))+v*x**i
    return clean(p)

def decode(field):
    if not isinstance(field,list) or len(field)!=3:raise ValueError('three components required')
    out=[]
    for comp in field:
        if not isinstance(comp,dict):raise ValueError('polynomial dictionary required')
        p={}
        for key,value in comp.items():
            if not isinstance(key, str):
                raise ValueError('polynomial powers must be encoded as strings')
            ns=key.split(',')
            if len(ns)!=3:raise ValueError('three powers required')
            powers=tuple(int(n) for n in ns)
            if any(n<0 or str(n)!=s for n,s in zip(powers,ns)):raise ValueError('canonical nonnegative integer powers required')
            if max(powers)>24 or len(comp)>512:raise ValueError('polynomial size gate')
            p[powers]=rational(value)
        out.append(clean(p))
    return out

def encode(field):return [{','.join(map(str,k)):str(v) for k,v in sorted(p.items())} for p in field]

def bilinear(u,v,t,L,B,C,mu=Q(1)):
    f=(t*t,1/t,1/t);pressure=mu/(t*t)
    Hu=[[derivative(u[i],j) for j in range(3)] for i in range(3)]
    Hv=[[derivative(v[i],j) for j in range(3)] for i in range(3)]
    energy=Q(0);grad=Q(0);mass=Q(0)
    for i in range(3):
        mass+=integral(multiply(u[i],v[i]),L,B,C)
        for j in range(3):
            q=integral(multiply(Hu[i][j],Hv[i][j]),L,B,C)
            grad+=q;energy+=mu*q+pressure/(f[i]*f[j])*integral(multiply(Hu[i][j],Hv[j][i]),L,B,C)
    return energy,grad,mass

def rod_certificate(*,t,length,width='1',depth='1',mu='1',field=None):
    t,L,B,C,mu=map(rational,(t,length,width,depth,mu))
    if not (0<t<=1 and L>0 and B>0 and C>0 and mu>0):raise ValueError('positive compressive state and geometry required')
    if field is None:raise ValueError('an admissible field is required')
    u=decode(field);f=(t*t,1/t,1/t)
    if any(trace(comp,x) for comp in u for x in (Q(0),L)):raise ValueError('nonzero end trace')
    div={}
    for i in range(3):div=add(div,scale(derivative(u[i],i),1/f[i]))
    if div:raise ValueError('not tangent-incompressible')
    q,g,m=bilinear(u,u,t,L,B,C,mu)
    if g<=0 or m<=0:raise ValueError('zero perturbation')
    floor=mu*min(Q(0),1-1/t**3)
    assert q>=floor*g
    return {'model':MODEL,'scope':'FULL_3D_TANGENT_SPACE_INSTABILITY_OR_FLOOR','inputs':{'t':str(t),'length':str(L),'width':str(B),'depth':str(C),'mu':str(mu),'field':encode(u)},
        'det_F':'1','pressure':str(mu/t**2),'lateral_piola':'0',
        'nominal_compressive_stress':str(mu*(t**-4-t*t)),
        'equilibrium_residual':'0','tangent_divergence':'0','end_trace':'0',
        'quadratic_variation':str(q),'gradient_norm_squared':str(g),'l2_norm_squared':str(m),
        'full_space_gradient_floor':str(floor),'continuum_gradient_infimum_upper':str(q/g),
        'continuum_l2_infimum_upper':str(q/m),'witness_integration_error':'0',
        'status':'PROVED_UNSTABLE' if q<0 else 'OSAKER',
        'stable_side_spectral_complement':'UNVERIFIED','full_space_positive_coercivity':False}

def verify(cert):
    try:return rod_certificate(**cert['inputs'])==cert
    except (KeyError,TypeError,ValueError,ZeroDivisionError,AssertionError):return False
