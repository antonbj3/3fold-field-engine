"""Exact NH boundary identities for two fixed homogeneous solid families.

Positive bulk stiffness is a statement about the zero-full-trace kernel.
It must never be reported as full mixed-boundary stability. Curved-arch
tractions are manufactured dead reference tractions, not a crown force.
"""
from fractions import Fraction as Q
from math import comb
from .continuum_stability import (
    rational, clean, add, scale, derivative, multiply, decode, encode,
)

MODEL = 'HOMOGENEOUS_INCOMPRESSIBLE_NH_BOUNDARY_IDENTITY_V1'

def geometry_values(geometry):
    if not isinstance(geometry, dict):
        raise ValueError('exact geometry mapping required')
    kind = geometry.get('kind')
    if kind == 'rod' and set(geometry) == {'kind','length','width','depth'}:
        L,B,C = map(rational,(geometry['length'],geometry['width'],geometry['depth']))
        if min(L,B,C) <= 0:
            raise ValueError('positive rod dimensions required')
        return kind,(L,B,C)
    if kind == 'parabolic_arch' and set(geometry) == {'kind','thickness','depth'}:
        d,C = map(rational,(geometry['thickness'],geometry['depth']))
        if min(d,C) <= 0:
            raise ValueError('positive arch dimensions required')
        return kind,(d,C)
    raise ValueError('unsupported geometry or extra fields')

def canonical_geometry(geometry):
    kind,values = geometry_values(geometry)
    keys = ('length','width','depth') if kind=='rod' else ('thickness','depth')
    return {'kind':kind,**{k:str(v) for k,v in zip(keys,values)}}

def _power(p,n):
    result = {(0,0,0):Q(1)}
    for _ in range(n):
        result = multiply(result,p)
    return result

def substitute(p,axis,value):
    result = {}
    for powers,c in p.items():
        key=list(powers); n=key[axis]; key[axis]=0
        result=add(result,multiply({tuple(key):c},_power(value,n)))
    return result

def restrict(p,axis,value):
    return substitute(p,axis,{(0,0,0):value})

def _int_box(p,bounds):
    total=Q(0)
    for powers,c in p.items():
        for axis,(lo,hi) in bounds.items():
            n=powers[axis]
            c *= (hi**(n+1)-lo**(n+1))/Q(n+1)
        if any(a not in bounds and powers[a] for a in range(3)):
            raise ValueError('unintegrated variable')
        total += c
    return total

def _arch_xy_moment(i,j,d):
    n=j+1
    total=Q(0)
    for k in range(n+1):
        exponent=i+2*k
        if exponent%2==0:
            c=Q(comb(n,k)*(-1)**k,n)*((1+d)**(n-k)-1)
            total += c*Q(2,exponent+1)
    return total

def integrate(p,geometry):
    kind,values=geometry_values(geometry)
    if kind=='rod':
        L,B,C=values
        return _int_box(p,{0:(Q(0),L),1:(-B/2,B/2),2:(-C/2,C/2)})
    d,C=values
    total=Q(0)
    for (i,j,k),c in p.items():
        if k%2==0:
            total+=c*_arch_xy_moment(i,j,d)*2*(C/2)**(k+1)/Q(k+1)
    return total

def surface_flux(vector,geometry):
    kind,values=geometry_values(geometry)
    total=Q(0)
    if kind=='rod':
        L,B,C=values
        bounds={0:(Q(0),L),1:(-B/2,B/2),2:(-C/2,C/2)}
        for axis in range(3):
            other={k:v for k,v in bounds.items() if k!=axis}
            lo,hi=bounds[axis]
            total+=_int_box(restrict(vector[axis],axis,hi),other)
            total-=_int_box(restrict(vector[axis],axis,lo),other)
        return total
    d,C=values
    for x,sign in ((Q(-1),-1),(Q(1),1)):
        total+=sign*_int_box(restrict(vector[0],0,x),{1:(Q(0),d),2:(-C/2,C/2)})
    for offset,sign in ((Q(0),-1),(d,1)):
        curve={(0,0,0):1+offset,(2,0,0):Q(-1)}
        # Upward normal times surface measure is (2x,1,0) dx dz.
        fx=substitute(vector[0],1,curve)
        fy=substitute(vector[1],1,curve)
        face=add(multiply({(1,0,0):Q(2)},fx),fy)
        total+=sign*_int_box(face,{0:(Q(-1),Q(1)),2:(-C/2,C/2)})
    for z,sign in ((-C/2,-1),(C/2,1)):
        face=restrict(vector[2],2,z)
        total+=sign*sum(c*_arch_xy_moment(i,j,d) for (i,j,k),c in face.items())
    return total

def full_zero_trace(field,geometry):
    if not isinstance(field, (list, tuple)) or len(field) != 3:
        raise ValueError("three polynomial components required")
    kind,values=geometry_values(geometry)
    if kind=='rod':
        L,B,C=values
        bounds=((0,L),(-B/2,B/2),(-C/2,C/2))
        return all(not restrict(p,a,x) for p in field for a,(lo,hi) in enumerate(bounds) for x in (lo,hi))
    d,C=values
    return all(
        not restrict(p,0,x) for p in field for x in (Q(-1),Q(1))
    ) and all(
        not restrict(p,2,z) for p in field for z in (-C/2,C/2)
    ) and all(
        not substitute(p,1,{(0,0,0):1+offset,(2,0,0):Q(-1)})
        for p in field for offset in (Q(0),d)
    )

def bilinear(u,v,t,geometry,mu=Q(1)):
    f=(t*t,1/t,1/t)
    Hu=[[derivative(u[i],j) for j in range(3)] for i in range(3)]
    Hv=[[derivative(v[i],j) for j in range(3)] for i in range(3)]
    g=Q(0); pressure_part=Q(0); mass=Q(0)
    for i in range(3):
        mass+=integrate(multiply(u[i],v[i]),geometry)
        for j in range(3):
            g+=integrate(multiply(Hu[i][j],Hv[i][j]),geometry)
            pressure_part+=integrate(multiply(Hu[i][j],Hv[j][i]),geometry)/(f[i]*f[j])
    return mu*g+mu/t**2*pressure_part,g,mass,pressure_part

def boundary_certificate(*,t,geometry,field,mu='1'):
    t,mu=map(rational,(t,mu))
    if not (0<t<=1 and mu>0):
        raise ValueError('positive compressive homogeneous state required')
    geometry=canonical_geometry(geometry)
    kind,values=geometry_values(geometry)
    u=decode(field)
    ends=(Q(0),values[0]) if kind=='rod' else (Q(-1),Q(1))
    if any(restrict(p,0,x) for p in u for x in ends):
        raise ValueError('nonzero end trace')
    f=(t*t,1/t,1/t)
    w=[scale(u[i],1/f[i]) for i in range(3)]
    div={}
    for i in range(3):
        div=add(div,derivative(w[i],i))
    if div:
        raise ValueError('not tangent-incompressible')
    q,g,m,volume=bilinear(u,u,t,geometry,mu)
    if g<=0 or m<=0:
        raise ValueError('nonzero field required')
    flux=[{} for _ in range(3)]
    for j in range(3):
        for i in range(3):
            flux[j]=add(flux[j],multiply(w[i],derivative(w[j],i)))
    boundary=surface_flux(flux,geometry)
    if boundary!=volume:
        raise ArithmeticError('null Lagrangian identity failed')
    zero_trace=full_zero_trace(u,geometry)
    if zero_trace and (volume!=0 or q!=mu*g):
        raise ArithmeticError('bulk identity failed')
    floor=mu*min(Q(0),1-t**-3)
    if q<floor*g:
        raise ArithmeticError('pointwise lower bound failed')
    return {
        'model':MODEL,
        'inputs':{'t':str(t),'mu':str(mu),'geometry':geometry,'field':encode(u)},
        'equilibrium_residual':'0','det_F':'1','pressure':str(mu/t**2),
        'traction_contract':'FREE_LATERAL_SIDES' if kind=='rod' else 'MANUFACTURED_DEAD_PIOLA_TRACTIONS',
        'tangent_divergence':'0','end_trace':'0','quadratic_variation':str(q),
        'gradient_norm_squared':str(g),'l2_norm_squared':str(m),
        'pressure_volume_form':str(volume),'pressure_boundary_flux':str(boundary),
        'witness_integration_error':'0','continuum_gradient_infimum_upper':str(q/g),
        'full_space_gradient_floor':str(floor),'full_zero_trace_field':zero_trace,
        'bulk_kernel_gradient_floor':str(mu),'bulk_kernel_scope':'EXACT_TANGENT_FIELDS_WITH_ZERO_TRACE_ON_ENTIRE_BOUNDARY',
        'remaining_boundary_spectrum':'UNVERIFIED','full_space_positive_coercivity':False,
        'status':'PROVED_UNSTABLE' if q<0 else 'OSAKER',
    }

def verify_boundary_certificate(cert):
    if not isinstance(cert,dict):
        return False
    try:
        return boundary_certificate(**cert['inputs'])==cert
    except (KeyError,TypeError,ValueError,ZeroDivisionError,ArithmeticError,AttributeError):
        return False
