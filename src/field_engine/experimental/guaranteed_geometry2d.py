"""Compose the reviewed 2D Poisson certificate with a constructive geometry map.

This consumer handles a positive uniform scale of an admitted reference box only.
The whole continuous scale family is bounded by exact scaling of C=int f*u.
Unsupported SDF/scan geometry returns UNKNOWN before field construction.
"""
from fractions import Fraction as Q
from .geometry_contract import BoxScale, GeometryContract, revalidate, rat
from . import guaranteed_scalar as gs


def _reference_box(mesh,evidence):
    gs._require_admitted(mesh)
    L=tuple(map(rat,evidence.lengths))
    if len(L)!=2:
        raise ValueError('2D box evidence required by this Poisson consumer')
    p=mesh.xy
    if any(not (0<=x<=L[0] and 0<=y<=L[1]) for x,y in p):
        raise ValueError('reference mesh differs from contract box')
    A=sum(gs.orient(*(p[i] for i in t))/2 for t in mesh.triangles)
    if A!=L[0]*L[1]:
        raise ValueError('reference disk must fill the declared box')
    for i,a in enumerate(mesh.boundary):
        b=mesh.boundary[(i+1)%len(mesh.boundary)]
        if not any(p[a][j]==p[b][j]==v for j in (0,1) for v in (Q(0),L[j])):
            raise ValueError('reference boundary does not lie on box faces')
    return L


def poisson_with_geometry(mesh, evidence, contract, *, source=1, coefficient=1,
                          expected_unit='model', expected_pitch=Q(1,4), maximum_geometry_relative_width=Q(1,1000)):
    """Compute a represented interval and a distinct all-geometries interval.

    A contract is always recomputed; dataclasses.replace or a copied hash cannot
    preserve its admission. Physical acquisition remains UNKNOWN. Geometric
    scales must be positive and their explicit topology/resolution must hold.
    Returned geometry widening is an endpoint addition, not an independent
    bound for two arbitrary unknown continuum responses.
    """
    if type(contract) is not GeometryContract:
        raise ValueError('typed geometry contract required')
    fresh=revalidate(contract,evidence)
    if fresh.unit!=expected_unit:
        raise ValueError('consumer length unit mismatch')
    if fresh.pitch!=rat(expected_pitch):
        raise ValueError('consumer grid pitch mismatch')
    if type(evidence) is not BoxScale or len(evidence.lengths)!=2:
        return {'status':'UNKNOWN','reason':'This solver needs an explicit positive 2D box-scale map; distance/topology metadata alone is insufficient',
                'physical_acquisition':'UNKNOWN','solve_executed':False}
    if fresh.topology.status!='VERIFIED' or not fresh.topology.preserved or fresh.feature_resolution!='VERIFIED_SEPARATION_GE_PITCH':
        return {'status':'UNKNOWN','reason':'Unproved topology or unresolved primitive feature separation','solve_executed':False}
    _reference_box(mesh,evidence)
    f,k=rat(source),rat(coefficient)
    if f<=0 or k<=0:
        raise ValueError('positive uniform scalar source/coefficient required')
    a,b=rat(evidence.scale_lower),rat(evidence.scale_upper)
    width=(b**4-a**4)/a**4
    budget=rat(maximum_geometry_relative_width)
    if budget<0:
        raise ValueError('nonnegative geometry budget required')
    if width>budget:
        return {'status':'BUDGET_FAIL','reason':'Entire-family geometry width exceeds frozen budget','solve_executed':False,
                'geometry_relative_width_upper':gs.up(width)}
    v,psi=gs.solve_poisson_fields(mesh,source=f,k=k)
    base=gs.poisson_certificate(mesh,v,psi,source=f,k=k)
    lo,hi=max(Q(0),rat(base['lower'])),rat(base['upper'])
    if lo>hi:
        raise ArithmeticError('ordered nonnegative continuum interval required')
    # Constant source/coefficient and zero Dirichlet data: C(lambda Omega)=lambda^4 C(Omega).
    lower,upper=a**4*lo,b**4*hi
    widening=max(max(Q(0),lo-lower),max(Q(0),upper-hi))
    return {'status':'VERIFIED_MODEL_INTERVAL','quantity':'poisson_compliance',
            'lower':gs.down(lower),'upper':gs.up(upper),
            'reference_lower':gs.down(lo),'reference_upper':gs.up(hi),
            'reference_numerical_width_upper':gs.up(hi-lo),
            'geometry_endpoint_addition_upper':gs.up(widening),
            'geometry_relative_width_upper':gs.up(width),
            'relative_width_upper':gs.up((upper-lower)/lower) if lower>0 else None,
            'geometry':'ALL_POSITIVE_BOX_SCALES_IN_DECLARED_INTERVAL',
            'reference_geometry':'ADMITTED_BOX_MESH',
            'contract_sha256':fresh.evidence_sha256,'physical_acquisition':'UNKNOWN',
            'source_and_coefficient':'UNIFORM_CONSTANT_ON_EACH_TARGET',
            'units':f'f^2 {expected_unit}^4/k','solve_executed':True}
