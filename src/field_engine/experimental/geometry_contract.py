"""Recomputed geometry bounds for explicit models, with fail-closed UNKNOWN.

Proof producers here cover primitive solids, corresponding triangle surfaces and
trilinear SDFs on a covered box. A numeric bound or caller-supplied flag is not a
proof. Physical acquisition and contact branch state are separate unknowns.
Normals on polyhedra are facewise, excluding the measure-zero ridges. A feature
separation >= pitch does not itself prove topology of a digitized reconstruction.
The length unit is the caller's declaration, not a measured or verified fact.
"""
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product
import hashlib
import json
import math
import numbers


def rat(x):
    if isinstance(x, Q):
        return x
    if isinstance(x, (bool, str, bytes)) or type(x).__name__ == 'bool_':
        raise TypeError('exact numeric scalar required')
    if isinstance(x, numbers.Integral):
        return Q(int(x))
    y = float(x)
    if not math.isfinite(y):
        raise ValueError('nonfinite scalar')
    ratio = getattr(x, 'as_integer_ratio', None)
    if ratio is None or Q(*ratio()) != Q.from_float(y):
        raise ValueError('no silent binary64 rounding; pass a Fraction')
    return Q.from_float(y)


def sqrt_bounds(x, bits=96):
    x = rat(x)
    if x < 0:
        raise ValueError('negative square root')
    if x > 0:
        bits = max(bits, bits + (x.denominator.bit_length()-x.numerator.bit_length()+2)//2)
    k = math.isqrt((x.numerator << (2*bits)) // x.denominator)
    lo = Q(k, 1 << bits)
    hi = lo if lo*lo == x else Q(k+1, 1 << bits)
    assert lo*lo <= x <= hi*hi
    return lo, hi


@lru_cache(maxsize=1)
def pi_bounds():
    def atan(z, n=40):
        terms = [(-1)**k*z**(2*k+1)/Q(2*k+1) for k in range(n)]
        s = sum(terms, Q(0)); nxt = (-1)**n*z**(2*n+1)/Q(2*n+1)
        return min(s, s+nxt), max(s, s+nxt)
    a,b = atan(Q(1,5));c,d = atan(Q(1,239))
    return 16*a-4*d, 16*b-4*c


def _canon(x):
    if is_dataclass(x):
        return {'type':type(x).__name__, **{f.name:_canon(getattr(x,f.name)) for f in fields(x)}}
    if isinstance(x, (tuple,list)):
        return [_canon(v) for v in x]
    if isinstance(x, dict):
        return {str(k):_canon(v) for k,v in x.items()}
    if x is None or isinstance(x, (str,bool)):
        return x
    return str(rat(x))


def fingerprint(x):
    return hashlib.sha256(json.dumps(_canon(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class Bound:
    status: str
    lower: object = None
    upper: object = None
    unit: str = ''
    reason: str = ''


def verified(lo, hi, unit, reason):
    lo,hi=rat(lo),rat(hi)
    if lo>hi:
        raise ValueError('ordered bound required')
    return Bound('VERIFIED',lo,hi,unit,reason)


def unknown(unit, reason):
    return Bound('UNKNOWN',unit=unit,reason=reason)


def na(unit, reason):
    return Bound('NOT_APPLICABLE',unit=unit,reason=reason)


def error(hi, unit, reason):
    return verified(0,hi,unit,reason)


@dataclass(frozen=True)
class Topology:
    status: str = 'UNKNOWN'
    solid_betti: object = None
    preserved: object = None
    electrode_connected: object = None
    reason: str = 'No verified solid/completion/reconstruction proof'


@dataclass(frozen=True)
class BoxScale:
    lengths: tuple
    scale_lower: object = 1
    scale_upper: object = 1
    unit: str = 'model'


@dataclass(frozen=True)
class SphereOffset:
    nominal_radius: object
    radius_lower: object
    radius_upper: object
    unit: str = 'model'


@dataclass(frozen=True)
class AxisSlabs:
    """Union [a,b] x [0,Ly] x [0,Lz], relative to the full enclosing box."""
    intervals: tuple
    transverse_lengths: tuple = (1,1)
    unit: str = 'model'


@dataclass(frozen=True)
class MeshPair:
    reference_vertices: tuple
    target_vertices: tuple
    reference_faces: tuple
    target_faces: tuple
    unit: str = 'model'


@dataclass(frozen=True)
class SDFGrid:
    axes: tuple
    samples: tuple
    reference: AxisSlabs
    unit: str = 'model'


@dataclass(frozen=True)
class Unverified:
    representation: str
    digest: str = ''
    unit: str = 'model'
    reason: str = 'No supported global proof; samples/metadata are insufficient'


@dataclass(frozen=True)
class GeometryContract:
    evidence_sha256: str
    representation: str
    dimension: int
    unit: str
    distance: Bound
    normal_chord: Bound
    normal_angle_rad: Bound
    surface_measure: Bound
    volume_measure: Bound
    area_relative_error: Bound
    volume_relative_error: Bound
    minimum_solid_feature: Bound
    minimum_void_channel: Bound
    topology: Topology
    pitch: object
    feature_resolution: str
    distance_metric: str
    physical_acquisition: str = 'UNKNOWN'
    contact_branch: str = 'UNKNOWN'
    reconstruction_topology: str = 'UNKNOWN'

    def as_dict(self):
        return _canon(self)


def _sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)),Q(0))


def _unit(unit):
    if unit not in ('m','mm','model','source_unit_undocumented'):
        raise ValueError('explicit supported length unit required')
    return unit


def _relative(lo,hi):
    return max(abs(lo-1),abs(hi-1))


def _resolution(wall,channel,pitch):
    b=[v for v in (wall,channel) if v.status!='NOT_APPLICABLE']
    if any(v.status!='VERIFIED' for v in b):
        return 'UNKNOWN'
    return 'VERIFIED_SEPARATION_GE_PITCH' if all(v.lower>=pitch for v in b) else 'SUBGRID_FEATURE'


def verify_geometry(evidence, *, pitch):
    """Derive every field afresh. UNKNOWN is not an empty or zero-error region."""
    pitch=rat(pitch)
    if pitch<=0:
        raise ValueError('positive pitch required')
    unit=_unit(evidence.unit); kind=type(evidence).__name__; dim=3
    dist=unknown(unit,'No global correspondence/distance proof')
    normal=unknown('1','No verified oriented derivative/correspondence proof')
    area=unknown(unit+'^2','No verified surface measure')
    vol=unknown(unit+'^3','No verified solid or completion')
    ae=unknown('1','No verified area correspondence')
    ve=unknown('1','No verified volume correspondence')
    wall=unknown(unit,'No global minimum feature proof')
    channel=unknown(unit,'No global channel proof')
    topo=Topology();metric='UNKNOWN';reconstruction='UNKNOWN'
    if type(evidence) is BoxScale:
        L=tuple(map(rat,evidence.lengths));dim=len(L)
        a,b=rat(evidence.scale_lower),rat(evidence.scale_upper)
        if dim not in (2,3) or min(L)<=0 or not 0<a<=b:
            raise ValueError('positive 2D/3D box and ordered positive scale required')
        delta=max(abs(a-1),abs(b-1))
        dist=error(delta*sqrt_bounds(sum(x*x for x in L))[1],unit,'Global homothety correspondence')
        normal=error(0,'1','Positive scale preserves each oriented face normal')
        V=math.prod(L);A=2*sum(math.prod(L[:i]+L[i+1:]) for i in range(dim))
        area=verified(A*a**(dim-1),A*b**(dim-1),unit+f'^{dim-1}','Exact box boundary measure')
        vol=verified(V*a**dim,V*b**dim,unit+f'^{dim}','Exact box d-volume')
        ae=error(_relative(a**(dim-1),b**(dim-1)),'1','Exact surface scaling')
        ve=error(_relative(a**dim,b**dim),'1','Exact volume scaling')
        wall=verified(min(L)*a,min(L)*b,unit,'Minimum separation of canonical opposing box faces; no introduced neck')
        channel=na(unit,'No internal void channel in this primitive family')
        topo=Topology('VERIFIED',(1,0,0) if dim==3 else (1,0),True,True,'Explicit globally invertible positive homothety; full opposing electrodes transported')
        metric='GLOBAL_BOUNDARY_HAUSDORFF_UPPER';reconstruction='NOT_APPLICABLE_EXPLICIT_SOLID'
    elif type(evidence) is SphereOffset:
        r,a,b=map(rat,(evidence.nominal_radius,evidence.radius_lower,evidence.radius_upper))
        if not 0<a<=b or r<=0:
            raise ValueError('positive ordered sphere radii required')
        p,q=pi_bounds();dist=error(max(abs(a-r),abs(b-r)),unit,'Global radial correspondence; exact signed-distance L-infinity also bounded')
        normal=error(0,'1','Radial oriented normals coincide')
        area=verified(4*p*a*a,4*q*b*b,unit+'^2','Machin pi enclosure and exact radii')
        vol=verified(4*p*a**3/3,4*q*b**3/3,unit+'^3','Machin pi enclosure and exact radii')
        ae=error(_relative((a/r)**2,(b/r)**2),'1','Exact area ratio')
        ve=error(_relative((a/r)**3,(b/r)**3),'1','Exact volume ratio')
        wall=verified(2*a,2*b,unit,'Sphere diameter; no internal neck/wall introduced')
        channel=na(unit,'Solid balls have no internal void channel')
        topo=Topology('VERIFIED',(1,0,0),True,None,'Positive concentric radii; boundary reach is at least radius_lower')
        metric='GLOBAL_BOUNDARY_HAUSDORFF_UPPER';reconstruction='NOT_APPLICABLE_EXPLICIT_SOLID'
    elif type(evidence) is AxisSlabs:
        intervals,Ly,Lz=_slabs(evidence);lo,hi=intervals[0][0],intervals[-1][1]
        gaps=[intervals[i+1][0]-intervals[i][1] for i in range(len(intervals)-1)]
        inner=[x for ab in intervals for x in ab if lo<x<hi]
        delta=max([Q(0)]+[g/2 for g in gaps]+[min(x-lo,hi-x) for x in inner])
        dist=error(delta,unit,'All added interior faces and missing exterior strips bounded globally')
        widths=[b-a for a,b in intervals];V=sum(widths)*Ly*Lz
        A=sum(2*(w*Ly+w*Lz+Ly*Lz) for w in widths)
        A0=2*((hi-lo)*(Ly+Lz)+Ly*Lz);V0=(hi-lo)*Ly*Lz
        area=verified(A,A,unit+'^2','Sum of disjoint box boundary areas')
        vol=verified(V,V,unit+'^3','Sum of disjoint box volumes')
        ae=error(abs(A/A0-1),'1','Area includes every hidden separating face')
        ve=error(abs(V/V0-1),'1','Exact lost solid volume')
        w=min(widths+[Ly,Lz]);wall=verified(w,w,unit,'Minimum canonical solid slab separation')
        channel=verified(min(gaps),min(gaps),unit,'Every void gap enumerated') if gaps else na(unit,'No internal void gap')
        topo=Topology('VERIFIED',(len(intervals),0,0),len(intervals)==1,len(intervals)==1,'Exact disjoint boxes; left/right full electrodes at enclosing endpoints')
        if len(intervals)==1:
            normal=error(0,'1','Identical enclosing box')
        metric='GLOBAL_BOUNDARY_HAUSDORFF_UPPER';reconstruction='NOT_APPLICABLE_EXPLICIT_SOLID'
    elif type(evidence) is MeshPair:
        dist,normal,area,ae=_mesh_pair(evidence)
        metric='GLOBAL_BOUNDARY_HAUSDORFF_UPPER'
        topo=Topology(reason='Surface correspondence gives no solid embedding/completion proof; open scans remain UNKNOWN')
    elif type(evidence) is SDFGrid:
        dist,vol,ve=_sdf_grid(evidence)
        metric='SIGNED_FIELD_LINF_ON_COVERED_BOX'
        topo=Topology(reason='Reference topology is not a proof for the trilinear zero set; no reconstruction theorem')
    elif type(evidence) is Unverified:
        if evidence.representation not in ('MESH','BREP','SDF','SCAN'):
            raise ValueError('unsupported representation')
        kind=evidence.representation
    else:
        raise TypeError('unsupported proof producer')
    angle=error(min(Q(22,7),Q(11,7)*normal.upper),'rad','theta <= pi/2 times chord, pi <=22/7') if normal.status=='VERIFIED' else unknown('rad',normal.reason)
    return GeometryContract(fingerprint(evidence),kind,dim,unit,dist,normal,angle,area,vol,ae,ve,wall,channel,topo,pitch,_resolution(wall,channel,pitch),metric,reconstruction_topology=reconstruction)


def _slabs(e):
    intervals=tuple(tuple(map(rat,ab)) for ab in e.intervals)
    if not intervals or any(len(ab)!=2 or ab[0]>=ab[1] for ab in intervals):
        raise ValueError('nonempty positive intervals required')
    if any(a[1]>=b[0] for a,b in zip(intervals,intervals[1:])):
        raise ValueError('strictly separated ordered slabs required')
    L=tuple(map(rat,e.transverse_lengths))
    if len(L)!=2 or min(L)<=0:
        raise ValueError('positive transverse lengths required')
    return intervals,*L


def _mesh_pair(e):
    p=tuple(tuple(map(rat,v)) for v in e.reference_vertices)
    q=tuple(tuple(map(rat,v)) for v in e.target_vertices)
    if not p or len(p)!=len(q) or any(len(v)!=3 for v in p+q):
        raise ValueError('corresponding 3D vertices required')
    faces=[]
    for fs in (e.reference_faces,e.target_faces):
        out=[]
        for face in fs:
            if len(face)!=3 or any(isinstance(i,bool) or not isinstance(i,numbers.Integral) for i in face):
                raise ValueError('integer triangle indices required')
            t=tuple(int(i) for i in face)
            if len(set(t))!=3 or min(t)<0 or max(t)>=len(p):
                raise ValueError('invalid triangle')
            out.append(t)
        if not out or len({frozenset(t) for t in out})!=len(out):
            raise ValueError('distinct nonempty triangles required')
        faces.append(tuple(out))
    f,g=faces
    if len(f)!=len(g) or any(set(a)!=set(b) for a,b in zip(f,g)):
        raise ValueError('corresponding unoriented triangles required')
    A0lo=A0hi=A1lo=A1hi=Q(0);c2=Q(0)
    for t,u in zip(f,g):
        c=_cross(_sub(p[t[1]],p[t[0]]),_sub(p[t[2]],p[t[0]]))
        d=_cross(_sub(q[u[1]],q[u[0]]),_sub(q[u[2]],q[u[0]]))
        s,t2=_dot(c,c),_dot(d,d)
        if min(s,t2)<=0:
            raise ValueError('degenerate triangle')
        a,b=sqrt_bounds(s);aa,bb=sqrt_bounds(t2)
        A0lo+=a/2;A0hi+=b/2;A1lo+=aa/2;A1hi+=bb/2
        num=_dot(c,d);denlo,denhi=sqrt_bounds(s*t2)
        coslo=num/(denhi if num>=0 else denlo)
        c2=max(c2,min(Q(4),max(Q(0),2-2*coslo)))
    delta=max(sqrt_bounds(_dot(_sub(a,b),_sub(a,b)))[1] for a,b in zip(p,q))
    # Vertex correspondence extends barycentrically over every full triangle.
    return (error(delta,e.unit,'Exact corresponding vertices plus convexity on each triangle'),
            error(sqrt_bounds(c2)[1],'1','Exact oriented cross products on every corresponding triangle'),
            verified(A1lo,A1hi,e.unit+'^2','Every triangle area enclosed by exact square checks'),
            error(_relative(A1lo/A0hi,A1hi/A0lo),'1','Enclosed complete-surface area ratio'))


def _sdf_grid(e):
    if type(e.reference) is not AxisSlabs or e.reference.unit!=e.unit:
        raise ValueError('matching exact slab reference and unit required')
    intervals,Ly,Lz=_slabs(e.reference)
    axes=tuple(tuple(map(rat,ax)) for ax in e.axes)
    if len(axes)!=3 or any(len(ax)<2 for ax in axes):
        raise ValueError('3D covered grid required')
    steps=[]
    for ax in axes:
        diffs=[b-a for a,b in zip(ax,ax[1:])]
        if min(diffs)<=0 or len(set(diffs))!=1:
            raise ValueError('uniform positive axis steps required')
        steps.append(diffs[0])
    extent=((intervals[0][0],intervals[-1][1]),(0,Ly),(0,Lz))
    if any(ax[0]>a or ax[-1]<b for ax,(a,b) in zip(axes,extent)):
        raise ValueError('reference solid must lie in covered box')
    vals=tuple(map(rat,e.samples));shape=tuple(len(ax) for ax in axes)
    if len(vals)!=math.prod(shape):
        raise ValueError('grid samples shape mismatch')
    boxes=[((a,b),(Q(0),Ly),(Q(0),Lz)) for a,b in intervals]
    node_err=Q(0)
    for i,xyz in enumerate(product(*axes)):
        nearest=None
        for box in boxes:
            if all(a<=x<=b for x,(a,b) in zip(xyz,box)):
                d=-min(v for x,(a,b) in zip(xyz,box) for v in (x-a,b-x));dl=du=d
            else:
                d2=sum(max(a-x,Q(0),x-b)**2 for x,(a,b) in zip(xyz,box));dl,du=sqrt_bounds(d2)
            if nearest is None:
                nearest=(dl,du)
            else:
                nearest=(min(nearest[0],dl),min(nearest[1],du))
        node_err=max(node_err,abs(vals[i]-nearest[0]),abs(vals[i]-nearest[1]))
    rest=sqrt_bounds(sum(h*h for h in steps))[1]/2
    def at(i,j,k):
        return vals[(i*shape[1]+j)*shape[2]+k]
    countlo=counthi=0
    for i,j,k in product(*(range(n-1) for n in shape)):
        vv=[at(i+a,j+b,k+c) for a,b,c in product((0,1),repeat=3)]
        if max(vv)<=0 and min(vv)<0:
            countlo+=1
        if min(vv)<0:
            counthi+=1
    vcell=math.prod(steps);vlo,vhi=countlo*vcell,counthi*vcell
    nominal=sum(b-a for a,b in intervals)*Ly*Lz
    return (error(node_err+rest,e.unit,'Every exact reference node checked; true signed-distance 1-Lipschitz weighted-variance interpolation remainder'),
            verified(vlo,vhi,e.unit+'^3','Trilinear corner convex hull; zero polynomial cells are exterior for sd<0'),
            error(max(abs(vlo/nominal-1),abs(vhi/nominal-1)),'1','Covered trilinear volume enclosure relative to exact reference'))


def revalidate(contract,evidence):
    if type(contract) is not GeometryContract:
        raise ValueError('recomputed GeometryContract required')
    fresh=verify_geometry(evidence,pitch=contract.pitch)
    if fresh!=contract:
        raise ValueError('stale, forged or incompatible geometry contract')
    return fresh


_LENGTH_TO_M={'m':Q(1),'mm':Q(1,1000)}


def _distance_budget_in_contract_unit(budget,budget_unit,contract_unit):
    if budget_unit is None:
        raise ValueError('distance budget needs distance_unit; bare numbers are not lengths')
    if budget_unit==contract_unit:
        return budget
    if budget_unit in _LENGTH_TO_M and contract_unit in _LENGTH_TO_M:
        return budget*_LENGTH_TO_M[budget_unit]/_LENGTH_TO_M[contract_unit]
    raise ValueError('distance budget unit does not match contract unit')


def assess(contract, *, evidence, distance=None, normal_chord=None, area_relative=None,
           volume_relative=None, require_topology=True, require_resolution=True,
           require_contact=False, require_physical=False, distance_unit=None,
           distance_metric='GLOBAL_BOUNDARY_HAUSDORFF_UPPER'):
    """Budget FAIL means this certificate cannot admit; it need not prove a violation.

    The contract unit is the caller's declaration (physical_acquisition stays
    UNKNOWN); a distance budget must state its unit and is converted exactly
    between m and mm only. A distance budget is checked only against a bound of
    the requested metric: a signed-field L-infinity band on an SDF is not a
    boundary Hausdorff bound (a sub-pitch channel can sit 2 units from the
    interpolant's zero set under a 0.87 signed-field band). An assessment that
    requests no check is UNKNOWN, never PASS.
    """
    contract=revalidate(contract,evidence)
    checks={}
    for name,bound,budget in [('distance',contract.distance,distance),('normal',contract.normal_chord,normal_chord),
                            ('area',contract.area_relative_error,area_relative),('volume',contract.volume_relative_error,volume_relative)]:
        if budget is None:
            continue
        budget=rat(budget)
        if budget<0:
            raise ValueError('nonnegative budget required')
        if name=='distance':
            budget=_distance_budget_in_contract_unit(budget,distance_unit,contract.unit)
            if contract.distance_metric!=distance_metric:
                checks[name]='UNKNOWN'
                continue
        checks[name]='UNKNOWN' if bound.status!='VERIFIED' else ('PASS' if bound.upper<=budget else 'FAIL')
    if require_topology:
        checks['topology']='UNKNOWN' if contract.topology.status!='VERIFIED' else ('PASS' if contract.topology.preserved else 'FAIL')
    if require_resolution:
        checks['feature_resolution']= {'UNKNOWN':'UNKNOWN','SUBGRID_FEATURE':'FAIL','VERIFIED_SEPARATION_GE_PITCH':'PASS'}[contract.feature_resolution]
    if require_contact:
        checks['contact_branch']='UNKNOWN'
    if require_physical:
        checks['physical_acquisition']='UNKNOWN'
    if not checks:
        return {'status':'UNKNOWN','checks':checks,'scope':'DECLARED_MODEL_ONLY','reason':'No check requested'}
    status='FAIL' if 'FAIL' in checks.values() else ('UNKNOWN' if 'UNKNOWN' in checks.values() else 'PASS')
    return {'status':status,'checks':checks,'scope':'DECLARED_MODEL_ONLY'}


def conditional_normal_from_sdf(distance_upper, hessian_sum_upper, common_ball_radius):
    """Conditional theorem, never geometric proof: |grad e|<=eps/t+H*t/2.

    Requires true consistently oriented signed distances, uniform C0 error and
    Hessian sum bound throughout a common ball. Reach must be proved for BOTH
    surfaces; scalar input metadata cannot discharge these obligations.
    """
    eps,H,r=map(rat,(distance_upper,hessian_sum_upper,common_ball_radius))
    if eps<0 or H<0 or r<=0:
        raise ValueError('nonnegative error/Hessian and positive common radius required')
    if eps==0:
        return {'status':'CONDITIONAL_ONLY','normal_chord_upper':Q(0),'step':Q(0)}
    t=min(r/2,sqrt_bounds(2*eps/H)[0]) if H else r/2
    if t<=0:
        raise ValueError('unresolved finite-difference step')
    return {'status':'CONDITIONAL_ONLY','normal_chord_upper':min(Q(2),eps/t+H*t/2),'step':t}


def conditional_tube_measures(distance_upper, curvature_upper, normal_dot_lower, nominal_area):
    """Conditional area ratio and volume error for a one-sheet normal graph.

    A verified bijective projection, full coverage, consistent solid orientation
    and |principal curvature|<=K on the tube are REQUIRED in addition to numbers.
    """
    d,K,c,A=map(rat,(distance_upper,curvature_upper,normal_dot_lower,nominal_area))
    if d<0 or K<0 or not 0<c<=1 or A<=0 or d*K>=1:
        raise ValueError('regular oriented tube required')
    return {'status':'CONDITIONAL_ONLY','area_ratio_lower':(1-d*K)**2,
            'area_ratio_upper':(1+d*K)**2/c,
            'volume_error_upper':A*d*(1+K*d+K*K*d*d/3)}
