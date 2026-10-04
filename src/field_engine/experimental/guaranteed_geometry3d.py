"""Separate geometry enclosures. Probe/median diagnostics are never bounds."""
from fractions import Fraction as F
from dataclasses import dataclass
import json
import weakref
import hashlib
import math
from .guaranteed_scalar3d import rat, down, up, sub, dot, cross, require_mesh


_VERIFIED = object()
_VERIFIED_OBJECTS = weakref.WeakSet()


def _register_verified(obj):
    _VERIFIED_OBJECTS.add(obj)
    return obj


def _verified(obj,kind):
    return type(obj) is kind and obj._token is _VERIFIED and obj in _VERIFIED_OBJECTS


def sdf_model_sha256(axes, sampled_sdf):
    """Hash of the exact SDF model (axes, shape, exact samples) used by the evidence."""
    import numpy as np
    axes = tuple(tuple(map(rat, ax)) for ax in axes); sd = np.asarray(sampled_sdf)
    samples = tuple(rat(sd[idx]) for idx in np.ndindex(sd.shape))
    return hashlib.sha256(repr((axes, sd.shape, samples)).encode()).hexdigest()


@dataclass(frozen=True,eq=False)
class VerifiedSDFBound:
    mesh: object
    distance_bound: F
    _summary_json: str
    _token: object = None

    @property
    def summary(self):
        return json.loads(self._summary_json)

    def matches(self, axes, sampled_sdf):
        """True only for the exact grid/samples that were verified.

        The evidence holds a hash, not the array; a consumer must check the
        array it actually solves on (review F2).
        """
        try:
            return sdf_model_sha256(axes, sampled_sdf) == self.summary['sdf_model_sha256']
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True,eq=False)
class VerifiedHomothety:
    mesh_sha256: str
    rmin: F
    rmax: F
    _summary_json: str
    _token: object = None

    @property
    def summary(self):
        return json.loads(self._summary_json)


@dataclass(frozen=True,eq=False)
class VerifiedShearFamily:
    mesh_sha256: str
    beta_bound: F
    kappa_min: F
    kappa_max: F
    _summary_json: str
    _token: object = None

    @property
    def summary(self):
        return json.loads(self._summary_json)


def verified_shear_family(mesh, beta_bound, displacement_axis=0, coordinate_axis=1):
    """All maps x -> x+beta*x_b*e_a, |beta|<=B<1, a!=b.

    det J=1; inverse=I-beta*e_a e_b^T. The reference pullback energy tensor
    A=J^-1 J^-T has exact Gershgorin bounds1-B and1+B+B². All material and
    finite-electrode faces are transported by the same explicit injective map.
    Physical acquisition cannot be replaced by membership in this model family.
    """
    require_mesh(mesh);B=rat(beta_bound);a,b=displacement_axis,coordinate_axis
    if a not in (0,1,2) or b not in (0,1,2) or a==b or not 0<=B<1:
        raise ValueError('distinct 3D shear axes and 0<=B<1 required')
    lo,hi=1-B,1+B+B*B;disp=B*max(abs(p[b]) for p in mesh.points)
    summary={'kind':'EXACT_TOPOLOGY_AND_ELECTRODE_PRESERVING_SHEAR_FAMILY',
             'mesh_sha256':mesh.sha256,'beta_abs_upper_exact':str(B),
             'displacement_axis':a,'coordinate_axis':b,'det_jacobian_exact':'1',
             'kappa_min_lower':down(lo),'kappa_max_upper':up(hi),
             'max_boundary_displacement_upper':up(disp),
             'scope':'ALL_CONTINUOUS_BETA_IN_DECLARED_INTERVAL_WITH_TRANSPORTED_ELECTRODES',
             'acquisition_error':'UNKNOWN_NOT_INCLUDED'}
    return _register_verified(VerifiedShearFamily(mesh.sha256,B,lo,hi,json.dumps(summary),_VERIFIED))


def shear_family_response(cert, family):
    """Poisson compliance/resistance on the complete unit-determinant family.

    Rayleigh/Dirichlet/Thomson comparison: response scales inversely with
    conductivity. det J=1 keeps the Poisson body-source measure unchanged.
    """
    if (not _verified(family,VerifiedShearFamily)
            or family.mesh_sha256!=cert.get('mesh_sha256')):
        raise ValueError('verified shear family bound to this exact certificate mesh required')
    if cert.get('quantity') not in ('poisson_compliance','effective_resistance') or not cert.get('constant_coefficient'):
        raise ValueError('constant-coefficient typed scalar response required')
    lo=max(F(0),rat(cert['lower']))/family.kappa_max;hi=rat(cert['upper'])/family.kappa_min
    flo,fhi=down(lo),up(hi);mid=float((rat(flo)+rat(fhi))/2)
    change=rat(cert['upper'])*max(1/family.kappa_min-1,1-1/family.kappa_max)
    return {'lower':flo,'upper':fhi,'estimate':mid,
            'absolute_error_upper':up(max(abs(rat(mid)-rat(flo)),abs(rat(fhi)-rat(mid)))),
            'relative_width_upper':up((rat(fhi)-rat(flo))/rat(flo)) if flo>0 else None,
            'numerical_width_upper':up(rat(cert['upper'])-rat(cert['lower'])),
            'geometry_error_upper_separate':up(max(abs(rat(flo)-rat(cert['lower'])),abs(rat(fhi)-rat(cert['upper'])))),
            'consumer_geometry_change_upper':up(change),
            'geometry_term_kind':'INTERVAL_TRANSPORT_CORRECTION; TRUE_QUANTITY_CHANGE_SEPARATELY_BOUNDED',
            'geometry_extra_width_upper':up((rat(fhi)-rat(flo))-(rat(cert['upper'])-rat(cert['lower']))),
            'geometry_evidence':family.summary,'physical_acquisition_error':'UNKNOWN'}


def field_shear_family_response(cert,family):
    """Transport admissible fields, retaining their directional energy moments.

    This avoids replacing all field directions by extreme tensor eigenvalues.
    The exact affine Piola current and pulled-back potential preserve all
    sources and traces. The two positive quadratic energies attain their
    maxima at a beta endpoint, closing the whole continuous design interval.
    """
    # Reuse scope/binding/type admission of the coarser enclosure.
    coarse=shear_family_response(cert,family)
    moments=cert.get('energy_moments')
    if not moments or moments.get('arithmetic')!='EXACT_RATIONAL_TETRA_VOLUME_MOMENTS':
        raise ValueError('exact certified field energy moments required')
    def tensor(key):
        data=moments.get(key)
        if not isinstance(data,(tuple,list)) or len(data)!=3 or any(len(row)!=3 for row in data):
            raise ValueError('exact 3x3 energy moment tensor required')
        A=tuple(tuple(F(x) for x in row) for row in data)
        if any(A[i][j]!=A[j][i] for i in range(3) for j in range(3)) or any(A[i][i]<0 for i in range(3)):
            raise ValueError('symmetric nonnegative energy moments required')
        return A
    G,Q=tensor('primal'),tensor('dual');a=family.summary['displacement_axis'];b=family.summary['coordinate_axis'];B=family.beta_bound
    E0=sum(G[i][i] for i in range(3));D0=sum(Q[i][i] for i in range(3))
    Emax=E0+2*B*abs(G[a][b])+B*B*G[a][a];Dmax=D0+2*B*abs(Q[a][b])+B*B*Q[b][b]
    if cert['quantity']=='effective_resistance':
        if Emax<=0:raise ValueError('positive finite-electrode primal energy required')
        lo,hi=1/Emax,Dmax
    else:lo,hi=max(F(0),2*F(moments['load_exact'])-Emax),Dmax
    flo,fhi=down(lo),up(hi);mid=float((rat(flo)+rat(fhi))/2)
    expansion=max(F(0),rat(cert['lower'])-rat(flo),rat(fhi)-rat(cert['upper']))
    difference=rat(cert['upper'])-rat(cert['lower'])+expansion
    # A readout interval correction and the change of two unknown continuum
    # responses are different quantities; explicitly bound both.
    change=min(rat(coarse['consumer_geometry_change_upper']),difference)
    return {'lower':flo,'upper':fhi,'estimate':mid,
            'absolute_error_upper':up(max(abs(rat(mid)-rat(flo)),abs(rat(fhi)-rat(mid)))),
            'relative_width_upper':up((rat(fhi)-rat(flo))/rat(flo)) if flo>0 else None,
            'numerical_width_upper':up(rat(cert['upper'])-rat(cert['lower'])),
            'geometry_error_upper_separate':up(max(abs(rat(flo)-rat(cert['lower'])),abs(rat(fhi)-rat(cert['upper'])))),
            'consumer_geometry_change_upper':up(change),
            'geometry_term_kind':'INTERVAL_TRANSPORT_CORRECTION; TRUE_QUANTITY_CHANGE_SEPARATELY_BOUNDED',
            'geometry_extra_width_upper':up((rat(fhi)-rat(flo))-(rat(cert['upper'])-rat(cert['lower']))),
            'primal_geometry_energy_increment_exact':str(Emax-E0),
            'dual_geometry_energy_increment_exact':str(Dmax-D0),
            'geometry_evidence':family.summary,'physical_acquisition_error':'UNKNOWN',
            'method':'EXACT_TRANSPORTED_PRIMAL_AND_PIOLA_CURRENT_ENERGY_MOMENTS'}


def sqrt_bounds(x, bits=80):
    x = rat(x)
    if x < 0:
        raise ValueError('nonnegative squared distance required')
    bits = max(bits, (x.denominator.bit_length()-x.numerator.bit_length())//2+80) if x else bits
    scale = 1 << bits
    s = math.isqrt((x.numerator << (2*bits))//x.denominator)
    lo = F(s, scale)
    return lo, lo if lo*lo == x else F(s+1, scale)


def boundary_triangles(mesh):
    require_mesh(mesh)
    faces = []
    for face, inc in zip(mesh.faces, mesh.incidences):
        if len(inc) != 1:
            continue
        ti, j, _ = inc[0]
        a, b, d = (mesh.points[i] for i in face); n = cross(sub(b, a), sub(d, a))
        opposite = mesh.points[mesh.tets[ti][j]]
        faces.append((face[0],face[2],face[1]) if dot(n,sub(opposite,a)) > 0 else face)
    return tuple(faces)


def support_planes(mesh, center=(0, 0, 0)):
    require_mesh(mesh); c = tuple(map(rat, center)); planes = []
    for face in boundary_triangles(mesh):
        a, b, d = (mesh.points[i] for i in face); n = cross(sub(b, a), sub(d, a))
        if dot(n, sub(c, a)) > 0:
            n = tuple(-x for x in n)
        if dot(n, sub(c, a)) >= 0 or any(dot(n, sub(v, a)) > 0 for v in mesh.points):
            raise ValueError('strict-center convex body required for geometry enclosure')
        planes.append((a, n))
    return tuple(planes)


def inradius_lower(mesh, center=(0, 0, 0)):
    c = tuple(map(rat, center)); planes = support_planes(mesh, c)
    return sqrt_bounds(min(dot(n, sub(c, a))**2/dot(n, n) for a, n in planes))[0]


def homothety_to_ellipsoid(mesh, semi_axes):
    """Geometry only: r_min E subset P subset r_max E, exact support tests.

    No PDE reference value is used. Binary64 semi-axes are exact model inputs.
    """
    from .guaranteed_scalar3d import _construct
    # The enclosure is bound to mesh.sha256; a replaced/hand-built mesh would
    # carry the original hash with other points (review F1).
    require_mesh(mesh)
    axes = tuple(map(rat, semi_axes))
    if len(axes) != 3 or min(axes) <= 0:
        raise ValueError('three positive exact semi axes required')
    normalized = _construct([tuple(v[i]/axes[i] for i in range(3)) for v in mesh.points],
                            mesh.tets, 'EXACT_NORMALIZED_LINEAR_IMAGE')
    rmin = inradius_lower(normalized)
    rmax = sqrt_bounds(max(dot(v, v) for v in normalized.points))[1]
    summary = {'rmin_lower': down(rmin), 'rmax_upper': up(rmax), 'mesh_sha256': mesh.sha256,
            'method': 'EXACT_CONVEX_SUPPORT_PLANES_AND_VERTEX_NORMS',
            'geometry_floor_relative': up((rmax/rmin)**5-1)}
    return _register_verified(VerifiedHomothety(mesh.sha256, rmin, rmax, json.dumps(summary), _VERIFIED))


def _transport_compliance(cert, lower_scale=1, upper_scale=1, evidence=None):
    """Constant-coefficient/source Dirichlet compliance scales as length^5.

    Requires an explicit domain-inclusion proof/evidence, not a median error.
    Evidence describes the verified modeled geometry, not physical acquisition.
    """
    if cert.get('quantity') != 'poisson_compliance' or not cert.get('constant_coefficient', False):
        raise ValueError('only constant-coefficient Poisson compliance has this scaling contract')
    a, b = rat(lower_scale), rat(upper_scale)
    if not evidence or a < 0 or b <= 0 or a > b:
        raise ValueError('verified domain-inclusion evidence and ordered nonnegative scales required')
    lo = a**5*rat(cert['lower']); hi = b**5*rat(cert['upper'])
    if cert['lower'] < 0:
        # Compliance is nonnegative; negative lower bounds cannot be scaled in
        # a domain-monotonicity direction. Use the known universal lower zero.
        lo = F(0)
    flo, fhi = down(lo), up(hi); mid = float((rat(flo)+rat(fhi))/2)
    geometry_error = max(abs(rat(flo)-rat(cert['lower'])), abs(rat(fhi)-rat(cert['upper'])))
    change=rat(cert['upper'])*max(abs(a**5-1),abs(b**5-1))
    return {'lower': flo, 'upper': fhi, 'estimate': mid,
            'absolute_error_upper': up(max(abs(rat(mid)-rat(flo)), abs(rat(fhi)-rat(mid)))),
            'relative_width_upper': up((rat(fhi)-rat(flo))/rat(flo)) if flo > 0 else None,
            'numerical_width_upper': up(rat(cert['upper'])-rat(cert['lower'])),
            'geometry_error_upper_separate': up(geometry_error),
            'consumer_geometry_change_upper':up(change),
            'geometry_term_kind':'INTERVAL_TRANSPORT_CORRECTION; TRUE_QUANTITY_CHANGE_SEPARATELY_BOUNDED',
            'geometry_extra_width_upper': up((rat(fhi)-rat(flo))-(rat(cert['upper'])-rat(cert['lower']))),
            'geometry_evidence': evidence, 'physical_acquisition_error': 'UNKNOWN'}


def ellipsoid_compliance(cert, enclosure):
    if (not _verified(enclosure,VerifiedHomothety)
            or cert.get('mesh_sha256') != enclosure.mesh_sha256):
        raise ValueError('verified homothety for this exact certificate mesh required')
    return _transport_compliance(cert, 1/enclosure.rmax, 1/enclosure.rmin, enclosure.summary)


def uncertain_convex_compliance(cert, mesh, distance_bound, evidence, axes=None, sampled_sdf=None):
    """Given erosion(P,delta) subset Omega subset P+ball(delta).

    Omega is the SDF model named by evidence.summary['sdf_model_sha256'].
    Pass axes and sampled_sdf to have the array a consumer used checked
    against that hash; a mismatch is refused.

    rho-ball subset convex P proves (1-delta/rho)P subset erosion and
    dilation subset (1+delta/rho)P. The acquisition/SDF sandwich is an explicit
    premise. A probe report lacking this evidence is refused.
    """
    delta = rat(distance_bound)
    if (delta < 0 or not _verified(evidence,VerifiedSDFBound)
            or evidence.mesh is not mesh or delta < evidence.distance_bound or cert.get('mesh_sha256') != mesh.sha256):
        raise ValueError('a verified global signed-distance enclosure is required; probes/medians are insufficient')
    if (axes is None) != (sampled_sdf is None) or (axes is not None and not evidence.matches(axes, sampled_sdf)):
        raise ValueError('sampled SDF differs from the verified SDF model')
    rho = inradius_lower(mesh)
    return _transport_compliance(cert, max(F(0), 1-delta/rho), 1+delta/rho,
                                {**evidence.summary, 'distance_bound_upper': up(delta), 'inradius_lower': down(rho)})


def segment_distance2(p, a, b):
    e, w = sub(b, a), sub(p, a); t = dot(w, e)/dot(e, e)
    t = max(F(0), min(F(1), t))
    v = tuple(w[i]-t*e[i] for i in range(3))
    return dot(v, v)


def triangle_distance2(p, a, b, c):
    """Exact rational closest-point test including edges, corners, obtuse faces."""
    e, h, w = sub(b, a), sub(c, a), sub(p, a)
    ee, eh, hh, we, wh = dot(e, e), dot(e, h), dot(h, h), dot(w, e), dot(w, h)
    den = ee*hh-eh*eh
    if den <= 0:
        raise ValueError('nondegenerate triangle required')
    u, v = (hh*we-eh*wh)/den, (ee*wh-eh*we)/den
    if u >= 0 and v >= 0 and u+v <= 1:
        z = tuple(w[i]-u*e[i]-v*h[i] for i in range(3))
        return dot(z, z)
    return min(segment_distance2(p, a, b), segment_distance2(p, b, c), segment_distance2(p, c, a))


def verify_grid_sdf(mesh, axes, sampled_sdf):
    """Verify every sample against exact triangles and pay interpolation rest.

    The model body is {trilinear(sampled_sdf)<0} restricted to this grid box.
    Since signed distance is1-Lipschitz, the interpolation defect is <=sqrt(3)h/2
    on isotropic cells (more generally sqrt(sum h_i²)/2), by Jensen and
    sum t_i(1-t_i)h_i². Sampling, sign and float error are charged explicitly.
    No finite probe/sample maximum is confused with a continuum maximum.
    """
    import numpy as np
    require_mesh(mesh); axes = tuple(tuple(map(rat, ax)) for ax in axes); sd = np.array(sampled_sdf,copy=True)
    if len(axes) != 3 or sd.shape != tuple(len(ax) for ax in axes) or not np.all(np.isfinite(sd)):
        raise ValueError('finite complete 3D sample box required')
    steps = []
    for ax in axes:
        if len(ax) < 2 or any(b <= a for a, b in zip(ax, ax[1:])):
            raise ValueError('strictly increasing axes required')
        steps.append(max(b-a for a, b in zip(ax, ax[1:])))
    if any(min(p[i] for p in mesh.points) <= axes[i][0] or max(p[i] for p in mesh.points) >= axes[i][-1] for i in range(3)):
        raise ValueError('reference body must lie strictly inside the covered grid box')
    planes = support_planes(mesh); tris = [tuple(mesh.points[i] for i in t) for t in boundary_triangles(mesh)]
    nodal_error, errors, worst_idx = F(0), [], None
    exact_samples=[]
    for idx in np.ndindex(sd.shape):
        p = tuple(axes[i][idx[i]] for i in range(3))
        d2 = min(triangle_distance2(p, *tri) for tri in tris)
        dl, dh = sqrt_bounds(d2)
        inside = all(dot(n, sub(p, a)) <= 0 for a, n in planes)
        if inside:
            dl, dh = -dh, -dl
        value = rat(sd[idx]);exact_samples.append(value);err = max(abs(value-dl), abs(value-dh)); errors.append(float(err))
        if err > nodal_error:
            nodal_error, worst_idx = err, idx
    interpolation = sqrt_bounds(sum(h*h for h in steps))[1]/2
    bound = nodal_error+interpolation
    summary = {'scope': 'GLOBAL_SIGNED_DISTANCE_ON_COVERED_BOX',
            'mesh_sha256': mesh.sha256,
            'sdf_model_sha256':sdf_model_sha256(axes,sd),
            'domain': 'NEGATIVE_TRILINEAR_SDF_RESTRICTED_TO_COMPLETE_GRID_BOX',
            'verified_nodes': int(sd.size), 'nodal_error_upper': up(nodal_error),
            'nodal_error_median_diagnostic': float(np.median(errors)), 'worst_node': worst_idx,
            'interpolation_error_upper': up(interpolation), 'distance_bound_upper': up(bound),
            'method': 'EXACT_TRIANGLE_DISTANCE_AT_ALL_NODES_PLUS_PROVED_LIPSCHITZ_INTERPOLATION',
            'acquisition_error': 'UNKNOWN_NOT_INCLUDED'}
    return _register_verified(VerifiedSDFBound(mesh, bound, json.dumps(summary), _VERIFIED))


def uncertain_box_resistance(lengths, face_uncertainty, conductivity=1):
    """Only axis-aligned face translations preserving full planar electrodes.

    Admissible affine potential/unit-current give identical bounds L/(sigma WH).
    No general resistance monotonicity or arbitrary SDF-electrode claim is made.
    The face-translation premise is asserted by the caller and not verified
    here. A signed-distance bound is not an admissible face_uncertainty: a
    wall thinner than that bound can disconnect an electrode (R=infinity).
    """
    L, W, H = tuple(map(rat, lengths)); d, k = rat(face_uncertainty), rat(conductivity)
    if d < 0 or min(L, W, H) <= 2*d or k <= 0:
        raise ValueError('positive dimensions, conductivity and bounded face translations required')
    lo = (L-2*d)/(k*(W+2*d)*(H+2*d)); hi = (L+2*d)/(k*(W-2*d)*(H-2*d))
    return {'lower': down(lo), 'upper': up(hi), 'relative_width_upper': up((hi-lo)/lo),
            'geometry_contract': 'INDEPENDENT_AXIS_ALIGNED_FACE_TRANSLATIONS_FULL_END_ELECTRODES',
            'geometry_premise': 'CALLER_ASSERTED_NOT_VERIFIED',
            'method': 'EXACT_AFFINE_PRIMAL_AND_UNIT_CURRENT_DUAL', 'physical_acquisition_error': 'UNKNOWN'}
