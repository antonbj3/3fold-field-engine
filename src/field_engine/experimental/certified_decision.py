"""Three-state decisions on explicitly declared mathematical geometry families.

PHYSICAL is the default scope and always abstains: this module has no measured
scanner/material/load port. MODEL certifies only the stated mathematical family.
An evidence hash binds inputs; it is not evidence of physical measurement accuracy.
Surface/plane contact is a geometric overlap precondition, not contact force.
Dependencies: reviewed guaranteed_scalar and separately reviewed scalar3d patch.
"""
from dataclasses import dataclass, asdict
from fractions import Fraction as Q
from numbers import Integral
import hashlib
import json
import weakref

from .guaranteed_scalar import rat, down, up

_STATES = ('JA', 'NEJ', 'OSÄKER')
_PHYSICAL_MISSING = ('verified_physical_acquisition_and_units', 'physical_model_applicability')
# Identity registry (review): dataclasses.replace() or a hand-built Decision is
# never an issued decision, even when all its fields are internally consistent.
_ISSUED = weakref.WeakSet()


@dataclass(frozen=True, eq=False)
class Decision:
    status: str
    model_status: str
    scope: str
    quantity: str
    units: str
    lower: float | None
    upper: float | None
    exact_interval: tuple[str | None, str | None]
    margin_lower: float
    exact_margin: str
    physical_admission: bool
    input_sha256: str
    missing: tuple[str, ...]
    contributions: tuple[tuple[str, str], ...]

    def __post_init__(self):
        """Refuse internally inconsistent reports, including replace() edits."""
        margin = Q(self.exact_margin)
        if (self.status not in _STATES or self.model_status not in _STATES
                or self.scope not in ('MODEL', 'PHYSICAL') or self.physical_admission is not False):
            raise ValueError('inconsistent decision: status/scope/admission')
        if self.scope == 'PHYSICAL' and self.status != 'OSÄKER':
            raise ValueError('inconsistent decision: PHYSICAL scope never certifies here')
        if self.status == 'OSÄKER':
            if margin != 0 or self.margin_lower != 0:
                raise ValueError('inconsistent decision: abstention carries no margin')
        elif self.status != self.model_status or not margin > 0 or not 0 <= self.margin_lower <= margin:
            raise ValueError('inconsistent decision: certain status needs its own strict exact margin')

    def to_dict(self):
        return asdict(self)


def is_issued(decision):
    """True only for a Decision object returned by this module (not a copy)."""
    return type(decision) is Decision and decision in _ISSUED


def _scope(scope, units):
    if scope not in ('MODEL', 'PHYSICAL'):
        raise ValueError('scope must explicitly be MODEL or PHYSICAL')
    if not isinstance(units, str) or not units.strip():
        raise ValueError('declare model/source units; do not infer scanner units')


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _decision(lo, hi, label, margin, *, scope, units, quantity, payload, terms, missing=()):
    # A MODEL answer is about the declared family only; it still lists what a
    # physical use lacks, so a certain MODEL status is never silent about it.
    missing = tuple(missing) + _PHYSICAL_MISSING
    if scope == 'PHYSICAL':
        status = 'OSÄKER'
        margin = Q(0)
    else:
        status = label
    if status in ('JA', 'NEJ') and margin <= 0:
        raise ArithmeticError('a certain decision requires strict positive margin')
    out = Decision(status, label, scope, quantity, units,
                   down(lo) if lo is not None else None,
                   up(hi) if hi is not None else None,
                   (str(lo) if lo is not None else None, str(hi) if hi is not None else None),
                   down(margin), str(margin), False, _digest(payload), tuple(missing), tuple(terms))
    _ISSUED.add(out)
    return out


def _surface(vertices, faces):
    """Copy and validate all exact inputs; no cached mutable admission flags."""
    points = tuple(tuple(rat(x) for x in p) for p in vertices)
    if not points or any(len(p) != 3 for p in points):
        raise ValueError('nonempty finite 3D vertex array required')
    triangles = []
    for f in faces:
        if len(f) != 3 or any(isinstance(i, bool) or not isinstance(i, Integral) for i in f):
            raise ValueError('three integer vertex indices required')
        f = tuple(int(i) for i in f)
        if any(i < 0 or i >= len(points) for i in f) or len(set(f)) != 3:
            raise ValueError('distinct in-range vertex indices required')
        a, b, c = (points[i] for i in f)
        ab, ac = tuple(b[i]-a[i] for i in range(3)), tuple(c[i]-a[i] for i in range(3))
        cross = tuple(ab[(i+1)%3]*ac[(i+2)%3]-ab[(i+2)%3]*ac[(i+1)%3] for i in range(3))
        if not any(cross):
            raise ValueError('degenerate observed triangles are unsupported')
        triangles.append(f)
    if not triangles:
        raise ValueError('nonempty observed triangulation required')
    triangles = tuple(triangles)
    used = tuple(sorted({i for f in triangles for i in f}))
    return points, triangles, used


def surface_plane_contact(vertices, faces, *, axis, plane_offset, coordinate_radius,
                          missing_axis_lower=None, threshold=0, scope='PHYSICAL', units):
    """Certify min signed plane gap < threshold over all allowed completions.

    Exact halfspace: x[axis] < plane_offset + threshold. All observed vertices
    move independently by at most coordinate_radius in this coordinate; faces
    retain their indexing. Every unseen point, if present, satisfies the supplied
    missing_axis_lower AFTER uncertainty. None leaves unseen points unrestricted.
    An observed vertex used by a face proves overlap for every completion.
    Separation needs a bound on every unseen point; a supplied vertex that no
    face uses is still a supplied point and enters the separation lower bound. Model radii/supports are assumptions, not scanner
    measurements. Completeness, watertightness, normals and winding are irrelevant
    to this one affine observable. Unknown lower endpoint is not replaced by zero.
    """
    _scope(scope, units)
    if isinstance(axis, bool) or not isinstance(axis, Integral) or axis not in (0, 1, 2):
        raise ValueError('axis must be 0, 1 or 2')
    axis = int(axis)
    offset, radius, tau = rat(plane_offset), rat(coordinate_radius), rat(threshold)
    if radius < 0:
        raise ValueError('nonnegative declared coordinate radius required')
    support = None if missing_axis_lower is None else rat(missing_axis_lower)
    points, triangles, used = _surface(vertices, faces)
    m = min(points[i][axis] for i in used) - offset
    m_all = min(p[axis] for p in points) - offset
    hi = m + radius
    lo = None if support is None else min(m_all-radius, support-offset)
    if hi < tau:
        label, margin, missing = 'JA', tau-hi, ()
    elif lo is not None and lo > tau:
        label, margin, missing = 'NEJ', lo-tau, ()
    else:
        label, margin = 'OSÄKER', Q(0)
        missing = ('unseen_surface_support',) if lo is None else ('tighter_geometry_family_or_changed_plane',)
    payload = {'vertices': [[str(x) for x in p] for p in points], 'faces': triangles,
               'axis': axis, 'plane_offset': str(offset), 'radius': str(radius),
               'missing_axis_lower': None if support is None else str(support),
               'threshold': str(tau), 'scope': scope, 'units': units,
               'observable': 'MIN_GAP_OVER_ALL_ALLOWED_COMPLETIONS'}
    terms = (('observed_min_gap_exact', str(m)), ('declared_geometry_radius', str(radius)),
             ('all_supplied_vertices_min_gap_exact', str(m_all)),
             ('unseen_lower_gap', 'UNKNOWN' if support is None else str(support-offset)),
             ('surface_discretization', 'EXACT_ALL_REPRESENTED_AFFINE_TRIANGLES'),
             ('PDE', 'NOT_REQUIRED_FOR_GEOMETRIC_OVERLAP'),
             ('physical_acquisition', 'UNKNOWN'))
    return _decision(lo, hi, label, margin, scope=scope, units=units,
                     quantity='geometric_plane_overlap', payload=payload, terms=terms, missing=missing)


def box_resistance(lengths, *, conductivity, face_radius, limits, scope='PHYSICAL', units):
    """P1/RT0 numerical + exact constructive box-family resistance decision.

    Full planar end electrodes; insulating side walls; fixed positive scalar
    conductivity. Six face positions independently move by at most face_radius.
    Every family member remains an axis-aligned box with the same terminal class.
    Units are declared model units, becoming ohms only for meters and S/m.
    No generic mesh, SDF, hidden topology or Gaussian confidence is admitted.
    """
    from . import guaranteed_scalar3d as c
    from . import guaranteed_geometry3d as g
    _scope(scope, units)
    lengths = tuple(rat(x) for x in lengths)
    if len(lengths) != 3:
        raise ValueError('three box dimensions required')
    k, d = rat(conductivity), rat(face_radius)
    bounds = tuple(rat(x) for x in limits)
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError('ordered distinct design limits required')
    family = g.uncertain_box_resistance(lengths, d, k)
    mesh = c.structured_mesh(2, lengths)
    electrodes = c.box_electrodes(mesh)
    potential, flux, _ = c.solve_fields(mesh, source=0, coefficient=k, electrodes=electrodes)
    cert = c.certificate(mesh, potential, flux, source=0, coefficient=k, electrodes=electrodes)
    nominal = lengths[0]/(k*lengths[1]*lengths[2])
    # Exact box transport law derives from affine voltage and unit-current
    # trial fields. Using the independently checked reference enclosure widens
    # this family by its numerical error; it does not assume solver exactness.
    lo, hi = rat(cert['lower'])*rat(family['lower'])/nominal, rat(cert['upper'])*rat(family['upper'])/nominal
    a, b = bounds
    if a < lo and hi < b:
        label, margin = 'JA', min(lo-a, b-hi)
    elif hi < a:
        label, margin = 'NEJ', a-hi
    elif lo > b:
        label, margin = 'NEJ', lo-b
    else:
        label, margin = 'OSÄKER', Q(0)
    geometric = max(rat(cert['lower'])-lo, hi-rat(cert['upper']))
    payload = {'lengths': [str(x) for x in lengths], 'conductivity': str(k), 'face_radius': str(d),
               'limits': [str(x) for x in bounds], 'scope': scope, 'units': units,
               'mesh_sha256': mesh.sha256, 'observable': 'BOX_FAMILY_EFFECTIVE_RESISTANCE'}
    terms = (('numeric_radius_upper', str(rat(cert['absolute_error_upper']))),
             ('geometry_endpoint_expansion_upper', str(geometric)),
             ('material', 'EXACT_DECLARED_SCALAR_'+str(k)),
             ('topology', 'CONSTRUCTIVE_BOX_WITH_TRANSPORTED_FULL_END_ELECTRODES'),
             ('geometry_premise', family['geometry_premise']),
             ('physical_acquisition', 'UNKNOWN'))
    return _decision(lo, hi, label, margin, scope=scope, units=units,
                     quantity='effective_resistance', payload=payload, terms=terms,
                     missing=('tighten_family_or_numerical_interval',) if label == 'OSÄKER' else ())


def scan_resistance(vertices, faces, *, scope='PHYSICAL', units):
    """Raw surfaces lack a constructive solid/electrode/material port here."""
    _scope(scope, units)
    points, triangles, _ = _surface(vertices, faces)
    payload = {'vertices': [[str(x) for x in p] for p in points], 'faces': triangles,
               'scope': scope, 'units': units, 'observable': 'RAW_SCAN_RESISTANCE'}
    return _decision(None, None, 'OSÄKER', Q(0), scope=scope, units=units,
                     quantity='effective_resistance', payload=payload,
                     terms=(('total_error', 'UNKNOWN'),),
                     missing=('constructive_solid_and_terminal_topology', 'electrode_and_material_contract'))
