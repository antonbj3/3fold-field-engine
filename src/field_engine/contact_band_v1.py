"""Conditional geometric contact bands; no force/slip inference.

Units: every length is mm (gaps, sigmas, radii, biases, thresholds, bounds).
Gap convention: positive = separation, negative = penetration.

Classes, judged on the bias-corrected band [lower_mm, upper_mm]:
  -1  closed: upper_mm <  contact_distance_mm on the whole band
  +1  open:   lower_mm >  contact_distance_mm on the whole band
   0  indeterminate (touching the threshold, unknown completion sign, or a
      known sign that contradicts the distance band)

A -1/+1 class is CONDITIONAL: it holds only inside the declared event
|error| <= z*scan_sigma_mm plus the deterministic radii. z is a per-contact
two-sided multiplier; it is not adjusted for the number of contacts. Joint
coverage over N contacts needs a caller-chosen z (e.g. a union bound
z = Phi^-1(1 - alpha/(2N))) or a joint law. Gaussian sigmas require an
externally justified joint law. Deterministic SDF and tessellation bounds are
added as radii, never treated as sigmas. A p99 is not a uniform bound. The
final one-ulp outward step is not a rigorous interval-arithmetic enclosure.
Unknown completion signs always preserve uncertainty.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class ContactBand:
    lower_mm: np.ndarray
    upper_mm: np.ndarray
    sigma_tot_mm: np.ndarray  # projected joint Gaussian gap sigma only; radii are inside lower/upper
    classes: np.ndarray  # -1 closed, +1 open, 0 indeterminate
    completion_unknown: np.ndarray
    sign_conflict: np.ndarray  # known completion sign contradicts the band; kept indeterminate
    evidence_id: str

    @property
    def uncertain(self):
        return self.classes == 0

    def guard_input(self):
        return {'uncertain_candidates': np.flatnonzero(self.uncertain).tolist(),
                'completion_unknown': np.flatnonzero(self.completion_unknown).tolist(),
                'sign_conflict': np.flatnonzero(self.sign_conflict).tolist(),
                'gap_bounds_mm': np.column_stack((self.lower_mm, self.upper_mm)).tolist(),
                'evidence_id': self.evidence_id,
                'scope': 'GEOMETRIC_PRECONDITION_ONLY'}


def _readonly(a):
    a.flags.writeable = False
    return a


def contact_band(gap_mm, *, scan_sigma_mm, sdf_radius_mm, z, tess_bias_mm=0.,
                 tess_radius_mm=0., scan_bias_mm=0., contact_distance_mm=0.,
                 completion_sign=None, gap_bounds_mm=None, evidence_id):
    """Build a band conditional on the declared Gaussian event and bounds.

    Bias convention: observed gap = corrected gap + scan_bias + tess_bias.
    scan_sigma_mm is the projected JOINT gap sigma; summing surface variances
    needs justified covariance. z has no default: it sets the coverage of
    every certain class. completion_sign is the geometric sign of the
    uncorrected gap (+1 outside/separated, -1 inside, 0 unknown); without it
    no branch is certain, and a known sign that contradicts the corrected band
    keeps that contact indeterminate. gap_bounds_mm, if supplied, encloses all
    completion distances BEFORE bias correction, e.g. magnitude
    [min(d_obs,dist(q,support)),d_obs], and must contain gap_mm.
    """
    gap=np.asarray(gap_mm,dtype=float)
    if gap.ndim!=1 or not np.all(np.isfinite(gap)):
        raise ValueError('finite one-dimensional gap array required')
    if not isinstance(evidence_id,str) or not evidence_id.strip():
        raise ValueError('evidence identifier required')
    vals=[np.broadcast_to(np.asarray(x,dtype=float),gap.shape).copy() for x in
          (scan_sigma_mm,sdf_radius_mm,tess_bias_mm,tess_radius_mm,scan_bias_mm)]
    sg,sd,b,te,sb=vals
    if any(not np.all(np.isfinite(v)) for v in vals) or np.any(sg<0) or np.any(sd<0) or np.any(te<0):
        raise ValueError('finite errors and nonnegative sigmas/radii required')
    if isinstance(z,bool) or np.ndim(z)!=0 or not np.isfinite(z) or z<0 or not np.isfinite(contact_distance_mm):
        raise ValueError('finite nonnegative scalar z and finite threshold required')
    if completion_sign is None:
        signs=np.zeros(gap.shape,dtype=np.int8)
    else:
        raw=np.asarray(completion_sign)
        if raw.dtype==bool or raw.shape!=gap.shape or not np.all(np.isin(raw,[-1,0,1])):
            raise ValueError('completion signs must be -1/0/+1 with gap shape')
        signs=raw.astype(np.int8)
    if gap_bounds_mm is None:
        lower=upper=gap
    else:
        bounds=np.asarray(gap_bounds_mm,dtype=float)
        if bounds.shape!=(len(gap),2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:,0]>bounds[:,1]):
            raise ValueError('ordered finite completion-distance intervals required')
        lower,upper=bounds.T
        if np.any((gap<lower)|(gap>upper)):
            raise ValueError('gap_mm must lie inside its completion-distance interval')
    radius=z*sg+sd+te
    lo=np.nextafter(lower-sb-b-radius,-np.inf)
    hi=np.nextafter(upper-sb-b+radius,np.inf)
    unknown=signs==0
    conflict=((signs==1)&(hi<0))|((signs==-1)&(lo>0))
    blocked=unknown|conflict
    labels=np.where((hi<contact_distance_mm)&~blocked,-1,
                    np.where((lo>contact_distance_mm)&~blocked,1,0)).astype(np.int8)
    return ContactBand(*(_readonly(a) for a in (lo,hi,sg,labels,unknown,conflict)),evidence_id)


def projected_sigma(covariance, projection):
    """Joint source covariance [mm^2] -> scalar gap sigma [mm].

    Symmetry and PSD are checked relative to the covariance scale, so the
    result does not depend on the length unit.
    """
    c=np.asarray(covariance,dtype=float);v=np.asarray(projection,dtype=float)
    if v.ndim!=1 or len(v)==0 or c.shape!=(len(v),len(v)) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(v)):
        raise ValueError('finite matched nonempty covariance and projection required')
    scale=float(np.max(np.abs(c)))
    tol=1e-12*scale
    if np.max(np.abs(c-c.T))>tol or np.min(np.linalg.eigvalsh(.5*(c+c.T)))<-tol*len(v):
        raise ValueError('symmetric positive semidefinite covariance required')
    return float(np.sqrt(max(float(v@c@v),0.)))


def region_with_uncertainty(region, band, *, n_nodes, contact_topology):
    """Conservative consumer guard: uncertain geometry forces global fallback.

    'node' is KONTAKT (n_nodes contacts); 'edge' is BAGKEDJA (n_nodes-1
    contacts). region is an integer node range [lo, hi). No promise of a
    minimal region or safe disconnected cut is made without a force/residual
    certificate.
    """
    expected=n_nodes if contact_topology=='node' else n_nodes-1 if contact_topology=='edge' else None
    if expected is None or len(band.classes)!=expected:
        raise ValueError('band count must match explicit contact topology')
    lo,hi=region
    if any(isinstance(v,bool) or not isinstance(v,Integral) for v in (lo,hi)) or not 0<=lo<hi<=n_nodes:
        raise ValueError('ordered integer region required')
    return (0,n_nodes) if np.any(band.uncertain) else region
