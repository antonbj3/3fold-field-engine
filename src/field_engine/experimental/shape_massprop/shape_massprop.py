"""Unified mass property interface. Mesh units mm; declared CSG units m.

For SDF, exactness refers to the Reynolds identity on each smooth branch;
the n-grid surface quadrature is approximate. Bounds returned are empirical
checks at the requested parameter, never certified enclosure bounds.
"""
import numpy as np
from . import femur_massprops as fm
from . import kinkprop as kp

GRIDS = {
    'finger': ([-.13, -.05, -.05], [-.03, .05, .05]),
    'femur': ([-.07, -.05, -.02], [.07, .05, .18]),
}


def _physical(C):
    return np.trace(C, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) - C


def shape_massprop(shape, theta, n=48, reference=False):
    """Return V,c,I,dV,dc,dI,error_bound and units.

    shape={'kind':'mesh','V': vertices,'F': faces,'frame': frozen_edit_frame}
    or shape='finger'/'femur' for the declared CSG leaves.
    The tensor derivative has leading parameter axis. Mesh coordinates, theta[2]
    and returned lengths use mm; V is mm^3, c is mm, I is mm^5;
    theta[0:2] use radians. CSG lengths use m, V m^3, I m^5, with angular
    parameters in radians. Derivative units divide by each parameter unit.
    Mesh derivatives
    are exact for the discrete geometry, without a floating-point enclosure.
    CSG derivatives use kink-split quadrature; the reported error is either
    uncalibrated or an empirical comparison at this theta, never a certificate.
    """
    theta = np.asarray(theta, float)
    if isinstance(shape, dict) and shape.get('kind') == 'mesh':
        V0, F, frame = shape['V'], shape['F'], shape['frame']
        V = fm.V_of_u(V0, frame, theta)
        vol, c, I = fm.mass_properties(V, F)
        J = fm.dM_du(V0, F, frame, theta)
        dI = np.zeros((3, 3, 3))
        for row, (i, j) in enumerate(((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))):
            dI[:,i,j] = J[4+row]
            dI[:,j,i] = J[4+row]
        return dict(V=vol, c=c, I=I, dV=J[0], dc=J[1:4].T,
                    dI=dI, error_bound={'type':'analytic_discrete_mesh','absolute':0.0,
                                       'roundoff_certified':False}, units='mm')
    if shape not in GRIDS:
        raise ValueError('shape must be a mesh descriptor, finger or femur')
    lo, hi = (np.asarray(x) for x in GRIDS[shape])
    base = kp.exact_properties(shape, theta)
    surf = kp.surface_mass_derivatives_kink(shape, theta, lo, hi, n,
                                            base_properties=base)
    vol = base['volume']
    c = base['com']
    I = _physical(base['central_inertia'])
    out = dict(V=vol, c=c, I=I, dV=surf['d_volume'], dc=surf['d_com'],
               dI=surf['d_physical_inertia'], units='m',
               error_bound={'type':'uncalibrated_surface_discretization',
                            'absolute':None, 'coverage':surf['surface']['coverage']})
    if reference:
        steps = [1e-5]*3 if shape == 'finger' else [1e-6]*2
        a = kp.exact_central_fd(shape, theta, steps)
        b = kp.exact_central_fd(shape, theta, [h/2 for h in steps])
        ref = {key:np.asarray(a[key]) for key in ('volume','com','physical_inertia')}
        step = {key:np.abs(ref[key]-np.asarray(b[key])) for key in ref}
        out['reference'] = ref
        out['error_bound'] = {'type':'empirical_at_this_theta', 'coverage':surf['surface']['coverage'],
                              'absolute':{'dV':np.abs(out['dV']-ref['volume'])+step['volume'],
                                          'dc':np.abs(out['dc']-ref['com'])+step['com'],
                                          'dI':np.abs(out['dI']-ref['physical_inertia'])+step['physical_inertia']}}
    return out
