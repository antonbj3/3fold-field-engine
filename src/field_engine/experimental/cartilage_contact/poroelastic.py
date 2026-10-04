"""Exploratory bounded relaxation between elastic drainage endpoints.

The exp and 1D Terzaghi kernels are alternative closures, not a solved 3D
biphasic contact problem. tau=L²/(H_A*k) has units seconds.
"""
import math

from .contact_api import cartilage_contact


def terzaghi(T, terms=200):
    if T <= 0:
        return 1.0
    return sum(8/(math.pi**2*(2*m+1)**2)*
               math.exp(-math.pi**2*(2*m+1)**2*T/4)
               for m in range(terms))


def force_time_envelope(seconds,R,thickness,delta,L=1e-3,
                        H_A_range=(.70e6,.76e6),
                        k_range=(1e-15,7.6e-15),
                        nu_drained_range=(.1,.2),nu_undrained=.499):
    """Return candidate F(s) envelope [N] for constant indentation.

    E_d follows from aggregate modulus M_d=H_A; shear modulus is kept fixed
    when changing drained to nearly incompressible Poisson ratio.
    """
    if seconds < 0 or L <= 0:
        raise ValueError('seconds must be nonnegative and L positive')
    samples=[]
    for H_A in H_A_range:
        for nu_d in nu_drained_range:
            E_d=H_A*(1+nu_d)*(1-2*nu_d)/(1-nu_d)
            mu=E_d/(2*(1+nu_d))
            E_u=2*mu*(1+nu_undrained)
            F_d=cartilage_contact(R,thickness,E_d,nu_d,delta)[0]
            F_u=cartilage_contact(R,thickness,E_u,nu_undrained,delta)[0]
            for k in k_range:
                tau=L*L/(H_A*k)
                T=seconds/tau
                for form,value in (('exponential',math.exp(-T)),
                                   ('terzaghi_1d',terzaghi(T))):
                    F=F_d+(F_u-F_d)*value
                    samples.append({'H_A_Pa':H_A,'k_m4_Ns':k,'nu_d':nu_d,
                                    'form':form,'tau_s':tau,'F_N':F,
                                    'F_drained_N':F_d,'F_undrained_N':F_u})
    return {'time_s':seconds,'F_N_interval':(min(s['F_N'] for s in samples),
                                           max(s['F_N'] for s in samples)),
            'tau_s_interval':(min(s['tau_s'] for s in samples),
                              max(s['tau_s'] for s in samples)),
            'samples':samples,'status':'model envelope; no biphasic validation'}
