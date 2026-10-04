"""Conservative model comparison for spherical contact of a bonded layer.

Bands here are discrepancies against the corrected Hankel calculation at the
given point. They are not continuum-wide FE error guarantees.
"""
import math

from .contact_api import cartilage_contact


def regime_report(R,t,E,nu,delta):
    F,a,pmax,_=cartilage_contact(R,t,E,nu,delta)
    Estar=E/(1-nu*nu)
    Hertz={'F':4/3*Estar*math.sqrt(R)*delta**1.5,
           'a':math.sqrt(R*delta),
           'pmax':2/math.pi*Estar*math.sqrt(delta/R)}
    mu=E/(2*(1+nu))
    M=2*mu*(1-nu)/(1-2*nu)
    Winkler={'F':M*math.pi*R*delta*delta/t,
             'a':math.sqrt(2*R*delta-delta*delta),
             'pmax':M*delta/t}
    base={'F':F,'a':a,'pmax':pmax}
    def errors(other):
        return {key:abs(other[key]/base[key]-1) for key in base}
    eh=errors(Hertz); ew=errors(Winkler)
    if a/t<.3 and max(eh.values())<.02:
        selected='halfspace'
        band=eh
    elif nu<=.3 and a/t>20 and max(ew.values())<.02:
        selected='thin_winkler'
        band=ew
    else:
        selected='hankel'
        band=None
    return {'selected':selected,'a_over_t':a/t,'reference':base,
            'halfspace':Hertz,'thin_winkler':Winkler,
            'halfspace_discrepancy':eh,'thin_discrepancy':ew,
            'selected_discrepancy_vs_hankel':band,
            'empirical_hankel_force_grid_max_vs_FE192':.013004,
            'band_status':'pointwise model discrepancy; no uniform FE certificate'}
