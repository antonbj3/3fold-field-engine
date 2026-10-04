import copy
from fractions import Fraction as Q
import unittest
from field_engine.experimental.homogeneous_boundary import (
    add, scale, derivative, multiply, encode, integrate, surface_flux,
    boundary_certificate, verify_boundary_certificate, full_zero_trace,
)

ROD={'kind':'rod','length':'10','width':'1','depth':'1'}
ARCH={'kind':'parabolic_arch','thickness':'1/4','depth':'1/4'}

def rod_bending(t=Q(4,5)):
    L=Q(10)
    w={(3,0,0):1/L**3,(4,0,0):-3/L**4,(5,0,0):3/L**5,(6,0,0):-1/L**6}
    iw={(k[0]+1,0,0):v/Q(k[0]+1) for k,v in w.items()}
    psi=add(scale(iw,-t),scale(multiply({(0,2,0):Q(1)},derivative(w,0)),-1/(2*t**5)))
    return [scale(derivative(psi,1),t*t),scale(derivative(psi,0),-1/t),{}]

def rod_bulk(t=Q(4,5)):
    xb={(1,0,0):Q(10),(2,0,0):Q(-1)}
    yb={(0,2,0):Q(1),(0,0,0):Q(-1,4)}
    zb={(0,0,2):Q(1),(0,0,0):Q(-1,4)}
    psi=multiply(multiply(multiply(xb,xb),multiply(yb,yb)),multiply(zb,zb))
    return [scale(derivative(psi,1),t*t),scale(derivative(psi,0),-1/t),{}]

class BoundaryTests(unittest.TestCase):
    def cert(self,t=Q(4,5)):
        return boundary_certificate(t=t,geometry=ROD,field=encode(rod_bending(t)))

    def test_exact_negative_free_side_witness_and_nonzero_flux(self):
        c=self.cert()
        self.assertEqual(c['status'],'PROVED_UNSTABLE')
        self.assertLess(Q(c['quadratic_variation']),0)
        self.assertLess(Q(c['pressure_boundary_flux']),0)
        self.assertEqual(c['pressure_boundary_flux'],c['pressure_volume_form'])
        self.assertFalse(c['full_zero_trace_field'])

    def test_bulk_guarantee_does_not_certify_whole_body(self):
        c=boundary_certificate(t='4/5',geometry=ROD,field=encode(rod_bulk()))
        self.assertEqual(c['pressure_volume_form'],'0')
        self.assertEqual(Q(c['quadratic_variation']),Q(c['gradient_norm_squared']))
        self.assertEqual(c['bulk_kernel_gradient_floor'],'1')
        self.assertTrue(c['full_zero_trace_field'])
        self.assertFalse(c['full_space_positive_coercivity'])
        self.assertEqual(c['remaining_boundary_spectrum'],'UNVERIFIED')
        self.assertEqual(c['status'],'OSAKER')

    def test_arch_volume_and_curved_surface_normals(self):
        self.assertEqual(integrate({(0,0,0):Q(1)},ARCH),Q(1,8))
        self.assertEqual(integrate({(0,1,0):Q(1)},ARCH),Q(19,192))
        # div(0,y,0)=1; sloping top and bottom faces must both be included.
        self.assertEqual(surface_flux([{}, {(0,1,0):Q(1)}, {}],ARCH),Q(1,8))
        self.assertEqual(surface_flux([{(1,0,0):Q(1)}, {}, {}],ARCH),Q(1,8))
        # div(x*y,0,0)=y. Top/bottom slope terms cannot cancel here.
        self.assertEqual(surface_flux([{(1,1,0):Q(1)}, {}, {}],ARCH),Q(19,192))

    def test_arch_has_actual_three_component_admissible_mode(self):
        b={(0,0,0):Q(1),(2,0,0):Q(-2),(4,0,0):Q(1)}
        p=multiply(b,{(0,0,2):Q(1)})
        t=Q(4,5)
        u=[scale(derivative(p,2),t*t),{},scale(derivative(p,0),-1/t)]
        c=boundary_certificate(t=t,geometry=ARCH,field=encode(u))
        self.assertEqual(c['tangent_divergence'],'0')
        self.assertEqual(c['traction_contract'],'MANUFACTURED_DEAD_PIOLA_TRACTIONS')
        self.assertGreater(Q(c['l2_norm_squared']),0)
        self.assertTrue(verify_boundary_certificate(c))

    def test_zero_trace_bulk_addition_leaves_pressure_unchanged(self):
        u=rod_bending();b=rod_bulk()
        c=self.cert()
        mixed=[add(a,scale(v,Q(17,100))) for a,v in zip(u,b)]
        cm=boundary_certificate(t='4/5',geometry=ROD,field=encode(mixed))
        self.assertEqual(c['pressure_volume_form'],cm['pressure_volume_form'])
        self.assertNotEqual(c['gradient_norm_squared'],cm['gradient_norm_squared'])

    def test_false_positive_spectral_claim_and_pressure_tamper_rejected(self):
        c=self.cert()
        for key,value in (('full_space_positive_coercivity',True),('status','PROVED_STABLE'),
                          ('bulk_kernel_gradient_floor','100'),('pressure_boundary_flux','0'),
                          ('quadratic_variation','-1')):
            changed=copy.deepcopy(c);changed[key]=value
            self.assertFalse(verify_boundary_certificate(changed))

    def test_positive_witness_never_certifies_full_stability(self):
        c=self.cert(Q(1))
        self.assertGreater(Q(c['quadratic_variation']),0)
        self.assertEqual(c['status'],'OSAKER')
        self.assertFalse(c['full_space_positive_coercivity'])

    def test_end_violation_and_divergence_violation_rejected(self):
        u=encode(rod_bending());u[1]['0,0,0']='1'
        with self.assertRaises(ValueError):boundary_certificate(t='4/5',geometry=ROD,field=u)
        u=encode(rod_bending());u[0]['1,0,0']='1';u[0]['2,0,0']='-1/10'
        with self.assertRaises(ValueError):boundary_certificate(t='4/5',geometry=ROD,field=u)

    def test_model_geometry_and_input_types_are_bound(self):
        for t in (True,0.8,'0','2'):
            with self.assertRaises((TypeError,ValueError)):
                boundary_certificate(t=t,geometry=ROD,field=encode(rod_bending()))
        with self.assertRaises(ValueError):
            boundary_certificate(t='4/5',geometry={**ARCH,'rise':'2'},field=encode(rod_bending()))
        with self.assertRaises(ValueError):boundary_certificate(t='4/5',geometry=ROD,field=[{}, {}, {}])

    def test_malformed_verifier_payloads_refuse(self):
        for payload in (None,False,[],{}, {'inputs':None},{'inputs':[]},{'inputs':{'geometry':[]}}):
            self.assertFalse(verify_boundary_certificate(payload))

if __name__=='__main__':unittest.main()
