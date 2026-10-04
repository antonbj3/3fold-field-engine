import copy,unittest
from fractions import Fraction as Q
from field_engine.experimental.continuum_stability import *

# This smooth bending velocity fixes both complete end faces and is exactly
# tangent-incompressible. It is a continuum test field, not an eigensolver fixture.
def bending_field(t=Q(4,5),L=Q(10),z_dependent=False):
    # w=(X/L)^3(1-X/L)^3; Ψ= -t int w dX − (1/(2t^5))Y² w_X.
    w={(3,0,0):1/L**3,(4,0,0):-3/L**4,(5,0,0):3/L**5,(6,0,0):-1/L**6}
    integral_w={(k[0]+1,0,0):v/Q(k[0]+1) for k,v in w.items()}
    psi=add(scale(integral_w,-t),scale(multiply({(0,2,0):Q(1)},derivative(w,0)),-Q(1,2)/t**5))
    if z_dependent:psi=multiply(psi,{(0,0,0):Q(1),(0,0,1):Q(1,10)})
    return [scale(derivative(psi,1),t*t),scale(derivative(psi,0),-1/t),{}]

class StabilityTests(unittest.TestCase):
    def cert(self,**kwargs):
        t=kwargs.pop('t',Q(4,5));return rod_certificate(t=t,length='10',field=encode(bending_field(t)),**kwargs)
    def test_negative_continuum_direction_and_replay(self):
        c=self.cert();self.assertEqual(c['status'],'PROVED_UNSTABLE');self.assertTrue(verify(c));self.assertLess(Q(c['quadratic_variation']),0)
    def test_positive_ritz_witness_never_certifies_stability(self):
        c=self.cert(t=Q(1));self.assertEqual(c['status'],'OSAKER');self.assertGreater(Q(c['quadratic_variation']),0);self.assertFalse(c['full_space_positive_coercivity'])
    def test_equilibrium_and_reference_reaction(self):
        c=self.cert();self.assertEqual(c['det_F'],'1');self.assertEqual(c['lateral_piola'],'0');self.assertEqual(c['equilibrium_residual'],'0');self.assertEqual(Q(c['nominal_compressive_stress']),Q(4,5)**-4-Q(4,5)**2)
    def test_all_matrix_blocks_obey_analytic_floor(self):
        t=Q(4,5);f=(t*t,1/t,1/t);floor=1-t**-3
        for i,j in product(range(3),repeat=2):
            for sign in (-1,1):
                H=[[Q(0)]*3 for _ in range(3)];H[i][j]+=1;H[j][i]+=sign
                n=sum(a*a for row in H for a in row);a=n+sum(H[k][l]*H[l][k]/(t*t*f[k]*f[l]) for k,l in product(range(3),repeat=2))
                self.assertGreaterEqual(a,floor*n)
    def test_true_z_dependence_admitted(self):
        c=rod_certificate(t='4/5',length='10',field=encode(bending_field(z_dependent=True)));self.assertTrue(verify(c));self.assertNotEqual(c['quadratic_variation'],self.cert()['quadratic_variation'])
    def test_pressure_term_is_necessary_for_instability(self):
        c=self.cert();self.assertGreater(Q(c['gradient_norm_squared']),0);self.assertLess(Q(c['quadratic_variation']),0)
    def test_end_violation_rejected(self):
        u=encode(bending_field());u[1]['0,0,0']='1'
        with self.assertRaises(ValueError):rod_certificate(t='4/5',length='10',field=u)
    def test_non_solenoidal_interior_rejected(self):
        u=encode(bending_field());u[0]['1,0,0']='1';u[0]['2,0,0']='-1/10'
        with self.assertRaises(ValueError):rod_certificate(t='4/5',length='10',field=u)
    def test_zero_field_rejected(self):
        with self.assertRaises(ValueError):rod_certificate(t='4/5',length='10',field=[{}, {}, {}])
    def test_wrong_state_geometry_and_status_tampering(self):
        c=self.cert()
        for key,value in (('status','STABLE'),('full_space_gradient_floor','1'),('pressure','0'),('quadratic_variation','-1')):
            d=copy.deepcopy(c);d[key]=value;self.assertFalse(verify(d))
        for key,value in (('t','9/10'),('length','11'),('mu','2'),('width','2')):
            d=copy.deepcopy(c);d['inputs'][key]=value;self.assertFalse(verify(d))
    def test_float_boolean_and_bad_domains(self):
        for t in (0.8,True,'0','-1','2'):
            with self.assertRaises((TypeError,ValueError)):self.cert(t=t)
        for field in ([{},{}],[{},[],{}],[{'-1,0,0':'1'},{},{}],[{'01,0,0':'1'},{},{}]):
            with self.assertRaises(ValueError):rod_certificate(t='4/5',length='10',field=field)
    def test_direct_polynomial_moment_with_units(self):
        self.assertEqual(integral({(0,2,2):Q(1)},Q(10),Q(2),Q(3)),Q(15))

if __name__=='__main__':unittest.main(verbosity=2)
