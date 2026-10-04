"""Fixed-model regression cases; run after applying PATCH.diff."""
import copy
import unittest
from fractions import Fraction as Q
from field_engine.experimental.hyperelastic_energy import (
    bar_certificate,scalar_certificate,verify,root_point,log_point)


class HyperelasticEnergyTests(unittest.TestCase):
    def test_exact_homogeneous_continuum(self):
        c=bar_certificate(['3/2']*4,body=0)
        self.assertEqual(c['energy_gap_upper'],'0')
        self.assertTrue(verify(c))

    def test_fe_residual_does_not_erase_body_force_variation(self):
        c=bar_certificate(['3/2'],traction='1/3',body=1)
        self.assertEqual(Q(c['energy_gap_upper']),Q(1,24))

    def test_sphere_radius_against_independent_analytical_reference(self):
        c=scalar_certificate('sphere','1.2128881479','1/1000','1/2')
        self.assertEqual(c['status'],'CERTIFIED')
        x=Q(c['center']);err=Q(c['state_error_upper'])
        self.assertLess(x-err,Q('1.212888147936601650621879'))
        self.assertGreater(x+err,Q('1.212888147936601650621880'))
        self.assertLess(Q(c['energy_gap_upper']),Q(1,10**16))
        self.assertTrue(verify(c))

    def test_arch_stability_and_fold(self):
        self.assertEqual(scalar_certificate('arch','0.18553911','1/100000','1/1000')['status'],'CERTIFIED')
        self.assertEqual(scalar_certificate('arch','0.02542716','1/100000','1/1000')['status'],'OSAKER')
        # Exact fold y≈0.11446, interval below is deliberately wider.
        c=scalar_certificate('arch','0.1145','1/1000','0.003')
        self.assertEqual(c['reason'],'NO_POSITIVE_STABILITY_MARGIN')

    def test_existence_is_required(self):
        c=scalar_certificate('arch','1/5','1/100000','1/1000')
        self.assertEqual(c['reason'],'EXISTENCE_NOT_BRACKETED')

    def test_orientation_exact_inputs_and_scope(self):
        self.assertEqual(bar_certificate([0])['status'],'OSAKER')
        self.assertEqual(bar_certificate([1],scope='FULL_3D')['status'],'OSAKER')
        self.assertEqual(scalar_certificate('sphere','1','1/100','1/2',scope='FULL_3D')['status'],'OSAKER')
        for bad in (1.5,True,float('nan')):
            with self.assertRaises(TypeError):bar_certificate([bad])

    def test_forgery_is_replayed(self):
        c=scalar_certificate('sphere','1.2128881479','1/1000','1/2')
        for key,value in [('energy_gap_upper','0'),('load','1/5'),('scope','FULL_3D')]:
            fake=copy.deepcopy(c);fake[key]=value
            self.assertFalse(verify(fake))

    def test_algebraic_root_enclosures(self):
        for n in (2,3):
            b=root_point(Q(2),n)
            self.assertLessEqual(b.lo**n,2)
            self.assertGreaterEqual(b.hi**n,2)
        self.assertEqual(log_point(1).encoded(),['0','0'])

if __name__=='__main__':unittest.main(verbosity=2)
