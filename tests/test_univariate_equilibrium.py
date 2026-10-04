import unittest,copy
from fractions import Fraction as Q
from field_engine.univariate_equilibrium import certify,verify

class TestUnivariateEquilibrium(unittest.TestCase):
    def test_nh_truss_is_globally_three_even_with_full_local_rank(self):
        c=(Q(1,1000),-Q(1,25),Q(1,1000),1);answer=certify(c)
        self.assertTrue(verify(answer,coefficients=c));self.assertTrue(answer['complete'])
        self.assertEqual(answer['status'],'GRENMANGD');self.assertEqual(len(answer['roots']),3)

    def test_coverage_deletion_and_wrong_query_rejected(self):
        c=(Q(1,1000),-Q(1,25),Q(1,1000),1);answer=certify(c)
        broken=copy.deepcopy(answer);broken['tiles'].pop()
        self.assertFalse(verify(broken,coefficients=c))
        self.assertFalse(verify(answer,coefficients=(Q(1,50),-Q(1,25),Q(1,50),1)))

    def test_fake_newton_box_and_lost_branch_rejected(self):
        c=(Q(1,1000),-Q(1,25),Q(1,1000),1);answer=certify(c)
        broken=copy.deepcopy(answer);broken['roots'][0]['box']=['0','1/1000000000000']
        self.assertFalse(verify(broken,coefficients=c))
        broken=copy.deepcopy(answer);broken['roots'].pop()
        self.assertFalse(verify(broken,coefficients=c))

    def test_budget_exhaustion_cannot_be_promoted_by_flag(self):
        c=(Q(1,1000),-Q(1,25),Q(1,1000),1);answer=certify(c,max_depth=1)
        self.assertTrue(verify(answer,coefficients=c));self.assertEqual(answer['status'],'OSAKER')
        answer['complete']=True;answer['status']='ENTYDIG'
        self.assertFalse(verify(answer,coefficients=c))

    def test_zero_polynomial_and_singular_root_keep_unknown(self):
        for c in [(0,), (0,0,0,1)]:
            answer=certify(c,max_depth=25)
            self.assertTrue(verify(answer,coefficients=c));self.assertFalse(answer['complete'])
            self.assertEqual(answer['status'],'OSAKER')

    def test_unique_and_empty_root_sets(self):
        c=(Q(1,50),-Q(1,25),Q(1,50),1);answer=certify(c)
        self.assertTrue(verify(answer,coefficients=c));self.assertEqual(answer['status'],'ENTYDIG')
        answer=certify((1,0,1));self.assertTrue(verify(answer,coefficients=(1,0,1)))
        self.assertTrue(answer['complete']);self.assertEqual(answer['roots'],[])

    def test_floating_inputs_cannot_claim_exactness(self):
        for x in [0.1,float('nan'),True]:
            with self.assertRaises(TypeError):certify((x,1))




class TestReplayPremises(unittest.TestCase):
    def setUp(self):
        self.c=(Q(1,1000),-Q(1,25),Q(1,1000),1)
        self.answer=certify(self.c)

    def test_interior_coverage_hole_rejected(self):
        broken=copy.deepcopy(self.answer)
        # Removing a final tile tests the bound endpoint; an interior hole
        # independently tests adjacency of the claimed complete cover.
        interior=next(i for i,t in enumerate(broken['tiles'])
                      if 0<i<len(broken['tiles'])-1 and t['kind']=='EXCLUDED')
        del broken['tiles'][interior]
        self.assertFalse(verify(broken,coefficients=self.c))

    def test_unreferenced_root_in_partial_certificate_rejected(self):
        broken=copy.deepcopy(self.answer)
        tile=next(t for t in broken['tiles'] if t['kind']=='ROOT')
        tile.clear();tile.update(interval=next(t['interval'] for t in self.answer['tiles']
                                            if t['kind']=='ROOT'),kind='UNKNOWN')
        broken['complete']=False;broken['status']='OSAKER'
        self.assertFalse(verify(broken,coefficients=self.c))

    def test_changed_newton_image_or_derivative_rejected(self):
        for key in ('newton','derivative'):
            broken=copy.deepcopy(self.answer)
            broken['roots'][0][key]=['0','0']
            self.assertFalse(verify(broken,coefficients=self.c))

if __name__=='__main__':unittest.main()
