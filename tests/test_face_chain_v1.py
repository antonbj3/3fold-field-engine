import unittest,copy,json,time
from pathlib import Path
from field_engine.face_chain_v1 import verify,fixture
class TestChain(unittest.TestCase):
    def setUp(self):
        # Every regression pairs a known valid certificate with a feasible case.
        self.assertEqual(verify(**fixture(1))['verdict'], 'NO')
        feasible = fixture(1)
        feasible['b'] = [1, 1]
        self.assertEqual(verify(**feasible)['verdict'], 'UNKNOWN')

    def test_exact_chains(self):
        for n in (1,2,4,8):
            self.assertEqual(verify(**fixture(n))['verdict'],'NO')
    def test_omit_necessary_face(self):
        z=fixture(2);z['steps']=z['steps'][:1]
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_reject_feasible_neighbor(self):
        z=fixture(1);z['b'][0]=1 # a=1,b=1,c=1 feasible
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_reject_feasible_boundary(self):
        z=fixture(1);z['b'][-1]=0 # a=b=c=0 feasible
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_reject_float(self):
        z=fixture(1);z['A'][0][0]=1.0
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_reject_wrong_orientation(self):
        z=fixture(1);z['steps'][0]['side']='c'
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_reject_wrong_block(self):
        z=fixture(2);z['steps'][0]['block']=1
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_reject_zero_separator(self):
        z=fixture(1);z['terminal']=[0]*3
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_reject_nonzero_linear_residual(self):
        z=fixture(1);z['terminal']=[1,1,-1]
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_reject_overlapping_blocks(self):
        z=fixture(2);z['blocks'][1]=z['blocks'][0]
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_c_face(self):
        z=fixture(1);z['A'][0]=[0,0,1];z['steps'][0]['side']='c'
        self.assertEqual(verify(**z)['verdict'],'NO')
    def test_scaling_equalities(self):
        z=fixture(1);z['A'][0]=[3,0,0];z['steps'][0]['y']=['1/3',0]
        self.assertEqual(verify(**z)['verdict'],'NO')
    def test_invalid_fraction(self):
        z=fixture(1);z['steps'][0]['y']=['1/0',0]
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_bad_dimension(self):
        z=fixture(1);z['A'][0]=[1]
        self.assertEqual(verify(**z)['verdict'],'UNKNOWN')
    def test_review_near_zero_axis_is_feasible_not_a_face(self):
        z=fixture(1); z['b'][0]='1/1000000000000000000000000000000'
        self.assertEqual(verify(**z)['verdict'], 'UNKNOWN')
    def test_review_permuted_block_axis(self):
        z=fixture(1); z['blocks']=[(2,1,0)]; z['steps'][0]['side']='c'
        self.assertEqual(verify(**z)['verdict'], 'NO')
        z['b'][0]=1
        self.assertEqual(verify(**z)['verdict'], 'UNKNOWN')
    def test_review_boolean_index_is_rejected(self):
        z=fixture(1); z['blocks']=[(False,1,2)]
        self.assertEqual(verify(**z)['verdict'], 'UNKNOWN')
    def test_review_strong_negative_axis_needs_different_certificate(self):
        z=dict(A=[[1,0,0]],b=[-1],blocks=[(0,1,2)],steps=[],terminal=[1])
        self.assertEqual(verify(**z)['verdict'], 'UNKNOWN')
if __name__=='__main__':
    start=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestChain)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    Path('raw/face_tests.json').write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'wall_s':time.perf_counter()-start},indent=2)+'\n')
    Path('raw/face_certificates.json').write_text(json.dumps([dict(input=fixture(n),output=verify(**fixture(n))) for n in (1,2,4)],indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
