"""Regression checks for the mathematical contract, not speed assertions."""
import unittest
import json
from fractions import Fraction as F
import numpy as np
from field_engine.experimental.ccd_band import Band, certify, exact_ccd, integer_filter, SAFE, COLLISION, UNKNOWN, _vertices, _contact, _eval, _sign


class CCDTests(unittest.TestCase):
    def test_integer_filter_exclusion_and_rational_contact_without_backend(self):
        import builtins
        from unittest.mock import patch
        a=np.array([[.25,.25,2.**-28],[0,0,0],[1,0,0],[0,1,0]])
        b=a.copy();b[0,2]*=-1
        real_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='sympy':raise ImportError('exact backend unavailable')
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',guarded):
            free=integer_filter(np.stack([a,a]),'PT')
            hit=integer_filter(np.stack([a,b]),'PT')
        self.assertEqual(free.status,SAFE)
        self.assertEqual(free.method,'integer_bernstein')
        self.assertEqual(hit.status,COLLISION)
        self.assertEqual(hit.method,'integer_rational_witness')
        self.assertEqual(F(hit.witness),F(1,2))
        # Check the returned rational witness against the explicit planar fixture.
        z=F(a[0,2])+F(hit.witness)*(F(b[0,2])-F(a[0,2]))
        self.assertEqual(z,0)
        self.assertGreaterEqual(F(a[0,0]),0)
        self.assertGreaterEqual(F(a[0,1]),0)
        self.assertLessEqual(F(a[0,0])+F(a[0,1]),1)

    def test_tunnelling_and_contact_at_closed_endpoint(self):
        a=np.array([[.25,.25,1.],[0,0,0],[1,0,0],[0,1,0]])
        for z in (-1.,0.):
            b=a.copy();b[0,2]=z
            self.assertEqual(certify(np.stack([a,b]),'PT').status,COLLISION)

    def test_crossing_plane_outside_triangle_is_safe(self):
        a=np.array([[2,2,1.],[0,0,0],[1,0,0],[0,1,0]])
        b=a.copy();b[0,2]=-1
        self.assertEqual(certify(np.stack([a,b]),'PT').status,SAFE)

    def test_edge_edge_interior_and_disjoint(self):
        a=np.array([[0,.5,0],[1,.5,0],[.5,0,1],[.5,1,1]],float)
        b=a.copy();b[2:,2]=-1
        self.assertEqual(certify(np.stack([a,b]),'EE').status,COLLISION)
        a[2:,0]=2;b[2:,0]=2
        self.assertEqual(certify(np.stack([a,b]),'EE').status,SAFE)

    def test_irrational_contact_and_double_root(self):
        for z in (1/32,1/16):
            a=np.array([[.25,.25,z],[0,0,0],[1,0,0],[0,1,0]])
            b=a.copy();b[2,1]=1;b[3,2]=1
            d=exact_ccd(np.stack([a,b]),'PT')
            self.assertEqual(d.status,COLLISION)
            if z==1/32:self.assertEqual(d.method,'exact_algebraic_root')

    def test_cubic_interior_crossing(self):
        a=np.array([[.25,.25,1/32],[0,0,0],[1,0,0],[0,1,0]])
        b=a.copy();b[0,0]+=1/8;b[2,1]=1;b[3,2]=1
        self.assertEqual(exact_ccd(np.stack([a,b]),'PT').status,COLLISION)

    def test_missing_exact_backend_abstains(self):
        import builtins
        from unittest.mock import patch
        a=np.array([[.25,.25,1/32],[0,0,0],[1,0,0],[0,1,0]])
        b=a.copy();b[2,1]=1;b[3,2]=1
        real_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='sympy':raise ImportError('deliberately unavailable')
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',guarded):
            self.assertEqual(exact_ccd(np.stack([a,b]),'PT').status,UNKNOWN)

    def test_degenerate_triangle_and_parallel_edges(self):
        x=np.array([[.5,0,0],[0,0,0],[1,0,0],[2,0,0]],float)
        self.assertEqual(certify(np.stack([x,x]),'PT').status,COLLISION)
        x[0,1]=1
        self.assertEqual(certify(np.stack([x,x]),'PT').status,SAFE)
        x=np.array([[0,0,0],[1,0,0],[.5,0,0],[2,0,0]],float)
        self.assertEqual(certify(np.stack([x,x]),'EE').status,COLLISION)

    def test_coplanar_non_sampled_contact_abstains(self):
        a=np.array([[0,0,0],[1,0,0],[.2,-.1,0],[.2,-.05,0]],float)
        b=a.copy();b[2:,1]+=1
        # Actual contact exists at t=0.075, outside the reserve's three samples.
        self.assertEqual(exact_ccd(np.stack([a,b]),'EE').status,UNKNOWN)

    def test_nonfinite_overflow_and_subnormal(self):
        x=np.array([[.25,.25,1.],[0,0,0],[1,0,0],[0,1,0]])
        for scale in (2.**-140,2.**-530,2.**500):
            y=x*scale
            self.assertEqual(certify(np.stack([y,y]),'PT').status,SAFE)
        x[0,0]=np.nan
        self.assertEqual(certify(np.stack([x,x]),'PT').status,UNKNOWN)

    def test_exact_input_after_large_translation(self):
        x=np.array([[.25,.25,2.**-40],[0,0,0],[1,0,0],[0,1,0]])
        y=x+2.**20
        # The represented world has lost its gap; this must never be SAFE.
        self.assertEqual(certify(np.stack([y,y]),'PT').status,COLLISION)

    def test_arithmetic_encloses_exact_values(self):
        rng=np.random.default_rng(20261002)
        for _ in range(300):
            a=float(np.ldexp(rng.uniform(-1,1),int(rng.integers(-150,80))))
            b=float(np.ldexp(rng.uniform(-1,1),int(rng.integers(-150,80))))
            A,B=Band.from64(a),Band.from64(b)
            for z,truth in ((A+B,F(a)+F(b)),(A-B,F(a)-F(b)),(A*B,F(a)*F(b))):
                lo,hi=float(z.lower()),float(z.upper())
                if np.isfinite(lo) and np.isfinite(hi):
                    self.assertLessEqual(F(lo),truth);self.assertGreaterEqual(F(hi),truth)

    def test_represented_float32_narrow_gap_miss_regression(self):
        # Upstream Ando 493c1fb grants toi=1 at production ghat/eps on this
        # initially separated point/triangle sweep; exact contact is t=1/2.
        a=np.array([[.25,.25,2.**-28],[0,0,0],[1,0,0],[0,1,0]],np.float32).astype(float)
        b=a.copy();b[0,2]*=-1
        self.assertEqual(certify(np.stack([a,b]),'PT').status,COLLISION)
        b=a.copy();b[0,0]+=.125
        self.assertEqual(certify(np.stack([a,b]),'PT').status,SAFE)


if __name__=='__main__': unittest.main()
