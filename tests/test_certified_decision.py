"""Regression for geometry/physics scope, completion quantifiers and exact margins."""
import dataclasses
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
import unittest
import numpy as np
from field_engine.experimental import certified_decision as d


class CertifiedDecisionTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        self.faces = np.array([[0,1,2]])

    def surface(self, **kw):
        args = dict(axis=2, plane_offset=Q(1,100), coordinate_radius=Q(1,1000),
                    scope='MODEL', units='MODEL_LENGTH')
        args.update(kw)
        return d.surface_plane_contact(self.vertices, self.faces, **args)

    def test_penetrating_observation_for_every_completion(self):
        out = self.surface()
        self.assertEqual(out.status, 'JA')
        self.assertIsNone(out.lower)
        self.assertEqual(Q(out.exact_interval[1]), Q(-9,1000))
        self.assertGreater(Q(out.exact_margin), 0)
        self.assertFalse(out.physical_admission)

    def test_separated_observation_does_not_prove_separated_completion(self):
        out = self.surface(plane_offset=Q(-1,100))
        self.assertEqual(out.status, 'OSÄKER')
        self.assertIn('unseen_surface_support', out.missing)
        # Same observation permits a hidden penetrating point or none.
        self.assertLess(min(Q(1,100), Q(-1)), 0)
        self.assertGreater(Q(1,100), 0)

    def test_support_after_uncertainty_can_prove_separation(self):
        out = self.surface(plane_offset=Q(-1,100), missing_axis_lower=Q(-1,200))
        self.assertEqual(out.status, 'NEJ')
        self.assertEqual(Q(out.exact_margin), Q(1,200))

    def test_threshold_touching_has_no_certain_branch(self):
        for radius, offset in [(0,0),(Q(1,100),Q(1,100)),(Q(1,100),Q(-1,100))]:
            with self.subTest(radius=radius, offset=offset):
                out = self.surface(coordinate_radius=radius, plane_offset=offset, missing_axis_lower=0)
                self.assertEqual(out.status, 'OSÄKER')
                self.assertEqual(Q(out.exact_margin), 0)

    def test_physical_default_and_explicit_physical_never_promote_model(self):
        kw = dict(axis=2,plane_offset=Q(1,100),coordinate_radius=0,units='mm')
        for scope in [None,'PHYSICAL']:
            with self.subTest(scope=scope):
                opts = kw if scope is None else dict(kw,scope=scope)
                out = d.surface_plane_contact(self.vertices,self.faces,**opts)
                self.assertEqual(out.model_status,'JA')
                self.assertEqual(out.status,'OSÄKER')
                self.assertFalse(out.physical_admission)
                self.assertEqual(out.margin_lower,0)

    def test_hash_recomputed_after_input_edit_and_old_snapshot_is_immutable(self):
        first=self.surface()
        self.vertices[:,2]+=1
        second=self.surface()
        self.assertEqual(first.status,'JA')
        self.assertEqual(second.status,'OSÄKER')
        self.assertNotEqual(first.input_sha256,second.input_sha256)
        with self.assertRaises(FrozenInstanceError):first.status='NEJ'

    def test_unused_vertex_cannot_provide_contact_witness(self):
        self.vertices=np.vstack([self.vertices,[0.,0.,-100.]])
        out=self.surface(plane_offset=Q(-1,100))
        self.assertEqual(out.status,'OSÄKER')
        self.assertGreater(out.upper,0)

    def test_invalid_numeric_geometry_and_scope_fail_closed(self):
        for value in [float('nan'),float('inf'),True,np.bool_(True),'0.1']:
            for key in ['plane_offset','coordinate_radius','missing_axis_lower','threshold']:
                with self.subTest(value=value,key=key), self.assertRaises((TypeError,ValueError)):
                    self.surface(**{key:value})
        for opts in [dict(coordinate_radius=-1),dict(axis=True),dict(axis=3),dict(axis=1.0),
                     dict(scope='clinical'),dict(units=''),dict(units=True)]:
            with self.subTest(opts=opts),self.assertRaises((TypeError,ValueError)):
                self.surface(**opts)

    def test_invalid_faces_and_degenerate_surface_refused(self):
        for faces in [[[0,1,4]],[[0,1,-1]],[[0,0,2]],[[0.,1.,2.]],[[True,1,2]],[],[[0,1]]]:
            with self.subTest(faces=faces),self.assertRaises((TypeError,ValueError)):
                d.surface_plane_contact(self.vertices,faces,axis=2,plane_offset=0,coordinate_radius=0,units='MODEL_LENGTH')
        self.vertices[2]=[2,0,0]
        with self.assertRaises(ValueError):self.surface()

    def test_exact_subnormal_margin_and_outward_endpoints(self):
        tiny=Q(1,2**1100)
        out=self.surface(plane_offset=tiny,coordinate_radius=0)
        self.assertEqual(out.status,'JA')
        self.assertEqual(Q(out.exact_margin),tiny)
        self.assertLessEqual(Q(out.upper),0)
        self.assertGreaterEqual(Q(out.upper),-tiny)
        # Zero in binary64 is explicitly supported by the positive exact margin.
        self.assertEqual(out.margin_lower,0)

    def test_raw_closed_or_open_mesh_does_not_admit_resistance(self):
        for scope in ['MODEL','PHYSICAL']:
            out=d.scan_resistance(self.vertices,self.faces,scope=scope,units='MODEL_LENGTH')
            self.assertEqual(out.status,'OSÄKER')
            self.assertIsNone(out.lower)
            self.assertIsNone(out.upper)
            self.assertFalse(out.physical_admission)

    def test_box_model_decisions_and_physical_scope(self):
        for limits,expected,scope in [((Q(199,100),Q(201,100)),'JA','MODEL'),
                                      ((Q(21,10),Q(22,10)),'NEJ','MODEL'),
                                      ((2,Q(201,100)),'OSÄKER','MODEL'),
                                      ((Q(199,100),Q(201,100)),'OSÄKER','PHYSICAL')]:
            with self.subTest(limits=limits,scope=scope):
                out=d.box_resistance((2,1,1),conductivity=1,face_radius=Q(1,10000),limits=limits,scope=scope,units='MODEL_RESISTANCE')
                self.assertEqual(out.status,expected)
                self.assertFalse(out.physical_admission)
                self.assertLess(out.lower,2)
                self.assertGreater(out.upper,2)
                if expected!='OSÄKER':self.assertGreater(Q(out.exact_margin),0)

    def test_box_rejects_topology_collapse_and_invalid_material(self):
        for kw in [dict(face_radius=Q(1,2)),dict(conductivity=0),dict(conductivity=-1),
                   dict(conductivity=True),dict(lengths=(2,1)),dict(limits=(2,2))]:
            with self.subTest(kw=kw),self.assertRaises((TypeError,ValueError)):
                args=dict(lengths=(2,1,1),conductivity=1,face_radius=0,limits=(1,3),scope='MODEL',units='MODEL_RESISTANCE')
                args.update(kw);d.box_resistance(**args)


    def test_review_unused_vertex_enters_separation_bound(self):
        # Review F1: a supplied but unreferenced vertex below the plane must
        # not be ignored by NEJ (it was NEJ with margin 1/200 before).
        self.vertices=np.vstack([self.vertices,[0.,0.,-100.]])
        out=self.surface(plane_offset=Q(-1,100),missing_axis_lower=Q(-1,200))
        self.assertEqual(out.status,'OSÄKER')
        self.assertEqual(Q(out.exact_interval[0]),Q(-100)+Q(1,100)-Q(1,1000))
        # It still cannot provide a JA witness on its own.
        self.assertEqual(self.surface(plane_offset=Q(-1,100)).status,'OSÄKER')

    def test_review_decision_copies_are_refused_or_not_issued(self):
        phys=d.surface_plane_contact(self.vertices,self.faces,axis=2,plane_offset=1,coordinate_radius=0,units='mm')
        self.assertTrue(d.is_issued(phys))
        for edit in [dict(status='JA'),dict(status='NEJ'),dict(physical_admission=True),
                     dict(status='JA',exact_margin='1',margin_lower=1.0),dict(scope='MODEL',status='JA',margin_lower=0.5)]:
            with self.subTest(edit=edit),self.assertRaises(ValueError):
                dataclasses.replace(phys,**edit)
        model=self.surface(plane_offset=Q(-1,100))
        self.assertEqual(model.status,'OSÄKER')
        forged=dataclasses.replace(model,status='NEJ',model_status='NEJ',exact_margin='1',margin_lower=1.0)
        self.assertFalse(d.is_issued(forged))
        self.assertFalse(d.is_issued(dataclasses.replace(model)))
        self.assertTrue(d.is_issued(model))

    def test_review_model_certainty_lists_physical_gap(self):
        out=self.surface()
        self.assertEqual(out.status,'JA')
        self.assertIn('verified_physical_acquisition_and_units',out.missing)
        box=d.box_resistance((2,1,1),conductivity=1,face_radius=Q(1,10000),limits=(Q(199,100),Q(201,100)),scope='MODEL',units='MODEL_RESISTANCE')
        self.assertEqual(box.status,'JA')
        self.assertIn(('geometry_premise','CALLER_ASSERTED_NOT_VERIFIED'),box.contributions)
        self.assertIn('physical_model_applicability',box.missing)


if __name__=='__main__':unittest.main()
