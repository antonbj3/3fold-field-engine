from fractions import Fraction as F
import unittest
from field_engine.passage_support_v1 import balanced_support_exclusion


class PassageSupportTests(unittest.TestCase):
    def setUp(self):
        self.b = [[F(-2, 3), F(-1, 3)], [F(4, 3), F(-1, 3)], [F(-2, 3), F(2, 3)]]
        self.a = [[-x for x in v] for v in self.b]
        self.normals = [[-1, 0], [0, -1], [F(1, 2), F(1, 2)]]
        self.weights = [F(1, 4), F(1, 4), F(1, 2)]

    def test_equal_widths_do_not_imply_joint_translation_feasibility(self):
        # A=-B has exactly equal widths to B in every direction.
        for n in self.normals:
            paired = balanced_support_exclusion(self.a, self.b, [n, [-x for x in n]], [F(1, 2)] * 2)
            self.assertEqual(paired.status, "UNKNOWN")
            self.assertEqual(paired.obstruction, 0)
        result = balanced_support_exclusion(self.a, self.b, self.normals, self.weights)
        self.assertEqual(result.status, "CERTIFIED_NO_TRANSLATION")
        self.assertEqual(result.obstruction, F(1, 4))

    def test_translation_invariance(self):
        translated = [[v[0] + 100, v[1] - 72] for v in self.a]
        result = balanced_support_exclusion(translated, self.b, self.normals, self.weights)
        self.assertEqual(result.obstruction, F(1, 4))

    def test_uncertain_geometry_and_touching_remain_unknown(self):
        small = balanced_support_exclusion(self.a, self.b, self.normals, self.weights,
                                            inner_error_upper=F(1, 20), aperture_error_upper=F(1, 20))
        self.assertEqual(small.status, "CERTIFIED_NO_TRANSLATION")
        large = balanced_support_exclusion(self.a, self.b, self.normals, self.weights,
                                            inner_error_upper=F(1, 8), aperture_error_upper=F(1, 8))
        self.assertEqual(large.status, "UNKNOWN")
        identical = balanced_support_exclusion(self.b, self.b, self.normals, self.weights)
        self.assertEqual(identical.status, "UNKNOWN")
        self.assertIn("fixed projected shapes", identical.scope)

    def test_inconsistent_duals_and_inexact_inputs_refused(self):
        with self.assertRaises(ValueError):
            balanced_support_exclusion(self.a, self.b, [[1, 0]], [1])
        with self.assertRaises(ValueError):
            balanced_support_exclusion(self.a, self.b, self.normals, [-1, 1, 1])
        with self.assertRaises(TypeError):
            balanced_support_exclusion([[0.1, 0]], self.b, self.normals, self.weights)

    def test_numpy_float_inputs_refused(self):
        import numpy as np
        for bad in (np.float64(0.1), np.float32(0.1)):
            with self.assertRaises(TypeError):
                balanced_support_exclusion([[bad, 0]], self.b, self.normals, self.weights)
            with self.assertRaises(TypeError):
                balanced_support_exclusion(self.a, self.b, self.normals, self.weights,
                                            inner_error_upper=bad)


if __name__ == "__main__":
    unittest.main()
