"""An invalid binary32 axis cannot certify a finite box SDF sign."""
from fractions import Fraction
import numpy as np
from field_engine.experimental.certify_refine.certify_refine import Frame, decide
from field_engine.experimental.certify_refine.proven_adapters import SDF


def _first(value):
    return value[0] if isinstance(value, tuple) else value


def test_box_overflow_is_uncertain_in_each_axis_order():
    frame = Frame((0.0, 0.0, 0.0))
    for axis in range(3):
        center = [0.0] * 3
        point = [0.0] * 3
        center[axis] = 3.9e38
        point[axis] = 4.0e38
        node = {'op': 'box', 'center': center, 'half_extents': [1.0] * 3}
        exact = abs(Fraction(point[axis]) - Fraction(center[axis])) - 1
        assert exact > 0
        with np.errstate(all='ignore'):
            value, band = SDF().cheap((node, tuple(point), frame))
        assert decide(_first(value), _first(band)) == 'OSÄKER'


def test_finite_box_decisions_remain_decisive():
    frame = Frame((0.0, 0.0, 0.0))
    node = {'op': 'box', 'center': [0.0] * 3, 'half_extents': [1.0] * 3}
    assert decide(*SDF().cheap((node, (0.0, 0.0, 0.0), frame))) == 'INNE'
    assert decide(*SDF().cheap((node, (2.0, 0.0, 0.0), frame))) == 'UTE'
