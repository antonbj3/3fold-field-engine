import numpy as np

from field_engine.experimental.certify_refine.fpband.raytri import raytri, HIT
from field_engine.experimental.certify_refine.fpband.f32ops import fma32


def _row(v):
    return np.asarray(v, dtype=np.float64).reshape(1, 3)


def test_fma32_overflow_is_not_fixed_up_to_finite():
    a = np.array([np.float32(3.0e38)], dtype=np.float32)
    b = np.array([np.float32(2.0)], dtype=np.float32)
    c = np.array([np.float32(-1.0e-30)], dtype=np.float32)
    assert np.isinf(fma32(a, b, c)[0])
    assert np.isinf(fma32(-a, b, -c)[0])


def test_audit_u396_counterexample_is_not_certified_hit():
    """CLOUD-F-AUDIT-U396 CE1: exact answer is MISS (t* = -1/64); FMA mode must not certify HIT."""
    v0, v1, v2 = _row((-1, -1, -1 / 16)), _row((1, 2**64, 2**59)), _row((2**65, 1, 2**59))
    org, d, cf = _row((0, 0, 0)), _row((0, 0, 1)), _row((0, 0, 0))
    for fma in (False, True):
        res = raytri(v0, v1, v2, org, d, cf, fma)
        assert not np.any(np.asarray(res['dec']) == HIT)
