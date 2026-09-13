"""Prepared local coordinate intervals enclosing frozen world-space predicates."""
import numpy as np
from rt_interval_portable_v1 import fixtures as portable_fixtures, multiply, frozen_edges


def subtract(a, b, upper):
    a, b = np.broadcast_arrays(np.asarray(a, np.float32), np.asarray(b, np.float32))
    x, y = a.astype(np.float64), -b.astype(np.float64)
    high = x + y
    virtual_y = high - x
    residual = (x - (high - virtual_y)) + (y - virtual_y)
    rounded = high.astype(np.float32)
    value = rounded.astype(np.float64)
    if upper:
        needs = (value < high) | ((value == high) & (residual > 0))
        result = np.where(needs, np.nextafter(rounded, np.float32(np.inf)), rounded)
    else:
        needs = (value > high) | ((value == high) & (residual < 0))
        result = np.where(needs, np.nextafter(rounded, np.float32(-np.inf)), rounded)
        opposite_zeros = (a == 0) & (b == 0) & (np.signbit(a) != np.signbit(b))
        result = np.where((a == b) & ~opposite_zeros, np.float32(-0.), result)
    return result


def fixtures():
    triangles, queries = portable_fixtures()
    anchors = triangles.min(axis=1).copy()
    anchors[::4] = 0.
    return triangles, queries, anchors


def prepare(points, anchors):
    rounded = (points - anchors).astype(np.float32)
    low = np.nextafter(rounded, np.float32(-np.inf))
    high = np.nextafter(rounded, np.float32(np.inf))
    return np.stack((low[..., 0], high[..., 0], low[..., 1], high[..., 1]), axis=-1)


def bounds(triangles, queries, anchors):
    geometry = prepare(triangles[:, :, :2], anchors[:, None, :2])
    query = prepare(queries, anchors[:, :2])
    lo = np.stack((subtract(geometry[:, :, 0], query[:, None, 1], False),
                   subtract(geometry[:, :, 2], query[:, None, 3], False)), axis=-1)
    hi = np.stack((subtract(geometry[:, :, 1], query[:, None, 0], True),
                   subtract(geometry[:, :, 3], query[:, None, 2], True)), axis=-1)
    ulo, uhi = np.roll(lo, -1, axis=1), np.roll(hi, -1, axis=1)
    vlo, vhi = np.roll(lo, -2, axis=1), np.roll(hi, -2, axis=1)
    left_lo, left_hi = multiply(ulo[:, :, 0], uhi[:, :, 0], vlo[:, :, 1], vhi[:, :, 1])
    right_lo, right_hi = multiply(ulo[:, :, 1], uhi[:, :, 1], vlo[:, :, 0], vhi[:, :, 0])
    lower, upper = subtract(left_lo, right_hi, False), subtract(left_hi, right_lo, True)
    reject = (lower > 0).any(1) & (upper < 0).any(1)
    return geometry, query, lower, upper, reject
