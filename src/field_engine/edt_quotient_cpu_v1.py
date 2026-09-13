"""Bounded quotient estimator with exact integer floor correction."""
import numpy as np

BOUND = 1048576
DENSE_DENOMINATORS = (2, 4, 6, 8, 10, 1020, 1022)


def corrected_floor(numerator, denominator):
    n, d = np.broadcast_arrays(np.asarray(numerator, np.int64), np.asarray(denominator, np.int64))
    estimate = np.floor(n.astype(np.float32) / d.astype(np.float32)).astype(np.int64)
    mask = estimate * d > n
    while mask.any():
        estimate[mask] -= 1
        mask = estimate * d > n
    mask = (estimate + 1) * d <= n
    while mask.any():
        estimate[mask] += 1
        mask = (estimate + 1) * d <= n
    half = np.where(n >= 0, n // 2, -((-n + 1) // 2))
    return np.where(d == 2, half, estimate).astype(np.int32)


def boundary_cases():
    denominators = np.arange(2, 1023, 2, dtype=np.int64)
    quotients = np.array([-524288, -262144, -65536, -4096, -512, -2, -1, 0, 1, 2, 512, 4096, 65536, 262144, 524288], np.int64)
    n = quotients[:, None, None] * denominators[None, :, None] + np.array([-1, 0, 1])[None, None, :]
    d = np.broadcast_to(denominators[None, :, None], n.shape)
    selected = np.abs(n) <= BOUND
    return n[selected].astype(np.int32), d[selected].astype(np.int32)
