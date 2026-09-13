"""Bounded independent separable integer EDT for mixed boolean 3D masks."""
import numpy as np


def validate(mask):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise ValueError('Expected a boolean array')
    if mask.ndim != 3 or min(mask.shape) < 1 or max(mask.shape) > 128 or mask.size > 262144:
        raise ValueError('Expected bounded nonempty 3D shape')
    if not mask.any() or mask.all():
        raise ValueError('Both classes must be present')


def squared_distance(binary):
    """Distance to zero sites; exact int64 minima, without feature-index search."""
    validate(binary)
    infinity = sum((n - 1)**2 for n in binary.shape) + 1
    distance = np.where(binary, infinity, 0).astype(np.int64)
    for axis, length in enumerate(binary.shape):
        source = np.moveaxis(distance, axis, -1)
        result = np.empty_like(source)
        positions = np.arange(length, dtype=np.int64)
        for target in range(length):
            result[..., target] = np.min(source + (positions - target)**2, axis=-1)
        distance = np.moveaxis(result, -1, axis)
    return np.ascontiguousarray(distance)


def signed_distance(mask, pitch):
    """Preserve measured double reconstruction and float32 half-pitch correction."""
    validate(mask)
    if isinstance(pitch, (bool, np.bool_)) or not np.isscalar(pitch):
        raise ValueError('Expected scalar isotropic pitch')
    try:
        pitch = float(pitch)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('Expected real pitch') from error
    if not np.isfinite(pitch) or not 1e-12 <= pitch <= 1e12:
        raise ValueError('Pitch outside bounded contract')
    outside = (np.sqrt(squared_distance(~mask).astype(np.float64))*pitch).astype(np.float32)
    inside = (np.sqrt(squared_distance(mask).astype(np.float64))*pitch).astype(np.float32)
    signed = (outside-inside).astype(np.float32)
    return (signed-np.sign(signed)*np.float32(0.5*pitch)).astype(np.float32)
