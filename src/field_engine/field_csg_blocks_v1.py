"""Aligned sampled-field CSG on adaptive blocks, with lossless output packing.

Negative values denote interior. Min/max preserve level-set signs but generally
are not Euclidean signed distances. Construction uses a transient dense output;
operands are gathered from blocks in bounded chunks and are never densified.
"""
from numbers import Real
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks


def compose(left, right=None, *, operation='union', distance=0.):
    if not isinstance(left, AdaptiveSampleBlocks):
        raise ValueError('Adaptive block operand required')
    if operation not in ('union', 'difference', 'offset'):
        raise ValueError('Unknown field operation')
    if operation == 'offset':
        if right is not None or isinstance(distance, (bool, np.bool_)) or not isinstance(distance, Real) or not np.isfinite(distance):
            raise ValueError('Finite real offset and no second operand required')
    else:
        if not isinstance(distance, Real) or isinstance(distance, (bool, np.bool_)) or distance != 0:
            raise ValueError('Offset is only valid for the offset operation')
        if not isinstance(right, AdaptiveSampleBlocks):
            raise ValueError('Second adaptive block operand required')
        if left.shape != right.shape or left.pitch != right.pitch or not np.array_equal(left.origin, right.origin):
            raise ValueError('CSG operands must share the exact sampling grid')
    output = np.empty(left.shape, dtype=np.float32)
    flat = output.ravel()
    for start in range(0, output.size, 8192):
        stop = min(start+8192, output.size)
        indices = np.array(np.unravel_index(np.arange(start, stop), left.shape)).T
        a = left.samples_at(indices)
        if operation == 'offset':
            # Explicit float64 subtraction then a single float32 rounding.
            values = a.astype(np.float64)-float(distance)
        else:
            b = right.samples_at(indices)
            values = np.minimum(a,b) if operation == 'union' else np.maximum(a,-b)
        with np.errstate(over='ignore', invalid='ignore'):
            flat[start:stop] = values
    if not np.isfinite(output).all():
        raise ValueError('Operation exceeds finite float32 range')
    return AdaptiveSampleBlocks(output, left.origin, left.pitch)


if __name__ == '__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_csg_probe import main
    raise SystemExit(main())
