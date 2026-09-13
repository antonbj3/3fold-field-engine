"""Bounded exact integer EDT axis passes on CUDA; reconstruction remains on CPU."""
import numpy as np
import warp as wp
from field_edt_integer_v1 import validate


@wp.kernel
def axis_minimum(source: wp.array(dtype=wp.int32), target: wp.array(dtype=wp.int32),
                 stride: int, length: int):
    index = wp.tid()
    position = (index // stride) % length
    base = index-position*stride
    best = int(2147483647)
    for site in range(length):
        delta = position-site
        value = source[base+site*stride]+delta*delta
        best = wp.min(best, value)
    target[index] = best


def squared_distance(binary):
    validate(binary)
    sentinel = sum((n-1)**2 for n in binary.shape)+1
    initial = np.where(binary, sentinel, 0).astype(np.int32).ravel()
    source = wp.array(initial, dtype=wp.int32, device='cuda:0')
    target = wp.empty_like(source)
    strides = (binary.shape[1]*binary.shape[2], binary.shape[2], 1)
    for length, stride in zip(binary.shape, strides):
        wp.launch(axis_minimum, dim=binary.size,
                  inputs=[source, target, stride, length], device='cuda:0')
        source, target = target, source
    return source.numpy().reshape(binary.shape)


if __name__ == '__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_edt_parallel_probe import main
    raise SystemExit(main())
