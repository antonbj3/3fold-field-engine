"""Evaluate ordered CSG instructions in rectangular tiles with direct leaf reads.

Input ids precede result ids. Instruction i produces id len(fields)+i. Each
operation preserves the eager variant's float32 rounding, including offsets;
there is no reassociation. Intermediates are chunk arrays, not packed fields.
"""
from numbers import Real
import operator
import itertools
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks


def gather_tile(field, lower, upper):
    """Read an in-bounds rectangular tile by intersecting disjoint leaf boxes."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    nodes = field.nodes
    mask = (nodes['children'][:,0] < 0) & np.all(nodes['lo'] < upper,axis=1) & np.all(nodes['lo']+nodes['shape'] > lower,axis=1)
    result = np.empty(tuple(upper-lower),dtype=np.float32)
    for node in nodes[mask]:
        lo = np.maximum(lower,node['lo']); hi = np.minimum(upper,node['lo']+node['shape'])
        shape = tuple(node['stored_shape'])
        start = int(node['offset'])
        samples = field.payload[start:start+int(np.prod(shape))].reshape(shape)
        source = tuple(slice(0,1) if shape[a]==1 else slice(lo[a]-node['lo'][a],hi[a]-node['lo'][a]) for a in range(3))
        target = tuple(slice(lo[a]-lower[a],hi[a]-lower[a]) for a in range(3))
        result[target] = samples[source]
    return result


def evaluate(fields, instructions):
    fields, instructions = tuple(fields), tuple(instructions)
    if not 1 <= len(fields) <= 256 or not 1 <= len(instructions) <= 1024:
        raise ValueError('Bounded nonempty inputs and instructions required')
    if any(not isinstance(f, AdaptiveSampleBlocks) for f in fields):
        raise ValueError('Adaptive block operands required')
    first = fields[0]
    if any(f.shape != first.shape or f.pitch != first.pitch or not np.array_equal(f.origin, first.origin) for f in fields):
        raise ValueError('Operands must share the exact sampling grid')
    program = []
    used_inputs = set()
    for number, instruction in enumerate(instructions):
        if not isinstance(instruction, (tuple, list)) or len(instruction) != 3:
            raise ValueError('Three-item instruction required')
        operation, left, right = instruction
        if operation not in ('union', 'difference', 'offset'):
            raise ValueError('Unknown operation')
        def reference(value):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError('Integer reference required')
            try:
                index = operator.index(value)
            except TypeError:
                raise ValueError('Integer reference required') from None
            if not 0 <= index < len(fields)+number:
                raise ValueError('Reference must name an input or earlier instruction')
            if index < len(fields):
                used_inputs.add(index)
            return index
        left = reference(left)
        if operation == 'offset':
            if isinstance(right, (bool, np.bool_)) or not isinstance(right, Real) or not np.isfinite(right):
                raise ValueError('Finite real offset required')
            right = float(right)
        else:
            right = reference(right)
        program.append((operation,left,right))
    # Keep intermediates only until their final consumer, including dead results.
    last_use = {}
    for number, (operation,left,right) in enumerate(program):
        last_use[left] = number
        if operation != 'offset':
            last_use[right] = number
    final_id = len(fields)+len(program)-1
    last_use[final_id] = len(program)
    output = np.empty(first.shape, dtype=np.float32)
    tile_shape = (16,32,16)
    for lower in itertools.product(*(range(0,n,t) for n,t in zip(first.shape,tile_shape))):
        upper = tuple(min(n,l+t) for n,l,t in zip(first.shape,lower,tile_shape))
        target = tuple(slice(l,u) for l,u in zip(lower,upper))
        values = {i: gather_tile(fields[i],lower,upper) for i in used_inputs}
        for number, (operation,left,right) in enumerate(program):
            a = values[left]
            with np.errstate(over='ignore',invalid='ignore'):
                if operation == 'offset':
                    result = (a.astype(np.float64)-right).astype(np.float32)
                else:
                    b = values[right]
                    result = np.minimum(a,b) if operation == 'union' else np.maximum(a,-b)
            if not np.isfinite(result).all():
                raise ValueError('Operation exceeds finite float32 range')
            index = len(fields)+number
            if index in last_use:
                values[index] = result
            for operand in {left} if operation == 'offset' else {left,right}:
                if last_use[operand] == number:
                    del values[operand]
        output[target] = values[final_id]
    return AdaptiveSampleBlocks(output,first.origin,first.pitch)


if __name__ == '__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_csg_tiled_probe import main
    raise SystemExit(main())
