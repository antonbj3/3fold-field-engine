"""Evaluate ordered CSG instructions in chunks and pack only the final field.

Input ids precede result ids. Instruction i produces id len(fields)+i. Each
operation preserves the eager variant's float32 rounding, including offsets;
there is no reassociation. Intermediates are chunk arrays, not packed fields.
"""
from numbers import Real
import operator
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks


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
    flat = output.ravel()
    for start in range(0, output.size, 8192):
        stop = min(start+8192, output.size)
        indices = np.array(np.unravel_index(np.arange(start,stop),first.shape)).T
        values = {i: fields[i].samples_at(indices) for i in used_inputs}
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
        flat[start:stop] = values[final_id]
    return AdaptiveSampleBlocks(output,first.origin,first.pitch)


if __name__ == '__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_csg_fused_probe import main
    raise SystemExit(main())
