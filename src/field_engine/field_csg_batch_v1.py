"""Evaluate ordered CSG in one native CPU batch, then pack exactly.

Input ids precede result ids. Instruction i produces id len(fields)+i. Each
operation preserves the eager variant's float32 rounding, including offsets;
there is no reassociation. Intermediates are chunk arrays, not packed fields.
"""
from numbers import Real
import operator
import itertools
import ctypes
from pathlib import Path
import sys
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks, NODE
from adaptive_pack_native_v1 import NativeAdaptiveBlocks


def evaluate(fields, instructions, *, batch_library, pack_library):
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
    if sys.byteorder!='little':
        raise ValueError('Little-endian host required')
    if any(f.nodes.dtype!=NODE or f.nodes.ndim!=1 or not f.nodes.flags.c_contiguous or f.payload.dtype!=np.float32 or f.payload.ndim!=1 or not f.payload.flags.c_contiguous for f in fields):
        raise ValueError('Contiguous native field layout required')
    schema=np.dtype([('op','<i4'),('left','<i4'),('right','<i4'),('pad','<i4'),('offset','<f8')])
    encoded=np.zeros(len(program),dtype=schema)
    for i,(op,left,right) in enumerate(program):
        encoded[i]['op']={'union':0,'difference':1,'offset':2}[op]
        encoded[i]['left']=left
        encoded[i]['offset' if op=='offset' else 'right']=right
    pointers=(ctypes.c_void_p*len(fields))(*(f.nodes.ctypes.data for f in fields))
    payloads=(ctypes.c_void_p*len(fields))(*(f.payload.ctypes.data for f in fields))
    counts=np.array([len(f.nodes) for f in fields],dtype=np.int64)
    lengths=np.array([len(f.payload) for f in fields],dtype=np.int64)
    shape=np.array(first.shape,dtype=np.int32)
    output=np.empty(first.shape,dtype=np.float32)
    lib=ctypes.CDLL(str(Path(batch_library).resolve()))
    function=lib.csg_execute
    function.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int32,
                       ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int32,ctypes.c_void_p]
    function.restype=ctypes.c_int
    code=function(pointers,counts.ctypes.data,payloads,lengths.ctypes.data,len(fields),shape.ctypes.data,
                  encoded.ctypes.data,len(encoded),output.ctypes.data)
    if code:raise ValueError(f'Native CSG validation failed: {code}')
    return NativeAdaptiveBlocks(output,first.origin,first.pitch,library=pack_library)


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_csg_batch_probe import main
    raise SystemExit(main())
