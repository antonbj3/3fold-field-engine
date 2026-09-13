"""Ordered native CPU objective and central evaluations for duct radius descent.

Compile radius_objective_native_v1/objective.cpp with -O3 -fno-fast-math
-ffp-contract=off -fno-builtin-pow -shared -fPIC. NumPy scalar powers require
the libm pow call even for exponent two; replacing it with multiplication can differ. The context owns fixed objective parameters;
positive finite float64 radii and at most 64 controls/corners are supported.
"""
import ctypes
import math
from pathlib import Path
import numpy as np


class PreparedRadiusObjective:
    def __init__(self, geometry_terms, sdf_values, k_bend, flow, rho, mu,
                 k_area, k_intrude, library):
        lengths, indices, area_min, area_den, intrusion_den = geometry_terms
        self.lengths = np.array(lengths, dtype=np.float64, order='C', copy=True)
        raw_indices = np.asarray(indices)
        if (self.lengths.ndim != 1 or raw_indices.ndim != 1 or
            (raw_indices.size and (raw_indices.dtype.kind not in 'iu' or
             np.any(raw_indices < 0) or np.any(raw_indices >= 64)))):
            raise ValueError('One-dimensional lengths and bounded integer corner indices required')
        self.indices = np.array(raw_indices, dtype=np.int32, order='C', copy=True)
        self.n = len(self.lengths) + 1
        self.sdf = (np.zeros(self.n) if sdf_values is None else
                    np.array(sdf_values, dtype=np.float64, order='C', copy=True))
        self.obstacle = int(sdf_values is not None)
        self.params = np.array([k_bend, flow, rho, mu, k_area, k_intrude,
                                area_min, area_den, intrusion_den, math.pi], dtype=np.float64)
        if (self.lengths.shape != (self.n-1,) or self.indices.ndim != 1 or
            not 2 <= self.n <= 64 or len(self.indices) > 64 or self.sdf.shape != (self.n,) or
            np.any(self.indices < 0) or np.any(self.indices >= self.n) or
            any(not np.isfinite(a).all() for a in (self.lengths, self.sdf, self.params)) or
            np.any(self.lengths < 0) or min(flow, rho, mu, area_den, intrusion_den) <= 0):
            raise ValueError('Finite bounded radius objective geometry and physical parameters required')
        for a in (self.lengths, self.indices, self.sdf, self.params):
            a.flags.writeable = False
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        pointer = ctypes.c_void_p
        arguments = [ctypes.c_int]*3 + [pointer]*6 + [ctypes.c_int64]
        self.library.field_radius_evaluate.argtypes = arguments
        self.library.field_radius_evaluate.restype = ctypes.c_int
        self.library.field_radius_central.argtypes = arguments + [ctypes.c_double]
        self.library.field_radius_central.restype = ctypes.c_int
        self._fixed = (self.n, len(self.indices), self.obstacle, self.lengths.ctypes.data,
                       self.indices.ctypes.data, self.sdf.ctypes.data, self.params.ctypes.data)

    def _radii(self, radii):
        if (not isinstance(radii, np.ndarray) or radii.dtype != np.float64 or
            radii.shape != (self.n,) or not radii.flags.c_contiguous or
            not np.isfinite(radii).all() or np.any(radii <= 0)):
            raise ValueError('Positive finite contiguous float64 radii matching the context required')
        return radii

    def evaluate(self, radii):
        radii = self._radii(radii)
        output = np.empty(8)
        code = self.library.field_radius_evaluate(*self._fixed, radii.ctypes.data,
                                                 output.ctypes.data, output.size)
        if code:
            raise ValueError('Native radius objective refused inputs')
        return output[0], dict(dp_friction_pa=output[1], dp_bend_pa=output[2],
            pen_area=output[3], pen_intrude=output[4], dp_total_unconstrained_pa=output[5],
            n_bends=int(output[6]), max_viol_intrude_mm=float(output[7]))

    def central_evaluations(self, radii, h=0.5):
        radii = self._radii(radii)
        if not np.isfinite(h) or h <= 0:
            raise ValueError('Positive finite central difference step required')
        output = np.empty((2*self.n, 8))
        code = self.library.field_radius_central(*self._fixed, radii.ctypes.data,
                                               output.ctypes.data, output.size, h)
        if code:
            raise ValueError('Native central evaluations refused inputs')
        return output
