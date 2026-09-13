"""Reuse sparse column ordering while recomputing every numerical factorization.

Create one SparsePatternSolve per repeated float64 solve sequence and call it
with the same arguments as lastfalt_v1_fem.solve. A changed compressed sparsity
pattern rebuilds the ordering. Matrix values and right-hand sides are never
cached. Other dtypes retain the original solver path.
"""
import numpy as np
from scipy.sparse import linalg
from lastfalt_v1_fem import solve as reference_solve


class SparsePatternSolve:
    def __init__(self):
        self._shape = None
        self._indices = None
        self._indptr = None
        self._order = None
        self.ordering_builds = 0
        self.ordering_reuses = 0

    def __call__(self, K, f, fixed_dofs):
        if K.dtype != np.dtype('float64') or np.asarray(f).dtype != np.dtype('float64'):
            return reference_solve(K, f, fixed_dofs)
        n = K.shape[0]
        free = np.ones(n, dtype=bool)
        free[fixed_dofs] = False
        free_idx = np.where(free)[0]
        A = K[free_idx][:, free_idx].tocsc()
        A.sum_duplicates()
        ff = f[free_idx]
        same = (A.shape == self._shape and np.array_equal(A.indices, self._indices)
                and np.array_equal(A.indptr, self._indptr))
        if same:
            values = linalg.spsolve(A[:, self._order], ff, permc_spec='NATURAL')
            uf = np.empty_like(values)
            uf[self._order] = values
            self.ordering_reuses += 1
        else:
            try:
                factor = linalg.splu(A)
            except RuntimeError:
                return reference_solve(K, f, fixed_dofs)
            uf = factor.solve(ff)
            self._shape = A.shape
            self._indices = A.indices.copy()
            self._indptr = A.indptr.copy()
            self._order = np.argsort(factor.perm_c)
            self.ordering_builds += 1
        u = np.zeros(n)
        u[free_idx] = uf
        return u
