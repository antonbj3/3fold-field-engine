#!/usr/bin/env python3
"""Shared Q1 hex8 linear-elastic core: element stiffness, global assembly and a direct solve.

The element stiffness uses closed Gauss integration; global assembly is vectorised on the host through
a scipy COO matrix, whose duplicate entries are summed by sorting, so the assembly is deterministic and
needs no atomics. A matrix-free Warp K*v operator exists separately (in lastfalt_v1_verify.py) and uses
int64 fixed-point scatter-add rather than float32 atomic_add, which is not deterministic; it is used
for cross-checking against this assembly.

For the grid sizes used here (at most about 25k degrees of freedom) the host sparse solve is faster and
deterministic, so it owns the main loop.

Entry points: hex8_ke(a,b,c,E,nu) -> (24,24) element stiffness; build_grid(nx,ny,nz) ->
(element node ids, index arrays); assemble_K_cpu(...) -> sparse K; solve(K,f,fixed_dofs) -> u;
compliance(f,u) -> scalar.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- Hex8 referens-elementet (naturliga koord xi,eta,zeta i [-1,1]) ---------------------------
_NODE_NAT = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def _shape_derivs(xi, eta, zeta):
    """dN/d(xi,eta,zeta), shape (8,3)."""
    dN = np.zeros((8, 3))
    for i, (xi_i, eta_i, zeta_i) in enumerate(_NODE_NAT):
        dN[i, 0] = 0.125 * xi_i * (1 + eta * eta_i) * (1 + zeta * zeta_i)
        dN[i, 1] = 0.125 * eta_i * (1 + xi * xi_i) * (1 + zeta * zeta_i)
        dN[i, 2] = 0.125 * zeta_i * (1 + xi * xi_i) * (1 + eta * eta_i)
    return dN


def _D_matrix(E, nu):
    """3D isotrop Hooke, Voigt-ordning [xx,yy,zz,yz,xz,xy]."""
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = c * (1 - nu)
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = c * nu
    G = E / (2 * (1 + nu))
    D[3, 3] = D[4, 4] = D[5, 5] = G
    return D


def hex8_ke(a, b, c, E=1.0, nu=0.3):
    """Elementstyvhet (24x24) for en REGULAR hex av dim (a,b,c) [mm], 2x2x2 Gauss, E=1 unit-material
    (SIMP skalar med rho^p * E0 utanfor). a,b,c = full elementlangd i x,y,z."""
    D = _D_matrix(E, nu)
    Ke = np.zeros((24, 24))
    # nodkoord (lokala, centrerat)
    nc = _NODE_NAT * np.array([a / 2, b / 2, c / 2])
    for xi in _GP:
        for eta in _GP:
            for zeta in _GP:
                dN_nat = _shape_derivs(xi, eta, zeta)          # (8,3) d/d(nat)
                J = nc.T @ dN_nat                                # (3,3) = sum_i x_i * dN_i/dnat
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)
                dN_xyz = dN_nat @ Jinv.T                          # (8,3) d/d(x,y,z)
                B = np.zeros((6, 24))
                for i in range(8):
                    dNx, dNy, dNz = dN_xyz[i]
                    B[0, 3 * i + 0] = dNx
                    B[1, 3 * i + 1] = dNy
                    B[2, 3 * i + 2] = dNz
                    B[3, 3 * i + 1] = dNz
                    B[3, 3 * i + 2] = dNy
                    B[4, 3 * i + 0] = dNz
                    B[4, 3 * i + 2] = dNx
                    B[5, 3 * i + 0] = dNy
                    B[5, 3 * i + 1] = dNx
                Ke += (B.T @ D @ B) * abs(detJ)
    return Ke


def node_id(i, j, k, ny, nz):
    return (i * (ny + 1) + j) * (nz + 1) + k


def build_grid(nx, ny, nz):
    """Elementens (i,j,k) hornnod-id:n, ordning matchar _NODE_NAT."""
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ii = ii.ravel(); jj = jj.ravel(); kk = kk.ravel()
    offs = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    elem_nodes = np.zeros((nx * ny * nz, 8), dtype=np.int64)
    for c, (di, dj, dk) in enumerate(offs):
        elem_nodes[:, c] = node_id(ii + di, jj + dj, kk + dk, ny, nz)
    return elem_nodes, (ii, jj, kk)


def assemble_K_cpu(elem_nodes, rho, Ke0, p, E0, n_nodes):
    """Vektoriserad COO-montering. rho: (n_elem,) i (0,1]. Returnerar CSR K (3n x 3n)."""
    n_elem = elem_nodes.shape[0]
    dof = (elem_nodes[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(n_elem, 24)  # (n_elem,24)
    scale = E0 * np.power(rho, p)                      # (n_elem,)
    Ke_all = scale[:, None, None] * Ke0[None, :, :]      # (n_elem,24,24)
    rows = np.repeat(dof[:, :, None], 24, axis=2).reshape(-1)
    cols = np.repeat(dof[:, None, :], 24, axis=1).reshape(-1)
    data = Ke_all.reshape(-1)
    K = sp.coo_matrix((data, (rows, cols)), shape=(3 * n_nodes, 3 * n_nodes)).tocsr()
    return K


def solve(K, f, fixed_dofs):
    n = K.shape[0]
    free = np.ones(n, dtype=bool)
    free[fixed_dofs] = False
    free_idx = np.where(free)[0]
    Kff = K[free_idx][:, free_idx]
    ff = f[free_idx]
    uf = spla.spsolve(Kff.tocsc(), ff)
    u = np.zeros(n)
    u[free_idx] = uf
    return u


def compliance(f, u):
    return float(f @ u)
