"""Exact mass properties of a closed triangle mesh and exact dM/du for femur_edit.osteotomy.

Density 1. Coordinates in mm. Mass properties via the divergence theorem as exact
polynomials in the vertex coordinates. dM/du by the chain rule
sum_v dM/dV_v . dV_v/du with analytic dV/du from src/femur/femur_edit.py.
No finite differences on the main derivative path.
"""
import numpy as np

I3 = np.eye(3)
_IDX_IJ = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _skew(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def _rot(axis, ang):
    a = np.asarray(axis, float)
    K = _skew(a)
    c, s = np.cos(ang), np.sin(ang)
    aa = np.outer(a, a)
    return c[..., None, None] * I3 + s[..., None, None] * K + (1 - c)[..., None, None] * aa


def _drot(axis, ang):
    a = np.asarray(axis, float)
    K = _skew(a)
    c, s = np.cos(ang), np.sin(ang)
    aa = np.outer(a, a)
    return -s[..., None, None] * I3 + c[..., None, None] * K + s[..., None, None] * aa


def _ddrot(axis, ang):
    a = np.asarray(axis, float)
    K = _skew(a)
    c, s = np.cos(ang), np.sin(ang)
    aa = np.outer(a, a)
    return -c[..., None, None] * I3 - s[..., None, None] * K + c[..., None, None] * aa


def _tri(V, F):
    A = V[F[:, 0]]
    B = V[F[:, 1]]
    C = V[F[:, 2]]
    BC = np.cross(B, C)
    CA = np.cross(C, A)
    AB = np.cross(A, B)
    d = np.einsum('ij,ij->i', A, BC)
    s = A + B + C
    Q = (A[:, :, None] * A[:, None, :] + B[:, :, None] * B[:, None, :]
         + C[:, :, None] * C[:, None, :] + s[:, :, None] * s[:, None, :])
    return A, B, C, BC, CA, AB, d, s, Q


def moments(V, F):
    A, B, C, BC, CA, AB, d, s, Q = _tri(V, F)
    Vol = d.sum() / 6.0
    P = (d[:, None] * s).sum(0) / 24.0
    S = (d[:, None, None] * Q).sum(0) / 120.0
    return Vol, P, S


def mass_properties(V, F):
    Vol, P, S = moments(V, F)
    c = P / Vol
    I_O = np.trace(S) * I3 - S
    I_c = I_O - Vol * ((c @ c) * I3 - np.outer(c, c))
    return Vol, c, I_c


def pack(Vol, c, I):
    v = np.empty(10, dtype=np.result_type(Vol, c, I))
    v[0] = Vol
    v[1:4] = c
    v[4] = I[0, 0]
    v[5] = I[0, 1]
    v[6] = I[0, 2]
    v[7] = I[1, 1]
    v[8] = I[1, 2]
    v[9] = I[2, 2]
    return v


def mass_vector(V, F):
    Vol, c, I = mass_properties(V, F)
    return pack(Vol, c, I)


# ---------------------------------------------------------------- dV/du

def dV_du(V, frame, u):
    """Analytic dV/du (n,3,3): [vertex, coord, u]. Nonzero on active vertices only."""
    n = len(V)
    dV = np.zeros((n, 3, 3), dtype=np.result_type(V, u))
    w = frame['w']
    act = np.where(w > 0)[0]
    if len(act) == 0:
        return dV
    th, ph, s = u
    a = np.asarray(frame['shaft'], float)
    b = np.asarray(frame['a_v'], float)
    X = V[act] - frame['c']
    wa = w[act]
    Ra = _rot(a, wa * th)
    Rb = _rot(b, wa * ph)
    dRa = wa[:, None, None] * _drot(a, wa * th)
    dRb = wa[:, None, None] * _drot(b, wa * ph)
    R1 = np.einsum('mij,mj->mi', Ra, X)
    dR1 = np.einsum('mij,mj->mi', dRa, X)
    dV[act, :, 0] = np.einsum('mij,mj->mi', Rb, dR1)
    dV[act, :, 1] = np.einsum('mij,mj->mi', dRb, R1)
    dV[act, :, 2] = wa[:, None] * a[None, :]
    return dV


def V_of_u(V, frame, u):
    """Complex-safe re-implementation of femur_edit.osteotomy (same arithmetic)."""
    th, ph, s = u
    w = frame['w']
    act = np.where(w > 0)[0]
    Vn = np.array(V, dtype=np.result_type(V, u))
    if len(act) == 0 or (th == 0 and ph == 0 and s == 0):
        return Vn
    a = np.asarray(frame['shaft'], float)
    b = np.asarray(frame['a_v'], float)
    X = V[act] - frame['c']
    wa = w[act]
    Ra = _rot(a, wa * th)
    Rb = _rot(b, wa * ph)
    R1 = np.einsum('mij,mj->mi', Ra, X)
    R2 = np.einsum('mij,mj->mi', Rb, R1)
    Vn[act] = V[act] + (wa * s)[:, None] * a[None, :] + (R2 - X)
    return Vn


def dV_du2(V, frame, u):
    """Analytic d2V/du2 (n,3,3,3): [vertex, coord, u_a, u_b]. Active vertices only."""
    n = len(V)
    d2V = np.zeros((n, 3, 3, 3), dtype=np.result_type(V, u))
    w = frame['w']
    act = np.where(w > 0)[0]
    if len(act) == 0:
        return d2V
    th, ph, s = u
    a = np.asarray(frame['shaft'], float)
    b = np.asarray(frame['a_v'], float)
    X = V[act] - frame['c']
    wa = w[act]
    Ra = _rot(a, wa * th)
    Rb = _rot(b, wa * ph)
    dRa = wa[:, None, None] * _drot(a, wa * th)
    dRb = wa[:, None, None] * _drot(b, wa * ph)
    ddRa = (wa ** 2)[:, None, None] * _ddrot(a, wa * th)
    ddRb = (wa ** 2)[:, None, None] * _ddrot(b, wa * ph)
    R1 = np.einsum('mij,mj->mi', Ra, X)
    dR1 = np.einsum('mij,mj->mi', dRa, X)
    # th,th
    d2V[act, :, 0, 0] = np.einsum('mij,mj->mi', Rb, np.einsum('mij,mj->mi', ddRa, X))
    # ph,ph
    d2V[act, :, 1, 1] = np.einsum('mij,mj->mi', ddRb, R1)
    # th,ph
    d2V[act, :, 0, 1] = np.einsum('mij,mj->mi', dRb, dR1)
    d2V[act, :, 1, 0] = d2V[act, :, 0, 1]
    return d2V


# ---------------------------------------------------------------- dM/dV and dM/du

def _final_grads(dvol, dP, dS, Vol, P, c, T):
    """Per-triangle gradients of (Vol,c,I) wrt one vertex slot.

    dvol (T,3); dP (T,3,3); dS (T,3,3,3). Returns dvol (T,3), dc (T,3,3),
    dIc (T,3,3,3) with dc[t,i,k]=d c_i/d coord_k, dIc[t,i,j,k]=d I_ij/d coord_k.
    """
    P_over_Vol2 = P / Vol ** 2
    dc = dP / Vol - P_over_Vol2[None, :, None] * dvol[:, None, :]
    trX = np.einsum('tmmk->tk', dS)
    dIO = I3[None, :, :, None] * trX[:, None, None, :] - dS
    cdc = np.einsum('m,tmk->tk', c, dc)
    dT = (I3[None, :, :, None] * (2 * cdc)[:, None, None, :]
          - (dc[:, :, None, :] * c[None, None, :, None]
             + c[None, :, None, None] * dc[:, None, :, :]))
    dIc = dIO - dvol[:, None, None, :] * T[None, :, :, None] - Vol * dT
    return dvol, dc, dIc


def _tri_slot_grads(V, F):
    """List of (dvol, dc, dIc, vertex_index_array) for the three slots."""
    A, B, C, BC, CA, AB, d, s, Q = _tri(V, F)
    Vol, P, S = moments(V, F)
    c = P / Vol
    T = (c @ c) * I3 - np.outer(c, c)
    out = []
    for cross_, base, idx in ((BC, A, F[:, 0]), (CA, B, F[:, 1]), (AB, C, F[:, 2])):
        dvol = cross_ / 6.0
        dP = (np.einsum('tk,ti->tik', cross_, s) + d[:, None, None] * I3[None]) / 24.0
        qterm = np.einsum('tk,tij->tijk', cross_, Q)
        b_s = base + s
        t1 = np.einsum('ik,tj->tijk', I3, b_s)
        t2 = np.einsum('jk,ti->tijk', I3, b_s)
        dS = (qterm + d[:, None, None, None] * (t1 + t2)) / 120.0
        out.append((*_final_grads(dvol, dP, dS, Vol, P, c, T), idx))
    return out


def dM_dV(V, F):
    """Per-vertex gradient (n,10,3) of M=[Vol,cx,cy,cz,I00,I01,I02,I11,I12,I22]."""
    n = len(V)
    G = np.zeros((n, 10, 3), dtype=np.result_type(V.dtype, np.float64))
    for dvol, dc, dIc, idx in _tri_slot_grads(V, F):
        np.add.at(G[:, 0, :], idx, dvol)
        np.add.at(G[:, 1:4, :], idx, dc)
        for m, (i, j) in enumerate(_IDX_IJ):
            np.add.at(G[:, 4 + m, :], idx, dIc[:, i, j, :])
    return G


def dM_du(V, F, frame, u, dV=None, G=None):
    """Exact chain-rule dM/du (10,3) with M=[Vol,cx,cy,cz,I00,I01,I02,I11,I12,I22].

    dV/du uses the base V (the rotation acts on base coordinates, as in osteotomy);
    dM/dV is evaluated on the deformed mesh V(u) unless a precomputed G at u=0 is
    supplied. Passing G is valid only for u=0.
    """
    if dV is None:
        dV = dV_du(V, frame, u)
    if G is None:
        G = dM_dV(V_of_u(V, frame, u), F)
    return np.einsum('vlk,vka->la', G, dV)
