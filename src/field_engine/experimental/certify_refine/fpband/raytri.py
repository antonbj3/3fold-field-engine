"""Watertight ray/triangle (Woop, Benthin, Wald 2013, JCGT 2(1)) in binary32,
local frame, with a rigorous a-posteriori error band.  See RESULTS.md, lemmas R0-R9.

Inputs are world-space float64: v0, v1, v2, org, dir, frame centre cf (N,3).
Reduction: x' = f32(f64(x - cf)); dir' = f32(dir).
Decision codes: 1 = HIT, 0 = MISS, -1 = UNCERTAIN.  Ray interval t in [0, +inf).
"""
import numpy as np
from .f32ops import F32, F64, U, ETA, U64, INFL, a64, fma32

HIT, MISS, UNC = 1, 0, -1


def reduce_pt(x, cf, red=True, eta=ETA):
    w = x - cf                                   # float64, |err| <= U64 |w|
    xr = w.astype(F32)                           # |err| <= U|xr| + ETA
    E = (U * a64(xr) + eta + U64 * np.abs(w)) * INFL if red else np.zeros_like(w)
    return xr, E


def _take(a, k):
    return np.take_along_axis(a, k[:, None], axis=1)[:, 0]


def raytri(v0, v1, v2, org, dirn, cf, fma, with_band=True, red=True, eta=ETA):
    """red=False / eta=0 are ABLATIONS (deliberately unsound) used as counterexamples."""
    with np.errstate(all="ignore"):
        return _raytri(v0, v1, v2, org, dirn, cf, fma, with_band, red, eta)


def _raytri(v0, v1, v2, org, dirn, cf, fma, with_band, red, eta):
    P0, E0 = reduce_pt(v0, cf, red, eta)
    P1, E1 = reduce_pt(v1, cf, red, eta)
    P2, E2 = reduce_pt(v2, cf, red, eta)
    O, EO = reduce_pt(org, cf, red, eta)
    D = dirn.astype(F32)
    ED = (U * a64(D) + eta) * INFL if red else np.zeros(D.shape)
    # R1: translate to ray origin
    verts = []
    for P, EP in ((P0, E0), (P1, E1), (P2, E2)):
        Q = (P - O).astype(F32)
        EQ = (U * a64(Q) + EP + EO) * INFL
        verts.append((Q, EQ))
    # R2: permutation (from rounded direction)
    kz = np.argmax(np.abs(D), axis=1)
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3
    Dz0 = _take(D, kz)
    sw = Dz0 < 0
    kx, ky = np.where(sw, ky, kx), np.where(sw, kx, ky)
    Dx, Dy, Dz = _take(D, kx), _take(D, ky), _take(D, kz)
    EDx, EDy, EDz = _take(ED, kx), _take(ED, ky), _take(ED, kz)
    # R3: shear constants
    Sx = (Dx / Dz).astype(F32)
    Sy = (Dy / Dz).astype(F32)
    Sz = (F32(1.0) / Dz).astype(F32)
    aDz = a64(Dz)
    den = aDz - EDz
    okS = den > 0
    dens = np.where(okS, den, 1.0)
    ESx = np.where(okS, (U * a64(Sx) + eta + (EDz + EDx) / dens) * INFL, np.inf)
    ESy = np.where(okS, (U * a64(Sy) + eta + (EDz + EDy) / dens) * INFL, np.inf)
    ESz = np.where(okS, (U * a64(Sz) + eta + EDz / (aDz * dens)) * INFL, np.inf)
    # R4: shear + scale vertices
    sh = []
    for Q, EQ in verts:
        Qx, Qy, Qz = _take(Q, kx), _take(Q, ky), _take(Q, kz)
        EQx, EQy, EQz = _take(EQ, kx), _take(EQ, ky), _take(EQ, kz)
        out = []
        for S, ES, Qc, EQc in ((Sx, ESx, Qx, EQx), (Sy, ESy, Qy, EQy)):
            if fma:
                R = fma32(-S, Qz, Qc)
                rnd = U * a64(R) + eta
            else:
                m = (S * Qz).astype(F32)
                R = (Qc - m).astype(F32)
                rnd = U * a64(R) + U * a64(m) + eta
            E = (rnd + EQc + a64(S) * EQz + ES * (a64(Qz) + EQz)) * INFL
            out.append((R, E))
        Rz = (Sz * Qz).astype(F32)
        ERz = (U * a64(Rz) + eta + a64(Sz) * EQz + ESz * (a64(Qz) + EQz)) * INFL
        out.append((Rz, ERz))
        sh.append(out)
    (Ax, EAx), (Ay, EAy), (Az, EAz) = sh[0]
    (Bx, EBx), (By, EBy), (Bz, EBz) = sh[1]
    (Cx, ECx), (Cy, ECy), (Cz, ECz) = sh[2]

    # R5: scaled barycentrics  a*b - c*d
    def dop(a, Ea, b, Eb, c, Ec, d, Ed):
        p2 = (c * d).astype(F32)
        if fma:
            r = fma32(a, b, (-p2).astype(F32))
            rnd = U * a64(r) + U * a64(p2) + 2 * eta
        else:
            p1 = (a * b).astype(F32)
            r = (p1 - p2).astype(F32)
            rnd = U * a64(r) + U * a64(p1) + U * a64(p2) + 2 * eta
        E = (rnd + a64(a) * Eb + Ea * (a64(b) + Eb) + a64(c) * Ed + Ec * (a64(d) + Ed)) * INFL
        return r, E
    Uu, EU = dop(Cx, ECx, By, EBy, Cy, ECy, Bx, EBx)
    Vv, EV = dop(Ax, EAx, Cy, ECy, Ay, EAy, Cx, ECx)
    Ww, EW = dop(Bx, EBx, Ay, EAy, By, EBy, Ax, EAx)
    # R6: determinant
    s1 = (Uu + Vv).astype(F32)
    det = (s1 + Ww).astype(F32)
    Edet = (U * a64(s1) + U * a64(det) + EU + EV + EW) * INFL
    # R7: scaled hit distance
    q1 = (Uu * Az).astype(F32)
    if fma:
        r1 = fma32(Vv, Bz, q1)
        T = fma32(Ww, Cz, r1)
        rnd = U * (a64(q1) + a64(r1) + a64(T)) + 3 * eta
    else:
        q2 = (Vv * Bz).astype(F32)
        q3 = (Ww * Cz).astype(F32)
        r1 = (q1 + q2).astype(F32)
        T = (r1 + q3).astype(F32)
        rnd = U * (a64(q1) + a64(q2) + a64(q3) + a64(r1) + a64(T)) + 3 * eta
    ET = (rnd + a64(Uu) * EAz + EU * (a64(Az) + EAz) + a64(Vv) * EBz + EV * (a64(Bz) + EBz)
          + a64(Ww) * ECz + EW * (a64(Cz) + ECz)) * INFL
    # R8: t = T / det
    t = (T / det).astype(F32)
    den2 = a64(det) - Edet
    okt = den2 > 0
    Et = np.where(okt, (U * a64(t) + eta + (ET + (a64(t) * (1 + U) + eta) * Edet)
                        / np.where(okt, den2, 1.0)) * INFL, np.inf)

    # plain binary32 decision (no band, no double fallback)
    neg = (Uu < 0) | (Vv < 0) | (Ww < 0)
    pos = (Uu > 0) | (Vv > 0) | (Ww > 0)
    plain = np.where((neg & pos) | (det == 0) | ~(t >= 0) | ~np.isfinite(t), MISS, HIT)
    res = dict(plain=plain, t=t, kx=kx, ky=ky, kz=kz, W=Ww)
    if not with_band:
        return res

    def fin(*xs):
        m = np.ones(len(t), bool)
        for x in xs:
            m &= np.isfinite(x)
        return m
    finUVW = okS & fin(Uu, Vv, Ww, EU, EV, EW)
    finAll = finUVW & fin(det, Edet, T, ET, t, Et) & okt
    Uh, Vh, Wh = Uu.astype(F64), Vv.astype(F64), Ww.astype(F64)
    pU, pV, pW = Uh > EU, Vh > EV, Wh > EW
    nU, nV, nW = Uh < -EU, Vh < -EV, Wh < -EW
    geU, geV, geW = Uh >= EU, Vh >= EV, Wh >= EW
    leU, leV, leW = Uh <= -EU, Vh <= -EV, Wh <= -EW
    anyP, anyN = pU | pV | pW, nU | nV | nW
    inside = ((geU & geV & geW) & anyP) | ((leU & leV & leW) & anyN)
    outside = anyP & anyN
    zero3 = (EU == 0) & (EV == 0) & (EW == 0) & (Uh == 0) & (Vh == 0) & (Wh == 0)
    th = t.astype(F64)
    tpos, tneg = th >= Et, th < -Et
    dec = np.full(len(t), UNC, np.int8)
    dec[finUVW & (outside | zero3)] = MISS
    dec[finAll & tneg] = MISS
    dec[finAll & inside & tpos & ~outside] = HIT
    res.update(dec=dec, Et=Et,
               q=dict(U=(Uu, EU), V=(Vv, EV), W=(Ww, EW), det=(det, Edet), T=(T, ET), t=(t, Et),
                      Ax=(Ax, EAx), Ay=(Ay, EAy), Az=(Az, EAz), Bx=(Bx, EBx), By=(By, EBy),
                      Bz=(Bz, EBz), Cx=(Cx, ECx), Cy=(Cy, ECy), Cz=(Cz, ECz),
                      Sx=(Sx, ESx), Sy=(Sy, ESy), Sz=(Sz, ESz),
                      A0=(verts[0][0], verts[0][1]), A1=(verts[1][0], verts[1][1]),
                      A2=(verts[2][0], verts[2][1])),
               valid=dict(uvw=finUVW, all=finAll))
    return res


def raytri_plain(v0, v1, v2, org, dirn, cf, fma):
    """Same binary32 operation sequence as raytri() without any bound arithmetic (cost baseline)."""
    with np.errstate(all="ignore"):
        P = [(x - cf).astype(F32) for x in (v0, v1, v2)]
        O = (org - cf).astype(F32)
        D = dirn.astype(F32)
        kz = np.argmax(np.abs(D), axis=1)
        kx = (kz + 1) % 3
        ky = (kx + 1) % 3
        sw = _take(D, kz) < 0
        kx, ky = np.where(sw, ky, kx), np.where(sw, kx, ky)
        Dx, Dy, Dz = _take(D, kx), _take(D, ky), _take(D, kz)
        Sx, Sy, Sz = Dx / Dz, Dy / Dz, F32(1.0) / Dz
        sh = []
        for Pi in P:
            Q = Pi - O
            Qx, Qy, Qz = _take(Q, kx), _take(Q, ky), _take(Q, kz)
            if fma:
                sh.append((fma32(-Sx, Qz, Qx), fma32(-Sy, Qz, Qy), Sz * Qz))
            else:
                sh.append((Qx - Sx * Qz, Qy - Sy * Qz, Sz * Qz))
        (Ax, Ay, Az), (Bx, By, Bz), (Cx, Cy, Cz) = sh
        if fma:
            Uu, Vv, Ww = (fma32(Cx, By, -(Cy * Bx)), fma32(Ax, Cy, -(Ay * Cx)), fma32(Bx, Ay, -(By * Ax)))
            T = fma32(Ww, Cz, fma32(Vv, Bz, Uu * Az))
        else:
            Uu, Vv, Ww = Cx * By - Cy * Bx, Ax * Cy - Ay * Cx, Bx * Ay - By * Ax
            T = Uu * Az + Vv * Bz + Ww * Cz
        det = Uu + Vv + Ww
        t = T / det
        neg = (Uu < 0) | (Vv < 0) | (Ww < 0)
        pos = (Uu > 0) | (Vv > 0) | (Ww > 0)
        plain = np.where((neg & pos) | (det == 0) | ~(t >= 0) | ~np.isfinite(t), MISS, HIT)
        return dict(plain=plain, t=t)
