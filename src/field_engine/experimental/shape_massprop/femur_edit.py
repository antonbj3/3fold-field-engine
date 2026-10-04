"""Standalone femur surface edit; numpy only. Coordinates and lengths are millimetres.

The registry is a mapping name -> (vertex indices, barycentric weights), or a
geomgr Registry with by_kind('landmark'). No attachment data is needed.
"""
import numpy as np

MEASURE_VERSION = 'ccd_av@p1_smooth-1'
TARGETS = ('CCD', 'AV', 'L_mech')
HEAD6 = ('SFH', 'AFH', 'LFH', 'PFH', 'MFH', 'IFH')
NECK4 = ('SNI', 'ANI', 'INI', 'PNI')
CUT_FRAC = 0.78
BAND_MM = 10.0


def _unit(v):
    """Source: results/N7c/geomgr/core.py:57."""
    return v / np.linalg.norm(v)


def _sphere_fit(P):
    """Source: results/N7c/geomgr/core.py:49."""
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    ctr = c[:3]
    return ctr, float(np.sqrt(c[3] + ctr @ ctr))


def _landmarks(reg, V):
    """Source: results/N7c/geomgr/ops.py:56; registry access adapted."""
    if hasattr(reg, 'by_kind'):
        anchors = {e.name: (e.indices, e.weights) for e in reg.by_kind('landmark')}
    else:
        anchors = reg
    return {name: np.asarray(weights) @ V[np.asarray(indices, dtype=int)]
            for name, (indices, weights) in anchors.items()}


def _measures_core(L, V):
    """Source: results/N7c/geomgr/core.py:61."""
    head_c, _ = _sphere_fit(np.array([L[k] for k in HEAD6]))
    neck_c = np.mean([L[k] for k in NECK4], 0)
    knee_c = 0.5 * (L['LEC'] + L['MEC'])
    mech = _unit(head_c - knee_c)
    proj = (V - knee_c) @ mech
    lo, hi = proj.min(), proj.max()
    cents = []
    for f in np.linspace(0.30, 0.70, 9):
        h = lo + f * (hi - lo)
        sl = V[np.abs(proj - h) < 3.0]
        if len(sl) > 10:
            cents.append(sl.mean(0))
    C = np.array(cents)
    _, _, vt = np.linalg.svd(C - C.mean(0))
    shaft = _unit(vt[0] * np.sign(vt[0] @ mech))
    neck = _unit(head_c - neck_c)
    ccd = float(np.degrees(np.arccos(np.clip(neck @ -shaft, -1, 1))))

    def pp(v):
        return v - (v @ mech) * mech

    med = _unit(pp(L['PMC'] - L['PLC']))
    ant = _unit(pp(knee_c - 0.5 * (L['PMC'] + L['PLC'])))
    ant = _unit(ant - (ant @ med) * med)
    nn = pp(neck)
    av = float(np.degrees(np.arctan2(nn @ ant, nn @ med)))
    return dict(L_mech=float(np.linalg.norm(head_c - knee_c)), CCD=ccd, AV=av,
                _head_c=head_c, _knee_c=knee_c, _mech=mech, _shaft=shaft)


def _p1_smooth(L, V, sigma=1.5):
    """Source: results/N7c/geomgr/measures.py:39."""
    m = _measures_core(L, V)
    knee_c, mech, head_c = m['_knee_c'], m['_mech'], m['_head_c']
    proj = (V - knee_c) @ mech
    lo, hi = proj.min(), proj.max()
    cents = []
    for f in np.linspace(0.30, 0.70, 9):
        h = lo + f * (hi - lo)
        w = np.exp(-0.5 * ((proj - h) / sigma) ** 2)
        cents.append((w[:, None] * V).sum(0) / w.sum())
    Cc = np.array(cents)
    _, _, vt = np.linalg.svd(Cc - Cc.mean(0))
    shaft = _unit(vt[0] * np.sign(vt[0] @ mech))
    neck_c = np.mean([L[k] for k in NECK4], 0)
    neck = _unit(head_c - neck_c)
    m['CCD'] = float(np.degrees(np.arccos(np.clip(neck @ -shaft, -1, 1))))
    m['_shaft'] = shaft
    m['_neck'] = neck
    return m


def _measure(reg, V, F):
    """Source: results/N7c/geomgr/ops.py:56."""
    L = _landmarks(reg, V)
    return _p1_smooth(L, V), L


def _smoothstep(x):
    """Source: results/N7c/geomgr/ops.py:45."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _rodrigues(X, axis, ang):
    """Source: results/N7c/geomgr/ops.py:50."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return X * c + np.cross(a, X) * s + np.outer(X @ a, a) * (1 - c)


def edit_frame(reg, V, F):
    """Source: results/N7c/geomgr/ops.py:62."""
    m, L = _measure(reg, V, F)
    h = (V - m['_knee_c']) @ m['_mech']
    hcut = CUT_FRAC * m['L_mech']
    c = V[np.abs(h - hcut) < 2.0].mean(0)
    neck_c = np.mean([L[k] for k in NECK4], 0)
    neck = _unit(m['_head_c'] - neck_c)
    a_v = _unit(np.cross(neck, m['_shaft']))
    w = _smoothstep((h - (hcut - BAND_MM)) / (2 * BAND_MM))
    return dict(y={k: float(m[k]) for k in TARGETS}, c=c, shaft=m['_shaft'],
                a_v=a_v, w=w, hcut=hcut)


def osteotomy(V, frame, u):
    """Source: results/N7c/geomgr/ops.py:74. u=(twist rad, varus rad, lengthening mm)."""
    th, ph, s = u
    w = frame['w']
    act = w > 0
    Vn = V.copy()
    if th == 0 and ph == 0 and s == 0:
        return Vn
    X = V[act] - frame['c']
    wa = w[act]
    Xr = _rodrigues(_rodrigues(X, frame['shaft'], wa * th), frame['a_v'], wa * ph)
    Vn[act] = V[act] + (wa * s)[:, None] * frame['shaft'] + (Xr - X)
    return Vn


def edit(V, F, reg, target, delta, iters=12, tol=1e-4):
    """Source: results/N7c/geomgr/ops.py:88; returns V1 and numeric report.

    The source's Instance logging and certificate are excluded from this array API.
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    fr = edit_frame(reg, V, F)
    k = TARGETS.index(target)
    y0 = np.array([fr['y'][t] for t in TARGETS])
    tgt = y0.copy()
    tgt[k] += delta

    def ym(u):
        m, _ = _measure(reg, osteotomy(V, fr, u), F)
        return np.array([m[t] for t in TARGETS])

    u = np.zeros(3)
    steps = np.array([np.radians(0.2), np.radians(0.2), 0.2])
    hist = []
    for _ in range(iters):
        err = ym(u) - tgt
        hist.append(float(np.abs(err).max()))
        if np.abs(err).max() < tol:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            du = np.zeros(3)
            du[j] = steps[j]
            J[:, j] = (ym(u + du) - ym(u - du)) / (2 * steps[j])
        u = u - np.linalg.solve(J, err)
    V1 = osteotomy(V, fr, u)
    y1 = ym(u)
    report = dict(target=target, delta=float(delta), measure=MEASURE_VERSION,
                  y0=dict(zip(TARGETS, y0.tolist())), y1=dict(zip(TARGETS, y1.tolist())),
                  target_err=float(y1[k] - tgt[k]),
                  held_maxabs=float(max(abs(y1[j] - y0[j]) for j in range(3) if j != k)),
                  u=dict(twist_deg=float(np.degrees(u[0])), varus_deg=float(np.degrees(u[1])),
                         lengthening_mm=float(u[2])), newton_maxerr=hist,
                  distal_bit_identical=bool(np.array_equal(V1[fr['w'] == 0], V[fr['w'] == 0])),
                  n_moved=int((fr['w'] > 0).sum()))
    return V1, report
