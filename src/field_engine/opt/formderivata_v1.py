#!/usr/bin/env python3
"""Shape derivatives straight out of the leaf formulas, and what the expression tree cannot say.

A gripper finger is declared here twice from the same three shape parameters
theta = (pad corner radius r, face tilt alpha, face curvature kappa):

  AS CLOSED FORM   `sdf_finger` returns the distance, its gradient, d sdf / d theta, the Hessian and
                   d grad / d theta, all analytic, differentiated through the branch that is active
                   at the query point -- max(face, max(rounded box in xy, z slab)).
  AS A TREE        `finger_trad_mm` builds the same finger from the engine's own expression
                   constructors (halfspace, box, offset, intersect, transform), so the shape the
                   derivatives are taken of is a shape the engine can actually evaluate and compile.

WHY BOTH. A shape derivative is only worth having if the shape it differentiates is the shape that
gets built. Declaring the finger twice and comparing makes that an assertion instead of an intention.

MEASURED, both live in about a second:

  tree against closed form, 400 points, four tilts   max 1.734 723 475 976 807e-17 m
                                                     7 nodes, leaf ops {box, halfspace}
  leaf derivatives against central FD at h = 1e-6    d sdf / d theta   8.235 523e-12
                                                     d grad / d theta  4.466 898e-10
                                                     on 320 pad-branch and 80 face-branch points

The FD comparison skips a point whose active branch flips under the step, because across a branch
flip the derivative is not the thing being measured -- a one-sided difference over a kink is a
difference of two different functions.

WHAT THE TREE CANNOT EXPRESS, stated rather than hidden: kappa. A parabolic face bow
q_x + kappa q_y^2 / 2 has no primitive in the engine's op set, so the tree is EXACT ONLY AT
kappa = 0 and the closed form is the only representation that carries the third parameter. The
nearest expressible face is a cylinder leaf of radius 1/kappa, and `treets_lucka_mot_cylinder`
measures what that substitution costs over the contact patch: 2.499 98e-06 m at kappa = 5 and
1.597 45e-04 m at kappa = 20. At kappa = 20 that is a sixth of a millimetre of surface mismatch,
which is not a rounding error on a contact patch.

THE THIRD PARAMETER IS ALSO NEARLY INERT AT THE GRIP THAT WAS MEASURED. Of the three parameters
built, two are active: d/dr is identically zero at that grip (the pad corner never touches), and
kappa barely moves the margin. Three parameters, two of which do anything.

WHAT IS NOT IN THIS REPOSITORY. Composing these leaf derivatives with a contact solver's adjoint
gives d(hold margin)/d theta, which was measured against central FD over three step lengths at
3.216 933e-11 relative, with the optimiser gaining 13.058 6 % of margin in 25 steps and stopping at
a kink where the tangential impulse goes to zero. That composition runs through an exact-cone
contact solver, which lives in the motion engine beside this repository; the numbers travel here as
data (`adjointreferens`) and are NOT re-derived by this module. What this module gates is its own
half: the leaves and the tree.

Also recorded there, because it is the reason the exact solver is worth its cost: a penalty-contact
baseline produces a gradient that is parallel along the path but 2.13x too short, turns antiparallel
(179.81 degrees) at its own optimum, and the shape it converges to scores -3.38 % under the exact
model while the penalty model reports +13.33 % for it. A gradient that points the right way and has
the wrong length is a different failure from one that points the wrong way; this baseline does both,
in different places.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD_ENGINE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FIELD_ENGINE, "ikarus_v1"))
import expr as IKE                                                              # noqa: E402

DATA = os.path.join(FIELD_ENGINE, "..", "..", "data", "formderivata_v1")

# Pad geometry, metres. The face plane sits at XF0 and touches the held box's face.
L_PAD, H_PAD, W_PAD = 0.06, 0.05, 0.06
XF0 = -0.05
MM = 1000.0
STOR = 1.0e4                       # mm; far beyond any query point, makes a box axis-free


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _drz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])


def fingerram(p, theta, sida):
    """World point to finger-local coordinates. sida=+1 left finger, -1 the mirrored right one."""
    _, alpha, _ = (float(v) for v in theta)
    p = np.asarray(p, dtype=np.float64)
    pm = p.copy()
    if sida < 0:
        pm[0] = -pm[0]
    C = np.array([XF0, 0.0, 0.0])
    return _rz(-alpha) @ (pm - C), pm, C


def sdf_vanster(p, theta):
    """(sdf, grad, d sdf/d theta, Hessian, d grad/d theta, active branch) for the left finger.

    Sign convention is the engine's: negative inside. The shape is
    max(face, max(rounded box in xy, z slab)), and every derivative below is taken through the
    branch that is active AT THIS POINT -- there is no smoothing, so at a branch boundary the
    derivative is one-sided by construction and the FD gate skips those points."""
    r, alpha, kappa = (float(v) for v in theta)
    q, pm, C = fingerram(p, theta, +1)
    R, dR, Rinv = _rz(alpha), _drz(alpha), _rz(-alpha)

    # --- face branch: a plane bowed by kappa ------------------------------------------------------
    df = q[0] + 0.5 * kappa * q[1] ** 2
    vf = np.array([1.0, kappa * q[1], 0.0])
    gf = R @ vf
    Hf = R @ np.diag([0.0, kappa, 0.0]) @ Rinv
    dq_da = -_drz(-alpha) @ (pm - C)
    dv_da = np.array([0.0, kappa * dq_da[1], 0.0])
    s_face = np.array([0.0, float(vf @ dq_da), 0.5 * q[1] ** 2])
    g_face = np.stack([np.zeros(3), dR @ vf + R @ dv_da,
                       R @ np.array([0.0, q[1], 0.0])], axis=1)

    # --- pad branch: rounded box in xy, intersected with a z slab ---------------------------------
    b = np.array([L_PAD / 2.0, H_PAD / 2.0])
    c0 = np.array([-L_PAD / 2.0, 0.0])
    dxy = np.abs(q[:2] - c0) - b
    ax, ay = max(dxy[0], 0.0), max(dxy[1], 0.0)
    ute = math.hypot(ax, ay)
    inne = min(max(dxy[0], dxy[1]), 0.0)
    dp2 = ute + inne - r
    dz = abs(q[2]) - W_PAD / 2.0
    g2 = np.zeros(2)
    if ute > 0:
        if ax > 0:
            g2[0] = ax / ute * math.copysign(1.0, q[0] - c0[0])
        if ay > 0:
            g2[1] = ay / ute * math.copysign(1.0, q[1] - c0[1])
    else:
        k = int(np.argmax(dxy))
        g2[k] = math.copysign(1.0, q[k] - c0[k])
    gpad_q = np.array([g2[0], g2[1], 0.0])
    if dz > dp2:                               # the z slab carries no shape parameter at all
        dpad, gpad_q = dz, np.array([0.0, 0.0, math.copysign(1.0, q[2])])
        s_pad, Hpad_q = np.zeros(3), np.zeros((3, 3))
    else:
        dpad = dp2
        Hpad_q = np.zeros((3, 3))
        if ute > 0:
            akt = np.diag([1.0 if ax > 0 else 0.0, 1.0 if ay > 0 else 0.0])
            Hpad_q[:2, :2] = (akt - np.outer(g2, g2)) / ute
        s_pad = np.array([-1.0, float(gpad_q @ dq_da), 0.0])
    Hpad = R @ Hpad_q @ Rinv
    g_pad = np.stack([np.zeros(3), dR @ gpad_q + R @ (Hpad_q @ dq_da), np.zeros(3)], axis=1)
    gpad = R @ gpad_q

    if df >= dpad:
        return df, gf, s_face, Hf, g_face, "face"
    return dpad, gpad, s_pad, Hpad, g_pad, "pad"


def sdf_finger(p, theta, sida=+1):
    """Either finger; the pair shares one theta, so the gripper stays symmetric by construction."""
    p = np.asarray(p, dtype=np.float64)
    if sida > 0:
        return sdf_vanster(p, theta)
    pm = p.copy()
    pm[0] = -pm[0]
    sdf, g, s_t, H, g_t, gren = sdf_vanster(pm, theta)
    g = g.copy()
    g[0] = -g[0]
    M = np.diag([-1.0, 1.0, 1.0])
    g_t = g_t.copy()
    g_t[0, :] = -g_t[0, :]
    return sdf, g, s_t, M @ H @ M, g_t, gren


def finger_trad_mm(theta):
    """The same finger as an expression tree, in millimetres.

    A distance field scales exactly, sdf_mm(1000 p) = 1000 sdf_m(p), so the tree may live in the
    millimetre units the rest of the CAD chain uses without any loss. EXACT ONLY AT kappa = 0."""
    r, alpha, _ = (float(v) for v in theta)
    C = [XF0 * MM, 0.0, 0.0]
    n = [math.cos(alpha), math.sin(alpha), 0.0]
    face = IKE.halfspace(n, n[0] * C[0] + n[1] * C[1] + n[2] * C[2])
    box2d = IKE.box((L_PAD * MM / 2.0, H_PAD * MM / 2.0, STOR),
                    center=(-L_PAD * MM / 2.0, 0.0, 0.0))
    pad_lokal = IKE.intersect(IKE.offset(box2d, r * MM),
                              IKE.box((STOR, STOR, W_PAD * MM / 2.0)))
    M = IKE.mat4_mul(IKE.mat4_translate(*C), IKE.mat4_rot_axis("z", math.degrees(alpha)))
    return IKE.intersect(face, IKE.transform(pad_lokal, M))


def evaluera_trad_m(trad, p_m):
    p = tuple(float(v) * MM for v in p_m)
    return IKE.eval_sdf_py(trad, p) / MM


def treets_lucka_mot_cylinder(kappa, y_halv, n=201):
    """What the nearest EXPRESSIBLE face costs, in metres.

    The parabolic face has no primitive; a cylinder of radius 1/kappa tangent at the face centre is
    the closest thing the op set can say. This is the maximum surface mismatch between the two over
    the contact patch, evaluated on the parabola's own zero set."""
    if kappa == 0.0:
        return 0.0
    R = 1.0 / kappa
    y = np.linspace(-y_halv, y_halv, n)
    qx = -0.5 * kappa * y ** 2
    return float(np.max(np.abs(np.sqrt((qx + R) ** 2 + y ** 2) - abs(R))))


def tradparitet(theta_vinklar=(0.0, 0.02, 0.05, -0.1), n_punkter=400, fro=0):
    """Tree against closed form at kappa = 0, over points around the pad."""
    rng = np.random.default_rng(fro)
    pts = np.column_stack([rng.uniform(-0.09, -0.045, n_punkter),
                           rng.uniform(-0.03, 0.03, n_punkter),
                           rng.uniform(-0.04, 0.04, n_punkter)])
    rader = []
    for alpha in theta_vinklar:
        th = np.array([0.004, alpha, 0.0])
        tr = finger_trad_mm(th)
        d = max(abs(evaluera_trad_m(tr, p) - sdf_vanster(p, th)[0]) for p in pts)
        rader.append({"alpha": alpha, "max_abs_diff_m": float(d)})
    tr0 = finger_trad_mm(np.array([0.004, 0.02, 0.0]))
    return {"rader": rader, "n_punkter": n_punkter,
            "max_abs_diff_m": max(r["max_abs_diff_m"] for r in rader),
            "nodantal": IKE.node_count(tr0), "lovoperationer": sorted(IKE.leaf_ops(tr0)),
            "kappa_ej_uttryckbar_m": {str(k): treets_lucka_mot_cylinder(k, 0.02)
                                      for k in (5.0, 20.0)}}


def lovderivator_mot_fd(theta=(0.004, 0.02, 5.0), steg=(1e-6, 1e-7, 1e-8), n_punkter=400, fro=1):
    """Central finite differences on d sdf/d theta and d grad/d theta, per active branch.

    Points whose branch flips under the step are skipped and counted, not silently averaged in."""
    th = np.asarray(theta, dtype=np.float64)
    rng = np.random.default_rng(fro)
    pts = [np.array([rng.uniform(-0.12, -0.02), rng.uniform(-0.06, 0.06),
                     rng.uniform(-0.06, 0.06)]) for _ in range(n_punkter)]
    ut = {}
    for h in steg:
        per = {}
        for p in pts:
            _, _, st, _, gt, gren = sdf_vanster(p, th)
            es = eg = 0.0
            for j in range(3):
                tp, tm = th.copy(), th.copy()
                tp[j] += h
                tm[j] -= h
                ap, am = sdf_vanster(p, tp), sdf_vanster(p, tm)
                if ap[5] != gren or am[5] != gren:
                    continue
                es = max(es, abs((ap[0] - am[0]) / (2 * h) - st[j]))
                eg = max(eg, float(np.max(np.abs((ap[1] - am[1]) / (2 * h) - gt[:, j]))))
            d = per.setdefault(gren, {"n": 0, "ds": 0.0, "dg": 0.0})
            d["n"] += 1
            d["ds"] = max(d["ds"], es)
            d["dg"] = max(d["dg"], eg)
        ut[str(h)] = per
    return {"theta": th.tolist(), "per_steg": ut}


def adjointreferens(sokvag=None):
    """The contact-composed numbers, carried as data. See the module docstring for why."""
    with open(os.path.join(sokvag or DATA, "formderivata.json")) as fh:
        return json.load(fh)["adjoint"]


def _selftest():
    t = tradparitet()
    print(f"tree vs closed form: {t['max_abs_diff_m']:.6e} m over {t['n_punkter']} points, "
          f"{t['nodantal']} nodes, leaves {t['lovoperationer']}")
    for k, v in sorted(t["kappa_ej_uttryckbar_m"].items()):
        print(f"  kappa {k}: nearest expressible face is off by {v:.6e} m")
    l = lovderivator_mot_fd()
    per = l["per_steg"]["1e-06"]
    for gren in sorted(per):
        d = per[gren]
        print(f"leaf derivatives, {gren} branch ({d['n']} points): "
              f"d sdf/d theta {d['ds']:.6e}, d grad/d theta {d['dg']:.6e}")
    a = adjointreferens()
    print(f"carried: contact-composed d margin/d theta vs FD {a['max_relerr']:.6e} "
          f"over steps {[r['h'] for r in a['rader']]}")

    fel = []
    if t["max_abs_diff_m"] > 1e-16:
        fel.append(f"tree and closed form disagree by {t['max_abs_diff_m']:.3e} m")
    if t["nodantal"] != 7:
        fel.append(f"the finger tree is {t['nodantal']} nodes, not 7")
    if per["pad"]["ds"] > 1e-10 or per["pad"]["dg"] > 1e-8:
        fel.append("pad-branch leaf derivatives do not match central FD")
    if per["face"]["ds"] > 1e-10 or per["face"]["dg"] > 1e-8:
        fel.append("face-branch leaf derivatives do not match central FD")
    if t["kappa_ej_uttryckbar_m"]["20.0"] < 1e-5:
        fel.append("the kappa gap must be reported, not rounded away")
    if fel:
        raise SystemExit("FAIL: " + "; ".join(fel))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
