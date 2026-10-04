"""Kink-aware surface-form mass-property derivatives for the CSG/SDF leaf shapes.

The leaf field oracle and the six-tetra marching-tetrahedra extraction are taken unchanged from
``src/sdfj/massprop.py``.  The only replacement is the surface quadrature: every extracted triangle
that crosses a branch change of the piecewise-smooth CSG field is split along the branch-change
curve ``f_a - f_b = 0`` (linear solve on the triangle edges), and each piece carries its own
branch velocity ``V_n = -(d sdf_b/d theta)/||grad sdf_b||``.

The module also carries two independent references for the exact mass properties:

* ``finger_exact_properties`` / ``femur_exact_properties``: analytic one-dimensional quadrature
  (no SDF mesh), used as the primary denominator for the derivative errors;
* the fine linear-tetrahedron central differences already in ``massprop.py``, used as the
  prescribed (mesh) reference.

Only numpy is used.
"""
from __future__ import annotations

import numpy as np
from . import massprop as mp

DEFAULT_GRADIENT_FLOOR = mp.DEFAULT_GRADIENT_FLOOR


# ---------------------------------------------------------------------------------------------
# smooth branch fields: values, gradients, d/dtheta for every branch of the CSG tree
# ---------------------------------------------------------------------------------------------
def _finger_branch_fields(points, theta, side=1):
    p = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    r, alpha, kappa = (float(v) for v in theta)
    c = np.array([mp.FINGER_XF0, 0.0, 0.0])
    pm = p.copy()
    if side < 0:
        pm[:, 0] *= -1.0
    d = pm - c
    ca, sa = np.cos(alpha), np.sin(alpha)
    q = np.empty_like(pm)
    q[:, 0] = ca * d[:, 0] + sa * d[:, 1]
    q[:, 1] = -sa * d[:, 0] + ca * d[:, 1]
    q[:, 2] = d[:, 2]
    dq = np.empty_like(pm)
    dq[:, 0] = -sa * d[:, 0] + ca * d[:, 1]
    dq[:, 1] = -ca * d[:, 0] - sa * d[:, 1]
    dq[:, 2] = 0.0
    qx, qy, qz = q.T
    n = len(p)

    # face branch
    vf0 = np.ones(n)
    vf1 = kappa * qy
    df = qx + 0.5 * kappa * qy * qy
    gf = np.column_stack((ca * vf0 - sa * vf1, sa * vf0 + ca * vf1, np.zeros(n)))
    sf = vf0 * dq[:, 0] + vf1 * dq[:, 1]
    s_face = np.column_stack((np.zeros(n), sf, 0.5 * qy * qy))

    # side branch: rounded box in xy
    c0 = np.array([-mp.FINGER_L / 2.0, 0.0])
    b = np.array([mp.FINGER_L / 2.0, mp.FINGER_H / 2.0])
    dxy = np.abs(q[:, :2] - c0) - b
    ax = np.maximum(dxy[:, 0], 0.0)
    ay = np.maximum(dxy[:, 1], 0.0)
    ute = np.hypot(ax, ay)
    inne = np.minimum(np.maximum(dxy[:, 0], dxy[:, 1]), 0.0)
    dp2 = ute + inne - r
    sx = np.where(qx - c0[0] >= 0.0, 1.0, -1.0)
    sy = np.where(qy - c0[1] >= 0.0, 1.0, -1.0)
    g2 = np.zeros((n, 2), dtype=np.float64)
    outside = ute > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gx = np.where(ax > 0.0, ax / np.where(outside, ute, 1.0) * sx, 0.0)
        gy = np.where(ay > 0.0, ay / np.where(outside, ute, 1.0) * sy, 0.0)
    g2[outside, 0] = gx[outside]
    g2[outside, 1] = gy[outside]
    if np.any(~outside):
        k = np.argmax(dxy[~outside], axis=1)
        rows = np.flatnonzero(~outside)
        g2[rows, k] = np.where(q[rows, k] - c0[k] >= 0.0, 1.0, -1.0)
    grad_side = np.column_stack((ca * g2[:, 0] - sa * g2[:, 1],
                                 sa * g2[:, 0] + ca * g2[:, 1],
                                 np.zeros(n)))
    s_side = np.column_stack((-np.ones(n), g2[:, 0] * dq[:, 0] + g2[:, 1] * dq[:, 1],
                              np.zeros(n)))

    # z branch: slab
    dz = np.abs(qz) - mp.FINGER_W / 2.0
    grad_z = np.column_stack((np.zeros(n), np.zeros(n), np.where(qz >= 0.0, 1.0, -1.0)))
    s_z = np.zeros((n, 3))

    values = np.column_stack((df, dp2, dz))
    gradients = np.stack((gf, grad_side, grad_z), axis=1)
    dthetas = np.stack((s_face, s_side, s_z), axis=1)
    if side < 0:
        gradients[:, :, 0] *= -1.0
    return values, gradients, dthetas


def _femur_branch_fields(points, theta):
    p = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    length, neck_angle = (float(v) for v in theta)
    x, y, z = p.T
    n = len(p)
    radial = np.hypot(x, y)
    safe = np.where(radial > 1.0e-14, radial, 1.0)
    ux = np.where(radial > 1.0e-14, x / safe, 1.0)
    uy = np.where(radial > 1.0e-14, y / safe, 0.0)
    radial_distance = radial - mp.FEMUR_RADIUS
    lower = -z
    upper = z - length

    g_side = np.column_stack((ux, uy, np.zeros(n)))
    g_lower = np.column_stack((np.zeros(n), np.zeros(n), -np.ones(n)))
    g_upper = np.column_stack((np.zeros(n), np.zeros(n), np.ones(n)))

    center = np.array([mp.FEMUR_NECK * np.sin(neck_angle), 0.0,
                       length + mp.FEMUR_NECK * np.cos(neck_angle)])
    rel = p - center
    distance = np.linalg.norm(rel, axis=1)
    sd = np.where(distance > 1.0e-14, distance, 1.0)
    sphere_distance = distance - mp.FEMUR_HEAD_RADIUS
    g_sphere = rel / sd[:, None]
    center_dlength = np.array([0.0, 0.0, 1.0])
    center_dangle = np.array([mp.FEMUR_NECK * np.cos(neck_angle), 0.0,
                              -mp.FEMUR_NECK * np.sin(neck_angle)])
    s_sphere = np.column_stack((-np.einsum("ij,j->i", rel, center_dlength) / sd,
                                -np.einsum("ij,j->i", rel, center_dangle) / sd))

    zz = np.zeros(n)
    s_zero = np.column_stack((zz, zz))
    s_upper = np.column_stack((-np.ones(n), zz))

    values = np.column_stack((radial_distance, lower, upper, sphere_distance))
    gradients = np.stack((g_side, g_lower, g_upper, g_sphere), axis=1)
    dthetas = np.stack((s_zero, s_zero, s_upper, s_sphere), axis=1)
    return values, gradients, dthetas


def _branch_fields(shape, points, theta, side=1):
    if shape == "finger":
        return _finger_branch_fields(points, theta, side=side)
    if shape == "femur":
        return _femur_branch_fields(points, theta)
    raise ValueError(f"unknown shape: {shape}")


def _active_branch(shape, values):
    if shape == "finger":
        return np.argmax(values, axis=1).astype(np.int8)
    cylinder = np.maximum(values[:, 0], np.maximum(values[:, 1], values[:, 2]))
    inner = np.argmax(values[:, :3], axis=1)
    return np.where(values[:, 3] < cylinder, 3, inner).astype(np.int8)


# ---------------------------------------------------------------------------------------------
# kink split
# ---------------------------------------------------------------------------------------------
def _crossing(pa, pb, ba, bb, va, vb):
    h0 = va[ba] - va[bb]
    h1 = vb[ba] - vb[bb]
    den = h0 - h1
    if abs(den) < 1.0e-30:
        t = 0.5
    else:
        t = h0 / den
    t = min(1.0, max(0.0, float(t)))
    return pa + t * (pb - pa)


def _crease_project(shape, theta, side, point, ba, bb, iterations=4):
    """Newton-project a point onto the true crease {f_a = 0, f_b = 0} of two smooth branches."""
    c = np.asarray(point, dtype=np.float64).copy()
    for _ in range(iterations):
        values, gradients, _ = _branch_fields(shape, c.reshape(1, 3), theta, side=side)
        f = np.array([values[0, ba], values[0, bb]])
        jac = np.stack((gradients[0, ba], gradients[0, bb]))
        jjt = jac @ jac.T
        if abs(np.linalg.det(jjt)) < 1.0e-30:
            break
        c = c - jac.T @ np.linalg.solve(jjt, f)
    return c


def _split_one(p, branch, values, crease):
    b0, b1, b2 = int(branch[0]), int(branch[1]), int(branch[2])
    if b0 == b1 == b2:
        return [(p, b0)]
    edges = [(0, 1), (1, 2), (2, 0)]
    differing = [(i, j) for i, j in edges if branch[i] != branch[j]]
    if len(differing) == 2:
        if b0 == b1:
            u, o1, o2 = 2, 0, 1
        elif b1 == b2:
            u, o1, o2 = 0, 1, 2
        else:
            u, o1, o2 = 1, 0, 2
        c1 = crease(p[u], p[o1], branch[u], branch[o1], values[u], values[o1])
        c2 = crease(p[u], p[o2], branch[u], branch[o2], values[u], values[o2])
        return [
            (np.stack((p[u], c1, c2)), branch[u]),
            (np.stack((c1, p[o1], p[o2])), branch[o1]),
            (np.stack((c1, p[o2], c2)), branch[o1]),
        ]
    # three distinct branches
    c01 = crease(p[0], p[1], b0, b1, values[0], values[1])
    c12 = crease(p[1], p[2], b1, b2, values[1], values[2])
    c20 = crease(p[2], p[0], b2, b0, values[2], values[0])
    return [
        (np.stack((p[0], c01, c20)), b0),
        (np.stack((p[1], c12, c01)), b1),
        (np.stack((p[2], c20, c12)), b2),
        (np.stack((c01, c12, c20)), -1),
    ]


def _project_to_surface(shape, points, theta, side=1, iterations=3):
    """Newton-project points onto the true zero level set of the analytic leaf SDF."""
    p = np.asarray(points, dtype=np.float64).copy()
    for _ in range(iterations):
        field = mp._field(shape, p, theta, side=side)
        gradient = field["gradient"]
        sdf = field["sdf"]
        gn2 = np.sum(gradient * gradient, axis=1)
        step = np.where(gn2 > 1.0e-30, sdf / gn2, 0.0)
        p = p - step[:, None] * gradient
    return p


def kink_surface(shape, theta, lo, hi, n, side=1,
                 gradient_floor=DEFAULT_GRADIENT_FLOOR, project=True):
    theta = np.asarray(theta, dtype=np.float64)
    raw = mp.marching_surface(shape, theta, lo, hi, n, side=side)
    triangles = raw["triangle_points"]
    if project:
        triangles = _project_to_surface(shape, triangles.reshape((-1, 3)), theta,
                                        side=side).reshape(triangles.shape)
    count = triangles.shape[0]
    flat = triangles.reshape((-1, 3))
    values, gradients, dthetas = _branch_fields(shape, flat, theta, side=side)
    k = values.shape[1]
    p = len(theta)
    values = values.reshape(count, 3, k)
    gradients = gradients.reshape(count, 3, k, 3)
    dthetas = dthetas.reshape(count, 3, k, p)
    vertex_branch = _active_branch(shape, values.reshape(-1, k)).reshape(count, 3)
    uniform = np.all(vertex_branch == vertex_branch[:, :1], axis=1)

    sub_points = []
    sub_branch = []

    def crease(pa, pb, ba, bb, va, vb):
        c = _crossing(pa, pb, ba, bb, va, vb)
        return _crease_project(shape, theta, side, c, int(ba), int(bb))

    for idx in np.flatnonzero(uniform):
        sub_points.append(triangles[idx])
        sub_branch.append(int(vertex_branch[idx, 0]))
    for idx in np.flatnonzero(~uniform):
        for pts, br in _split_one(triangles[idx], vertex_branch[idx], values[idx], crease):
            sub_points.append(pts)
            sub_branch.append(br)
    sub_points = np.stack(sub_points, axis=0)
    sub_branch = np.asarray(sub_branch, dtype=np.int64)
    total = sub_points.shape[0]

    centroids = sub_points.mean(axis=1)
    cvals, cgrads, cdts = _branch_fields(shape, centroids, theta, side=side)
    resolved = _active_branch(shape, cvals)
    use = np.where(sub_branch >= 0, sub_branch, resolved)
    rows = np.arange(total)
    g_b = cgrads[rows, use]
    s_b = cdts[rows, use]
    gnorm = np.linalg.norm(g_b, axis=1)
    velocity = -s_b / np.maximum(gnorm, 1.0e-14)[:, None]

    cross = np.cross(sub_points[:, 1] - sub_points[:, 0], sub_points[:, 2] - sub_points[:, 0])
    align = np.einsum("ij,ij->i", cross, g_b) < 0.0
    sub_points[align] = sub_points[align][:, [0, 2, 1], :]
    cross = np.cross(sub_points[:, 1] - sub_points[:, 0], sub_points[:, 2] - sub_points[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    centroid = sub_points.mean(axis=1)
    valid = (area > 0.0) & (gnorm >= gradient_floor) & np.all(np.isfinite(velocity), axis=1)
    total_area = float(np.sum(area))
    valid_area = float(np.sum(area[valid]))
    return {
        "triangle_points": sub_points,
        "area": area,
        "centroid": centroid,
        "velocity": velocity,
        "gradient_norm": gnorm,
        "valid": valid,
        "branch": use,
        "sub_triangles": int(total),
        "area_total": total_area,
        "area_valid": valid_area,
        "excluded_area": float(total_area - valid_area),
        "coverage": float(valid_area / total_area) if total_area else 0.0,
        "branch_coverage": float(np.sum(area[area > 0.0]) / total_area) if total_area else 0.0,
        "gradient_coverage": (float(np.sum(area[gnorm >= gradient_floor]) / total_area)
                              if total_area else 0.0),
        "triangles_valid": int(np.sum(valid)),
        "pitch_m": float(np.max(np.asarray(hi, dtype=np.float64)
                                - np.asarray(lo, dtype=np.float64)) / n),
    }


# ---------------------------------------------------------------------------------------------
# surface mass derivatives
# ---------------------------------------------------------------------------------------------
def surface_mass_derivatives_kink(shape, theta, lo, hi, n, side=1,
                                  gradient_floor=DEFAULT_GRADIENT_FLOOR,
                                  base_properties=None):
    theta = np.asarray(theta, dtype=np.float64)
    surface = kink_surface(shape, theta, lo, hi, n, side=side,
                           gradient_floor=gradient_floor)
    valid = surface["valid"]
    area = surface["area"][valid]
    centroid = surface["centroid"][valid]
    velocity = surface["velocity"][valid]
    if not len(area):
        raise RuntimeError("no valid kink-aware surface triangles")
    weight = area[:, None] * velocity
    d_volume = np.sum(weight, axis=0)
    d_first = np.einsum("np,ni->pi", weight, centroid)
    d_raw = np.einsum("np,ni,nj->pij", weight, centroid, centroid)

    d_com = None
    d_central = None
    d_physical = None
    if base_properties is not None:
        volume = float(base_properties["volume"])
        first = np.asarray(base_properties["first_moment"], dtype=np.float64)
        centre = first / volume
        d_com = (d_first * volume - first[None, :] * d_volume[:, None]) / (volume * volume)
        d_central = (d_raw
                     - d_volume[:, None, None] * np.outer(centre, centre)[None, :, :]
                     - volume * (np.einsum("pi,j->pij", d_com, centre)
                                 + np.einsum("i,pj->pij", centre, d_com)))
        d_physical = (np.trace(d_central, axis1=1, axis2=2)[:, None, None]
                      * np.eye(3)[None, :, :] - d_central)
    return {
        "d_volume": d_volume,
        "d_first_moment": d_first,
        "d_raw_second_moment": d_raw,
        "d_com": d_com,
        "d_central_inertia": d_central,
        "d_physical_inertia": d_physical,
        "surface": surface,
    }


# ---------------------------------------------------------------------------------------------
# exact analytic references (1-D quadrature, no SDF mesh)
# ---------------------------------------------------------------------------------------------
def _finger_area_moments(r, kappa, ny):
    a = mp.FINGER_L / 2.0
    b = mp.FINGER_H / 2.0
    c0x = -mp.FINGER_L / 2.0
    span = b + r
    y = np.linspace(-span, span, ny)
    dy = np.maximum(np.abs(y) - b, 0.0)
    s = np.sqrt(np.maximum(r * r - dy * dy, 0.0))
    inside = np.abs(y) <= b + r
    xlo = c0x - a - s
    xhi = c0x + a + s
    xpar = -0.5 * kappa * y * y
    xh = np.minimum(xhi, xpar)
    zero = np.zeros_like(y)
    wdt = np.where(inside, np.maximum(xh - xlo, 0.0), zero)
    ax = np.where(inside, 0.5 * (xh ** 2 - xlo ** 2), zero)
    ay = np.where(inside, y * wdt, zero)
    axx = np.where(inside, (xh ** 3 - xlo ** 3) / 3.0, zero)
    ayy = np.where(inside, y * y * wdt, zero)
    axy = np.where(inside, y * 0.5 * (xh ** 2 - xlo ** 2), zero)
    area = np.trapezoid(wdt, y)
    return np.array([area, np.trapezoid(ax, y), np.trapezoid(ay, y),
                     np.trapezoid(axx, y), np.trapezoid(ayy, y), np.trapezoid(axy, y)])


def finger_exact_properties(theta, ny=400001):
    r, alpha, kappa = (float(v) for v in theta)
    area, ax, ay, axx, ayy, axy = _finger_area_moments(r, kappa, ny)
    w = mp.FINGER_W
    volume = w * area
    first_q = np.array([w * ax, w * ay, 0.0])
    raw_q = np.array([[w * axx, w * axy, 0.0],
                      [w * axy, w * ayy, 0.0],
                      [0.0, 0.0, w ** 3 / 12.0 * area]])
    c = np.array([mp.FINGER_XF0, 0.0, 0.0])
    ca, sa = np.cos(alpha), np.sin(alpha)
    rot = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    first_q_world = rot @ first_q
    first = volume * c + first_q_world
    raw = (volume * np.outer(c, c) + np.outer(c, first_q_world)
           + np.outer(first_q_world, c) + rot @ raw_q @ rot.T)
    com = first / volume
    central = raw - volume * np.outer(com, com)
    return {"volume": volume, "first_moment": first, "raw_second_moment": raw,
            "com": com, "central_inertia": central}


def femur_exact_properties(theta):
    """Exact moments of the declared femur proxy: capped cylinder union spherical head.

    The declared neck geometry leaves the head disjoint from the shaft over the tested
    theta range (``neck cos(angle) > head_radius``); the union is then a disjoint sum. The
    function raises if the two bodies would overlap, rather than silently dropping the
    intersection.
    """
    length, neck_angle = (float(v) for v in theta)
    radius = mp.FEMUR_RADIUS
    head = mp.FEMUR_HEAD_RADIUS
    neck = mp.FEMUR_NECK
    xc = neck * np.sin(neck_angle)
    zc = length + neck * np.cos(neck_angle)
    if xc <= radius:
        separation = neck * np.cos(neck_angle)
    else:
        separation = np.hypot(xc - radius, neck * np.cos(neck_angle))
    if separation < head:
        raise RuntimeError("femur head overlaps the shaft; disjoint analytic reference invalid")

    vol_c = np.pi * radius ** 2 * length
    first_c = np.array([0.0, 0.0, vol_c * length / 2.0])
    raw_c = np.diag([vol_c * radius ** 2 / 4.0, vol_c * radius ** 2 / 4.0,
                     np.pi * radius ** 2 * length ** 3 / 3.0])

    vol_s = 4.0 / 3.0 * np.pi * head ** 3
    centre_s = np.array([xc, 0.0, zc])
    first_s = vol_s * centre_s
    raw_s = vol_s * np.outer(centre_s, centre_s) + (vol_s * head ** 2 / 5.0) * np.eye(3)

    volume = vol_c + vol_s
    first = first_c + first_s
    raw = raw_c + raw_s
    com = first / volume
    central = raw - volume * np.outer(com, com)
    return {"volume": volume, "first_moment": first, "raw_second_moment": raw,
            "com": com, "central_inertia": central}


def exact_properties(shape, theta, **kwargs):
    if shape == "finger":
        return finger_exact_properties(theta, **kwargs)
    if shape == "femur":
        return femur_exact_properties(theta, **kwargs)
    raise ValueError(f"unknown shape: {shape}")


def _fd_components(properties_fn, theta, steps):
    theta = np.asarray(theta, dtype=np.float64)
    base = properties_fn(theta)
    volume = float(base["volume"])
    first = np.asarray(base["first_moment"], dtype=np.float64)
    centre = first / volume
    out = {"volume": [], "first_moment": [], "com": [], "raw_second_moment": [],
           "central_inertia": [], "physical_inertia": []}
    for j, h in enumerate(steps):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += h
        minus[j] -= h
        a = properties_fn(plus)
        b = properties_fn(minus)
        dv = (a["volume"] - b["volume"]) / (2.0 * h)
        dm = (a["first_moment"] - b["first_moment"]) / (2.0 * h)
        draw = (a["raw_second_moment"] - b["raw_second_moment"]) / (2.0 * h)
        dcom = (dm * volume - first * dv) / (volume * volume)
        dc = (draw - dv * np.outer(centre, centre)
              - volume * (np.outer(dcom, centre) + np.outer(centre, dcom)))
        dp = np.trace(dc) * np.eye(3) - dc
        out["volume"].append(float(dv))
        out["first_moment"].append(np.asarray(dm).tolist())
        out["com"].append(np.asarray(dcom).tolist())
        out["raw_second_moment"].append(np.asarray(draw).tolist())
        out["central_inertia"].append(np.asarray(dc).tolist())
        out["physical_inertia"].append(np.asarray(dp).tolist())
    return out


def exact_central_fd(shape, theta, steps, **kwargs):
    return _fd_components(lambda t: exact_properties(shape, t, **kwargs), theta, steps)
