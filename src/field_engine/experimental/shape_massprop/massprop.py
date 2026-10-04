"""Surface and voxel mass-property derivatives for the bounded implicit-shape experiment."""
from __future__ import annotations

import math
import time

import numpy as np

FINGER_L = 0.06
FINGER_H = 0.05
FINGER_W = 0.06
FINGER_XF0 = -0.05
FINGER_R0 = 0.004
FINGER_ALPHA0 = 0.02
FINGER_KAPPA0 = 5.0
FEMUR_RADIUS = 0.018
FEMUR_HEAD_RADIUS = 0.026
FEMUR_NECK = 0.030
FEMUR_LENGTH0 = 0.120
FEMUR_NECK_ANGLE0 = 0.35
DEFAULT_GRADIENT_FLOOR = 0.25


def _flat_points(points):
    return np.asarray(points, dtype=np.float64).reshape((-1, 3))


def finger_field(points, theta, side=1):
    p = _flat_points(points)
    r, alpha, kappa = (float(v) for v in theta)
    c = np.array([FINGER_XF0, 0.0, 0.0], dtype=np.float64)
    pm = p.copy()
    if side < 0:
        pm[:, 0] *= -1.0
    d = pm - c
    ca, sa = math.cos(alpha), math.sin(alpha)
    q = np.empty_like(pm)
    q[:, 0] = ca * d[:, 0] + sa * d[:, 1]
    q[:, 1] = -sa * d[:, 0] + ca * d[:, 1]
    q[:, 2] = d[:, 2]
    dq = np.empty_like(pm)
    dq[:, 0] = -sa * d[:, 0] + ca * d[:, 1]
    dq[:, 1] = -ca * d[:, 0] - sa * d[:, 1]
    dq[:, 2] = 0.0

    qx, qy, qz = q.T
    vf = np.column_stack((np.ones_like(qx), kappa * qy, np.zeros_like(qx)))
    df = qx + 0.5 * kappa * qy * qy
    sf = np.einsum("ij,ij->i", vf, dq)
    gf = np.column_stack((ca * vf[:, 0] - sa * vf[:, 1],
                          sa * vf[:, 0] + ca * vf[:, 1],
                          vf[:, 2]))

    c0 = np.array([-FINGER_L / 2.0, 0.0])
    b = np.array([FINGER_L / 2.0, FINGER_H / 2.0])
    dxy = np.abs(q[:, :2] - c0) - b
    ax = np.maximum(dxy[:, 0], 0.0)
    ay = np.maximum(dxy[:, 1], 0.0)
    ute = np.hypot(ax, ay)
    inne = np.minimum(np.maximum(dxy[:, 0], dxy[:, 1]), 0.0)
    dp2 = ute + inne - r
    sx = np.where(qx - c0[0] >= 0.0, 1.0, -1.0)
    sy = np.where(qy - c0[1] >= 0.0, 1.0, -1.0)
    g2 = np.zeros((len(p), 2), dtype=np.float64)
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
    dp2_alpha = g2[:, 0] * dq[:, 0] + g2[:, 1] * dq[:, 1]
    dz = np.abs(qz) - FINGER_W / 2.0
    z_branch = dz > dp2
    dpad = np.maximum(dp2, dz)
    gpad_q = np.zeros_like(q)
    gpad_q[:, :2] = g2
    gpad_q[z_branch, 2] = np.where(qz[z_branch] >= 0.0, 1.0, -1.0)
    gpad = np.column_stack((ca * gpad_q[:, 0] - sa * gpad_q[:, 1],
                            sa * gpad_q[:, 0] + ca * gpad_q[:, 1],
                            gpad_q[:, 2]))
    spad = np.column_stack((np.where(z_branch, 0.0, -1.0),
                             np.where(z_branch, 0.0, dp2_alpha),
                             np.zeros(len(p))))


    face_active = df >= dpad
    sdf = np.where(face_active, df, dpad)
    gradient = np.where(face_active[:, None], gf, gpad)
    dtheta = np.column_stack((np.where(face_active, 0.0, spad[:, 0]),
                              np.where(face_active, sf, spad[:, 1]),
                              np.where(face_active, 0.5 * qy * qy, 0.0)))
    if side < 0:
        gradient[:, 0] *= -1.0
    branch = np.where(face_active, 0, np.where(z_branch, 2, 1)).astype(np.int8)
    return {"sdf": sdf, "gradient": gradient, "d_sdf_dtheta": dtheta, "branch": branch}


def femur_field(points, theta):
    p = _flat_points(points)
    length, neck_angle = (float(v) for v in theta)
    x, y, z = p.T
    radial = np.hypot(x, y)
    safe = np.where(radial > 1.0e-14, radial, 1.0)
    ux = np.where(radial > 1.0e-14, x / safe, 1.0)
    uy = np.where(radial > 1.0e-14, y / safe, 0.0)
    radial_distance = radial - FEMUR_RADIUS
    lower = -z
    upper = z - length
    axial = np.maximum(lower, upper)
    cylinder_distance = np.maximum(radial_distance, axial)
    cylinder_side = radial_distance >= axial
    cylinder_lower = (~cylinder_side) & (lower >= upper)
    cylinder_upper = (~cylinder_side) & (~cylinder_lower)
    cylinder_gradient = np.column_stack((np.where(cylinder_side, ux, 0.0),
                                         np.where(cylinder_side, uy, 0.0),
                                         np.where(cylinder_lower, -1.0,
                                                  np.where(cylinder_upper, 1.0, 0.0))))
    cylinder_branch = np.where(cylinder_side, 0, np.where(cylinder_lower, 1, 2)).astype(np.int8)
    cylinder_dtheta = np.column_stack((np.where(cylinder_upper, -1.0, 0.0),
                                       np.zeros(len(p))))

    center = np.array([FEMUR_NECK * math.sin(neck_angle),
                       0.0,
                       length + FEMUR_NECK * math.cos(neck_angle)])
    rel = p - center
    distance = np.linalg.norm(rel, axis=1)
    safe_distance = np.where(distance > 1.0e-14, distance, 1.0)
    sphere_distance = distance - FEMUR_HEAD_RADIUS
    sphere_gradient = rel / safe_distance[:, None]
    center_dlength = np.array([0.0, 0.0, 1.0])
    center_dangle = np.array([FEMUR_NECK * math.cos(neck_angle),
                              0.0,
                              -FEMUR_NECK * math.sin(neck_angle)])
    sphere_dtheta = np.column_stack((-np.einsum("ij,j->i", rel, center_dlength) / safe_distance,
                                     -np.einsum("ij,j->i", rel, center_dangle) / safe_distance))
    sphere_active = sphere_distance < cylinder_distance
    sdf = np.where(sphere_active, sphere_distance, cylinder_distance)
    gradient = np.where(sphere_active[:, None], sphere_gradient, cylinder_gradient)
    dtheta = np.where(sphere_active[:, None], sphere_dtheta, cylinder_dtheta)
    branch = np.where(sphere_active, 3, cylinder_branch).astype(np.int8)
    return {"sdf": sdf, "gradient": gradient, "d_sdf_dtheta": dtheta, "branch": branch}


def _field(shape, points, theta, side=1):
    if shape == "finger":
        return finger_field(points, theta, side=side)
    if shape == "femur":
        return femur_field(points, theta)
    raise ValueError(f"unknown shape: {shape}")


def regular_grid(lo, hi, n, centers=False):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if centers:
        axes = [lo[k] + (np.arange(n, dtype=np.float64) + 0.5) * (hi[k] - lo[k]) / n
                for k in range(3)]
    else:
        axes = [np.linspace(lo[k], hi[k], n + 1) for k in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))


def _grid_field(shape, theta, lo, hi, n, side=1):
    points = regular_grid(lo, hi, n, centers=False)
    values = _field(shape, points, theta, side=side)
    grid_shape = (n + 1, n + 1, n + 1)
    return {
        "points": points,
        "sdf": values["sdf"].reshape(grid_shape),
        "gradient": values["gradient"].reshape(grid_shape + (3,)),
        "dtheta": values["d_sdf_dtheta"].reshape(grid_shape + (len(theta),)),
        "branch": values["branch"].reshape(grid_shape),
    }


_CUBE_CORNERS = np.asarray([
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
], dtype=np.int64)
_TETRA = np.asarray([
    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
], dtype=np.int64)


def _cube_indices(n):
    shape = (n, n, n, 8)
    ii = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None, None, None], shape)
    jj = np.broadcast_to(np.arange(n, dtype=np.int64)[None, :, None, None], shape)
    kk = np.broadcast_to(np.arange(n, dtype=np.int64)[None, None, :, None], shape)
    return np.stack((ii + _CUBE_CORNERS[:, 0].reshape(1, 1, 1, 8),
                     jj + _CUBE_CORNERS[:, 1].reshape(1, 1, 1, 8),
                     kk + _CUBE_CORNERS[:, 2].reshape(1, 1, 1, 8)), axis=-1)


def _tet_indices(n):
    cubes = _cube_indices(n)
    return cubes[..., _TETRA, :]


def _edge(points, dtheta, gradients, branches, values, first, second, selected):
    f0 = values[selected, first]
    f1 = values[selected, second]
    den = f0 - f1
    t = np.divide(f0, den, out=np.zeros_like(f0), where=np.abs(den) > 1.0e-15)
    p0 = points[selected, first]
    p1 = points[selected, second]
    d0 = dtheta[selected, first]
    d1 = dtheta[selected, second]
    g0 = gradients[selected, first]
    g1 = gradients[selected, second]
    p = p0 + t[:, None] * (p1 - p0)
    d = d0 + t[:, None] * (d1 - d0)
    g = g0 + t[:, None] * (g1 - g0)
    b0 = branches[selected, first]
    b1 = branches[selected, second]
    b = np.where(t < 0.5, b0, b1)
    return p, d, g, b, b0 == b1


def _append_triangle(store, edges, edge_ok):
    store[0].append(np.stack([edge[0] for edge in edges], axis=1))
    store[1].append(np.stack([edge[1] for edge in edges], axis=1))
    store[2].append(np.stack([edge[2] for edge in edges], axis=1))
    store[3].append(np.stack([edge[3] for edge in edges], axis=1))
    store[4].append(edge_ok)


def marching_surface(shape, theta, lo, hi, n, side=1, gradient_floor=DEFAULT_GRADIENT_FLOOR,
                     sdf_band_pitch=1.5):
    grid = _grid_field(shape, theta, lo, hi, n, side=side)
    values = grid["sdf"]
    points = grid["points"]
    gradients = grid["gradient"]
    dtheta = grid["dtheta"]
    branches = grid["branch"]
    indices = _tet_indices(n).reshape((-1, 4, 3))
    node_ids = (indices[..., 2] * (n + 1) + indices[..., 1]) * (n + 1) + indices[..., 0]
    tet_points = points[node_ids]
    tet_values = values.reshape(-1)[node_ids]
    tet_dtheta = dtheta.reshape((-1, len(theta)))[node_ids]
    tet_gradients = gradients.reshape((-1, 3))[node_ids]
    tet_branches = branches.reshape(-1)[node_ids]
    inside = tet_values <= 0.0
    count = np.sum(inside, axis=1)
    store = ([], [], [], [], [])
    one = np.flatnonzero(count == 1)
    if len(one):
        active_vertex = np.argmax(inside[one], axis=1)
        for a in range(4):
            local = np.flatnonzero(active_vertex == a)
            selected = one[local]
            if not len(selected):
                continue
            outside = [j for j in range(4) if j != a]
            edges = []
            for j in outside:
                edges.append(_edge(tet_points, tet_dtheta, tet_gradients, tet_branches,
                                   tet_values, np.full(len(selected), a), np.full(len(selected), j),
                                   selected))
            _append_triangle(store, edges, np.logical_and.reduce([e[4] for e in edges]))
    three = np.flatnonzero(count == 3)
    if len(three):
        outside_vertex = np.argmin(inside[three], axis=1)
        for a in range(4):
            local = np.flatnonzero(outside_vertex == a)
            selected = three[local]
            if not len(selected):
                continue
            inside_vertices = [j for j in range(4) if j != a]
            edges = []
            for j in inside_vertices:
                edges.append(_edge(tet_points, tet_dtheta, tet_gradients, tet_branches,
                                   tet_values, np.full(len(selected), a), np.full(len(selected), j),
                                   selected))
            _append_triangle(store, edges, np.logical_and.reduce([e[4] for e in edges]))
    two = np.flatnonzero(count == 2)
    for a in range(4):
        for b in range(a + 1, 4):
            selected = two[inside[two, a] & inside[two, b]]
            if not len(selected):
                continue
            outside = [j for j in range(4) if j not in (a, b)]
            edges = {}
            for i, j in ((a, outside[0]), (a, outside[1]), (b, outside[0]), (b, outside[1])):
                edges[(i, j)] = _edge(tet_points, tet_dtheta, tet_gradients, tet_branches,
                                       tet_values, np.full(len(selected), i), np.full(len(selected), j),
                                       selected)
            ac, ad, bc, bd = edges[(a, outside[0])], edges[(a, outside[1])], edges[(b, outside[0])], edges[(b, outside[1])]
            for first, second, third in ((ac, bc, bd), (ac, bd, ad)):
                _append_triangle(store, (first, second, third),
                                 np.logical_and(first[4], np.logical_and(second[4], third[4])))
    if not store[0]:
        raise RuntimeError("level-set extraction produced no triangles")
    tri_points = np.concatenate(store[0], axis=0)
    tri_dtheta = np.concatenate(store[1], axis=0)
    tri_gradient = np.concatenate(store[2], axis=0)
    tri_branch = np.concatenate(store[3], axis=0)
    tri_edge_ok = np.concatenate(store[4], axis=0)
    cross = np.cross(tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0])
    mean_gradient = np.mean(tri_gradient, axis=1)
    flip = np.einsum("ij,ij->i", cross, mean_gradient) < 0.0
    tri_points[flip] = tri_points[flip][:, [0, 2, 1], :]
    tri_dtheta[flip] = tri_dtheta[flip][:, [0, 2, 1], :]
    tri_gradient[flip] = tri_gradient[flip][:, [0, 2, 1], :]
    cross = np.cross(tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    centroid = np.mean(tri_points, axis=1)
    gnorm = np.linalg.norm(tri_gradient, axis=2)
    safe_grad = np.maximum(gnorm, 1.0e-14)
    velocity = -np.mean(tri_dtheta / safe_grad[:, :, None], axis=1)
    branch_same = np.all(tri_branch == tri_branch[:, :1], axis=1)
    branch_valid = tri_edge_ok & branch_same & (area > 0.0)
    gradient_valid = (np.mean(gnorm, axis=1) >= gradient_floor) & (area > 0.0)
    valid = branch_valid & gradient_valid
    pitch = (np.asarray(hi) - np.asarray(lo)) / n
    scalar_pitch = float(np.max(pitch))
    location = sdf_band_pitch * scalar_pitch / np.maximum(np.mean(gnorm, axis=1), 1.0e-14)
    total_area = float(np.sum(area))
    return {
        "triangle_points": tri_points,
        "area": area,
        "centroid": centroid,
        "velocity": velocity,
        "gradient": tri_gradient,
        "gradient_norm": np.mean(gnorm, axis=1),
        "valid": valid,
        "branch": tri_branch,
        "location_uncertainty": location,
        "area_total": total_area,
        "area_valid": float(np.sum(area[valid])),
        "excluded_area": float(np.sum(area[~valid])),
        "branch_coverage": float(np.sum(area[branch_valid]) / total_area)
                            if total_area else 0.0,
        "gradient_coverage": float(np.sum(area[gradient_valid]) / total_area)
                             if total_area else 0.0,
        "branch_excluded_area": float(np.sum(area[~branch_valid])),
        "gradient_excluded_area": float(np.sum(area[~gradient_valid])),
        "triangles": int(len(area)),
        "triangles_valid": int(np.sum(valid)),
        "pitch_m": scalar_pitch,
    }


def _outer_sum(weight, points):
    return np.einsum("ni,nj->ij", weight[:, None] * points[:, :, None], points[:, None, :])


def _moment_bound(weight, points, location, velocity):
    scale = np.abs(weight) * (1.0 + np.abs(velocity))
    point_norm = np.linalg.norm(points, axis=1)
    vector = np.sum(scale * location * (1.0 + point_norm))
    matrix = np.sum(scale * location * (1.0 + point_norm + point_norm * point_norm))
    return float(vector), float(matrix)


def surface_mass_derivatives(shape, theta, lo, hi, n, side=1,
                             sdf_band_pitch=1.5, gradient_floor=DEFAULT_GRADIENT_FLOOR,
                             base_properties=None):
    theta = np.asarray(theta, dtype=np.float64)
    surface = marching_surface(shape, theta, lo, hi, n, side=side,
                               gradient_floor=gradient_floor,
                               sdf_band_pitch=sdf_band_pitch)
    valid = surface["valid"]
    points = surface["triangle_points"][valid]
    area = surface["area"][valid]
    centroid = surface["centroid"][valid]
    velocity = surface["velocity"][valid]
    location = surface["location_uncertainty"][valid]
    if not len(area):
        raise RuntimeError("no valid surface triangles")
    area_parameter = area[:, None]
    weight = area_parameter * velocity
    d_volume = np.sum(weight, axis=0)
    d_first = np.einsum("np,ni->pi", weight, centroid)
    d_raw = np.einsum("np,ni,nj->pij", weight, centroid, centroid)

    p0, p1, p2 = points[:, 0], points[:, 1], points[:, 2]
    m01, m02, m12 = (p0 + p1) * 0.5, (p0 + p2) * 0.5, (p1 + p2) * 0.5
    sub_centers = np.stack(((p0 + p1 + m01) / 3.0,
                            (p0 + p2 + m02) / 3.0,
                            (p0 + m01 + m02) / 3.0,
                            (p1 + p2 + m12) / 3.0), axis=1)
    sub_weight = weight[:, None, :] / 4.0
    sub_volume = np.sum(sub_weight, axis=(0, 1))
    sub_first = np.einsum("nkp,nki->pi", sub_weight, sub_centers)
    sub_raw = np.einsum("nkp,nki,nkj->pij", sub_weight, sub_centers, sub_centers)
    quadrature_volume = np.abs(d_volume - sub_volume)
    quadrature_first = np.abs(d_first - sub_first)
    quadrature_raw = np.abs(d_raw - sub_raw)

    point_norm = np.linalg.norm(centroid, axis=1)
    speed_factor = 1.0 + np.abs(velocity)
    location_scale = area[:, None] * location[:, None] * speed_factor
    location_volume = np.sum(location_scale, axis=0)
    location_first = np.sum(location_scale[:, :, None]
                            * (1.0 + point_norm[:, None, None]), axis=0)
    location_raw = np.sum(location_scale[:, :, None, None]
                          * (1.0 + point_norm[:, None, None, None]) ** 2, axis=0)
    total_volume = quadrature_volume + location_volume
    total_first = quadrature_first + location_first
    total_raw = quadrature_raw + location_raw

    d_com = None
    d_central = None
    d_physical = None
    d_com_bound = None
    d_central_bound = None
    d_physical_bound = None
    if base_properties is not None:
        volume = float(base_properties["volume"])
        first = np.asarray(base_properties["first_moment"], dtype=np.float64)
        centre = first / volume
        d_com = (d_first * volume - first[None, :] * d_volume[:, None]) / (volume * volume)
        d_central = (d_raw
                     - d_volume[:, None, None] * np.outer(centre, centre)[None, :, :]
                     - volume * (np.einsum("pi,j->pij", d_com, centre)
                                 + np.einsum("i,pj->pij", centre, d_com)))
        d_com_bound = (total_first * volume
                       + np.abs(first)[None, :] * total_volume[:, None]) / (volume * volume)
        d_central_bound = (total_raw
                           + total_volume[:, None, None]
                           * np.abs(np.outer(centre, centre))[None, :, :]
                           + volume * (np.einsum("pi,j->pij", d_com_bound, np.abs(centre))
                                       + np.einsum("i,pj->pij", np.abs(centre), d_com_bound)))
        d_physical = (np.trace(d_central, axis1=1, axis2=2)[:, None, None]
                      * np.eye(3)[None, :, :] - d_central)
        trace_bound = np.sum(total_raw, axis=(1, 2))
        d_physical_bound = trace_bound[:, None, None] * np.eye(3)[None, :, :] + total_raw
    result = {
        "d_volume": d_volume,
        "d_first_moment": d_first,
        "d_raw_second_moment": d_raw,
        "d_com": d_com,
        "d_central_inertia": d_central,
        "d_physical_inertia": d_physical,
        "quadrature_abs": {
            "volume": quadrature_volume,
            "first_moment": quadrature_first,
            "raw_second_moment": quadrature_raw,
        },
        "sdf_location_abs": {
            "volume": location_volume,
            "first_moment": location_first,
            "raw_second_moment": location_raw,
        },
        "total_abs": {
            "volume": total_volume,
            "first_moment": total_first,
            "raw_second_moment": total_raw,
        },
        "d_com_bound": d_com_bound,
        "d_central_inertia_bound": d_central_bound,
        "d_physical_inertia_bound": d_physical_bound,
        "surface_area": surface["area_total"],
        "valid_area": surface["area_valid"],
        "excluded_area": surface["excluded_area"],
        "branch_coverage": surface["branch_coverage"],
        "gradient_coverage": surface["gradient_coverage"],
        "branch_excluded_area": surface["branch_excluded_area"],
        "gradient_excluded_area": surface["gradient_excluded_area"],
        "triangles": surface["triangles"],
        "triangles_valid": surface["triangles_valid"],
        "pitch_m": surface["pitch_m"],
        "sdf_band_m": float(sdf_band_pitch * surface["pitch_m"]),
        "gradient_floor": float(gradient_floor),
        "coverage": float(surface["area_valid"] / surface["area_total"])
                       if surface["area_total"] else 0.0,
    }
    return result, surface


def linear_tetrahedral_properties(shape, theta, lo, hi, n, side=1):
    surface = marching_surface(shape, theta, lo, hi, n, side=side)
    points = surface["triangle_points"]
    area = surface["area"]
    keep = area > 1.0e-18
    if not np.any(keep):
        raise RuntimeError("linear reference produced no nondegenerate triangles")
    points = points[keep]
    reference = 0.5 * (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64))
    q = points - reference
    signed_volume = np.einsum("ni,ni->n", q[:, 0], np.cross(q[:, 1], q[:, 2])) / 6.0
    q_centroid = np.sum(q, axis=1) / 4.0
    first_q = np.sum(signed_volume[:, None] * q_centroid, axis=0)
    q_sum = np.sum(q, axis=1)
    q_diag = np.sum(q * q, axis=1)
    covariance = (4.0 * q_diag[:, :, None] * np.eye(3)[None, :, :]
                  - q_sum[:, :, None] * q_sum[:, None, :]) / 80.0
    raw_q = np.sum(signed_volume[:, None, None]
                    * (q_centroid[:, :, None] * q_centroid[:, None, :] + covariance), axis=0)
    volume = float(np.sum(signed_volume))
    if volume <= 0.0:
        raise RuntimeError("linear reference surface is not outward oriented")
    first = first_q + volume * reference
    raw = (raw_q + np.outer(reference, first_q) + np.outer(first_q, reference)
           + volume * np.outer(reference, reference))
    com = first / volume
    central = raw - volume * np.outer(com, com)
    return {"volume": volume, "first_moment": first, "com": com,
            "raw_second_moment": raw, "central_inertia": central,
            "triangles": int(len(area)), "triangles_used": int(np.sum(keep)),
            "reference_point": reference}


def voxel_properties(shape, theta, lo, hi, n, side=1):
    points = regular_grid(lo, hi, n, centers=True)
    values = _field(shape, points, theta, side=side)["sdf"]
    inside = values <= 0.0
    h = (np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)) / n
    cell_volume = float(np.prod(h))
    volume = float(np.sum(inside) * cell_volume)
    if volume == 0.0:
        raise RuntimeError("voxel reference contains no occupied cells")
    first = np.sum(points[inside], axis=0) * cell_volume
    com = first / volume
    raw = np.einsum("ni,nj->ij", points[inside], points[inside]) * cell_volume
    central = raw - volume * np.outer(com, com)
    return {"volume": volume, "first_moment": first, "com": com,
            "raw_second_moment": raw, "central_inertia": central,
            "occupied": int(np.sum(inside)), "cell_volume": cell_volume}


def _pack_properties(value):
    return {
        "volume": float(value["volume"]),
        "first_moment": np.asarray(value["first_moment"], dtype=np.float64).tolist(),
        "com": np.asarray(value["com"], dtype=np.float64).tolist(),
        "raw_second_moment": np.asarray(value["raw_second_moment"], dtype=np.float64).tolist(),
        "central_inertia": np.asarray(value["central_inertia"], dtype=np.float64).tolist(),
    }


def voxel_central_fd(shape, theta, lo, hi, n, step, side=1):
    theta = np.asarray(theta, dtype=np.float64)
    derivatives = []
    for j in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += step
        minus[j] -= step
        a = voxel_properties(shape, plus, lo, hi, n, side=side)
        b = voxel_properties(shape, minus, lo, hi, n, side=side)
        dv = (a["volume"] - b["volume"]) / (2.0 * step)
        dm = (a["first_moment"] - b["first_moment"]) / (2.0 * step)
        draw = (a["raw_second_moment"] - b["raw_second_moment"]) / (2.0 * step)
        dcom = (a["com"] - b["com"]) / (2.0 * step)
        dic = (a["central_inertia"] - b["central_inertia"]) / (2.0 * step)
        dphys = np.trace(dic) * np.eye(3) - dic
        derivatives.append({"volume": float(dv), "first_moment": np.asarray(dm).tolist(),
                            "com": np.asarray(dcom).tolist(),
                            "raw_second_moment": np.asarray(draw).tolist(),
                            "central_inertia": np.asarray(dic).tolist(),
                            "physical_inertia": np.asarray(dphys).tolist()})
    return derivatives


def linear_tetrahedral_central_fd(shape, theta, lo, hi, n, step, side=1):
    theta = np.asarray(theta, dtype=np.float64)
    derivatives = []
    for j in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += step
        minus[j] -= step
        a = linear_tetrahedral_properties(shape, plus, lo, hi, n, side=side)
        b = linear_tetrahedral_properties(shape, minus, lo, hi, n, side=side)
        d_central = (a["central_inertia"] - b["central_inertia"]) / (2.0 * step)
        d_physical = np.trace(d_central) * np.eye(3) - d_central
        derivatives.append({
            "volume": float((a["volume"] - b["volume"]) / (2.0 * step)),
            "first_moment": ((a["first_moment"] - b["first_moment"]) / (2.0 * step)).tolist(),
            "com": ((a["com"] - b["com"]) / (2.0 * step)).tolist(),
            "raw_second_moment": ((a["raw_second_moment"] - b["raw_second_moment"])
                                  / (2.0 * step)).tolist(),
            "central_inertia": d_central.tolist(),
            "physical_inertia": d_physical.tolist(),
        })
    return derivatives


def _central_bound(raw_bound, first_bound, volume_bound, first, volume):
    centre = np.asarray(first, dtype=np.float64) / volume
    com_bound = np.asarray(first_bound, dtype=np.float64)
    return (np.asarray(raw_bound, dtype=np.float64)
            + volume_bound * np.outer(np.abs(centre), np.abs(centre))
            + volume * (np.einsum("pi,j->pij", com_bound, np.abs(centre))
                        + np.einsum("i,pj->pij", np.abs(centre), com_bound)))


def _max_abs(value):
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def _pack_surface(surface):
    return {
        "d_volume": np.asarray(surface["d_volume"], dtype=np.float64).tolist(),
        "d_first_moment": np.asarray(surface["d_first_moment"], dtype=np.float64).tolist(),
        "d_raw_second_moment": np.asarray(surface["d_raw_second_moment"], dtype=np.float64).tolist(),
        "d_com": (None if surface["d_com"] is None
                  else np.asarray(surface["d_com"], dtype=np.float64).tolist()),
        "d_central_inertia": (None if surface["d_central_inertia"] is None
                              else np.asarray(surface["d_central_inertia"], dtype=np.float64).tolist()),
        "d_physical_inertia": (None if surface["d_physical_inertia"] is None
                               else np.asarray(surface["d_physical_inertia"], dtype=np.float64).tolist()),
        "quadrature_abs": {key: np.asarray(value, dtype=np.float64).tolist()
                           for key, value in surface["quadrature_abs"].items()},
        "sdf_location_abs": {key: np.asarray(value, dtype=np.float64).tolist()
                             for key, value in surface["sdf_location_abs"].items()},
        "total_abs": {key: np.asarray(value, dtype=np.float64).tolist()
                      for key, value in surface["total_abs"].items()},
        "d_com_bound": (None if surface["d_com_bound"] is None
                        else np.asarray(surface["d_com_bound"], dtype=np.float64).tolist()),
        "d_central_inertia_bound": (
            None if surface["d_central_inertia_bound"] is None
            else np.asarray(surface["d_central_inertia_bound"], dtype=np.float64).tolist()),
        "d_physical_inertia_bound": (
            None if surface["d_physical_inertia_bound"] is None
            else np.asarray(surface["d_physical_inertia_bound"], dtype=np.float64).tolist()),
        "surface_area": float(surface["surface_area"]),
        "valid_area": float(surface["valid_area"]),
        "excluded_area": float(surface["excluded_area"]),
        "branch_coverage": float(surface["branch_coverage"]),
        "gradient_coverage": float(surface["gradient_coverage"]),
        "branch_excluded_area": float(surface["branch_excluded_area"]),
        "gradient_excluded_area": float(surface["gradient_excluded_area"]),
        "coverage": float(surface["coverage"]),
        "triangles": int(surface["triangles"]),
        "triangles_valid": int(surface["triangles_valid"]),
        "pitch_m": float(surface["pitch_m"]),
        "sdf_band_m": float(surface["sdf_band_m"]),
        "gradient_floor": float(surface["gradient_floor"]),
    }


def compare_surface_reference(surface, reference, base=None, index=0):
    del base
    index = int(index)
    estimates = {
        "volume": np.asarray(surface["d_volume"], dtype=np.float64)[index],
        "first_moment": np.asarray(surface["d_first_moment"], dtype=np.float64)[index],
        "com": np.asarray(surface["d_com"], dtype=np.float64)[index],
        "raw_second_moment": np.asarray(surface["d_raw_second_moment"], dtype=np.float64)[index],
        "central_inertia": np.asarray(surface["d_central_inertia"], dtype=np.float64)[index],
        "physical_inertia": np.asarray(surface["d_physical_inertia"], dtype=np.float64)[index],
    }
    references = {
        "volume": np.asarray(reference["volume"], dtype=np.float64),
        "first_moment": np.asarray(reference["first_moment"], dtype=np.float64),
        "com": np.asarray(reference["com"], dtype=np.float64),
        "raw_second_moment": np.asarray(reference["raw_second_moment"], dtype=np.float64),
        "central_inertia": np.asarray(reference["central_inertia"], dtype=np.float64),
        "physical_inertia": np.asarray(reference["physical_inertia"], dtype=np.float64),
    }
    surface_bounds = {
        "volume": np.asarray(surface["total_abs"]["volume"], dtype=np.float64)[index],
        "first_moment": np.asarray(surface["total_abs"]["first_moment"], dtype=np.float64)[index],
        "com": np.asarray(surface["d_com_bound"], dtype=np.float64)[index],
        "raw_second_moment": np.asarray(surface["total_abs"]["raw_second_moment"], dtype=np.float64)[index],
        "central_inertia": np.asarray(surface["d_central_inertia_bound"], dtype=np.float64)[index],
        "physical_inertia": np.asarray(surface["d_physical_inertia_bound"], dtype=np.float64)[index],
    }
    reference_bounds = reference.get("combined_reference_bound", {})
    combined_bounds = {}
    absolute = {}
    relative = {}
    for key in estimates:
        error = np.abs(estimates[key] - references[key])
        absolute[key] = error.tolist() if np.ndim(error) else float(error)
        scale = max(_max_abs(references[key]), 1.0e-18)
        relative[key] = (error / scale).tolist() if np.ndim(error) else float(error / scale)
        ref_bound = np.asarray(reference_bounds.get(key, 0.0), dtype=np.float64)
        surface_bound = np.asarray(surface_bounds[key], dtype=np.float64)
        combined_bounds[key] = ((surface_bound + ref_bound).tolist()
                                if np.ndim(surface_bound) else float(surface_bound + ref_bound))
    contains = all(bool(np.all(np.asarray(absolute[key], dtype=np.float64)
                              <= np.asarray(combined_bounds[key], dtype=np.float64)))
                   for key in estimates)
    return {"absolute": absolute, "relative": relative,
            "combined_bound": combined_bounds, "bound_contains_fd": contains}


def _deterministic_rotations(count):
    result = []
    for i in range(count):
        a = 0.17 + 0.013 * i
        b = -0.11 + 0.007 * i
        c = 0.23 - 0.005 * i
        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)
        cc, sc = math.cos(c), math.sin(c)
        rx = np.asarray([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
        ry = np.asarray([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
        rz = np.asarray([[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]])
        result.append(rz @ ry @ rx)
    return result


def physical_inertia(central_second_moment):
    central_second_moment = np.asarray(central_second_moment, dtype=np.float64)
    return np.trace(central_second_moment) * np.eye(3) - central_second_moment


def second_moment_from_inertia(inertia):
    inertia = np.asarray(inertia, dtype=np.float64)
    return 0.5 * np.trace(inertia) * np.eye(3) - inertia


def pose_inertia(central_inertia, volume, centre, rotation, translation):
    centre = np.asarray(centre, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    rotated_centre = rotation @ centre + translation
    rotated = rotation @ np.asarray(central_inertia, dtype=np.float64) @ rotation.T
    parallel = float(volume) * (np.dot(rotated_centre, rotated_centre) * np.eye(3)
                                - np.outer(rotated_centre, rotated_centre))
    return {"centre": rotated_centre, "central_inertia": rotated,
            "origin_inertia": rotated + parallel}


def pose_check(central_inertia, volume, centre, rotations, translations):
    centre = np.asarray(centre, dtype=np.float64)
    central = np.asarray(central_inertia, dtype=np.float64)
    central_second = second_moment_from_inertia(central)
    raw_body = central_second + float(volume) * np.outer(centre, centre)
    direct = []
    for rotation, translation in zip(rotations, translations):
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)
        result = pose_inertia(central, volume, centre, rotation, translation)
        rotated_body_centre = rotation @ centre
        moved_centre = rotated_body_centre + translation
        moved_raw = rotation @ raw_body @ rotation.T
        moved_raw += float(volume) * (np.outer(rotated_body_centre, translation)
                                      + np.outer(translation, rotated_body_centre))
        moved_raw += float(volume) * np.outer(translation, translation)
        direct_origin = np.trace(moved_raw) * np.eye(3) - moved_raw
        rotated_raw = rotation @ raw_body @ rotation.T
        central_raw = rotated_raw - float(volume) * np.outer(rotated_body_centre,
                                                              rotated_body_centre)
        direct_central = np.trace(central_raw) * np.eye(3) - central_raw
        direct.append({
            "central": _max_abs(result["central_inertia"] - direct_central),
            "origin": _max_abs(result["origin_inertia"] - direct_origin),
        })
    return {
        "max_central_residual": float(max(row["central"] for row in direct)),
        "max_origin_residual": float(max(row["origin"] for row in direct)),
    }


def measure_pose_cost(central_inertia, volume, centre, count=20000):
    rotations = _deterministic_rotations(min(count, 1024))
    translations = [np.asarray([0.013 * (i % 17), -0.009 * (i % 13), 0.021 * (i % 19)])
                    for i in range(count)]
    for i in range(min(100, count)):
        pose_inertia(central_inertia, volume, centre, rotations[i % len(rotations)], translations[i])
    samples = []
    start_all = time.perf_counter_ns()
    for i in range(count):
        start = time.perf_counter_ns()
        pose_inertia(central_inertia, volume, centre, rotations[i % len(rotations)], translations[i])
        samples.append(time.perf_counter_ns() - start)
    total = time.perf_counter_ns() - start_all
    return {
        "poses": int(count),
        "median_ns": float(np.median(samples)),
        "maximum_ns": int(np.max(samples)),
        "total_ns": int(total),
        "mean_ns": float(np.mean(samples)),
    }


def surface_reference_experiment(shape, theta, lo, hi, surface_n, voxel_n, coarse_n, step,
                                 side=1, sdf_band_pitch=1.5):
    theta = np.asarray(theta, dtype=np.float64)
    base = linear_tetrahedral_properties(shape, theta, lo, hi, voxel_n, side=side)
    binary_base = voxel_properties(shape, theta, lo, hi, voxel_n, side=side)
    surface, _ = surface_mass_derivatives(shape, theta, lo, hi, surface_n, side=side,
                                           sdf_band_pitch=sdf_band_pitch,
                                           base_properties=base)
    fine = linear_tetrahedral_central_fd(shape, theta, lo, hi, voxel_n, step, side=side)
    coarse = linear_tetrahedral_central_fd(shape, theta, lo, hi, coarse_n, step, side=side)
    packed_surface = _pack_surface(surface)
    reference = []
    for fine_row, coarse_row in zip(fine, coarse):
        ref = dict(fine_row)
        delta = {}
        bound = {}
        for key in ("volume", "first_moment", "com", "raw_second_moment",
                    "central_inertia", "physical_inertia"):
            difference = np.abs(np.asarray(fine_row[key], dtype=np.float64)
                                - np.asarray(coarse_row[key], dtype=np.float64))
            delta[key] = difference
            bound[key] = 4.0 * difference
        ref["resolution_delta"] = {key: value.tolist() for key, value in delta.items()}
        ref["combined_reference_bound"] = {key: value.tolist() for key, value in bound.items()}
        reference.append(ref)
    comparisons = [compare_surface_reference(surface, ref, base, index=j)
                   for j, ref in enumerate(reference)]
    return {
        "shape": shape,
        "theta": theta.tolist(),
        "surface": packed_surface,
        "reference_method": "linear_tetrahedral_voxel",
        "voxel_base": _pack_properties(base),
        "binary_voxel_base": _pack_properties(binary_base),
        "reference": reference,
        "comparison": comparisons,
    }
