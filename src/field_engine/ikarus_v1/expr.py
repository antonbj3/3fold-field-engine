#!/usr/bin/env python3
"""SDF expression system: a part is an expression tree, not a mesh.

Leaves are either analytic primitives (parameters stored exactly and symbolically, never baked to a mesh)
or grid fields (voxel occupancy/SDF maps in .npz form, sampled trilinearly). Interior nodes are boolean
ops (union/intersect/subtract, hard or smooth), an affine transform (exact 4x4), or offset/scale (wall
thickness is an offset node, not a remesh). The expression is the file: JSON round-trips it losslessly and
no mesh exists until an explicit export or marching-cubes call.

Sign convention: SDF < 0 inside, > 0 outside, 0 on the surface (Hart/Bloomenthal), so union = min,
intersect = max, subtract(A,B) = max(A,-B); every boolean is two scalar ops per node, which is what
eval_warp.py compiles.

Each constructor returns a plain dict node. eval_sdf_py(node, p) is the pure-python reference evaluator
used as ground truth for the generated kernel; save/load wrap a versioned envelope.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------------------------- transforms
def mat4_identity():
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def mat4_translate(tx, ty, tz):
    m = mat4_identity()
    m[0][3], m[1][3], m[2][3] = tx, ty, tz
    return m


def mat4_rot_axis(axis, deg):
    """Exact rotation matrix about a coordinate axis ('x'|'y'|'z') by deg degrees -- kept symbolic
    (axis+angle), NOT baked into a generic 4x4 until serialize time, so round-trip stays exact in the
    stored parameters (angle in degrees, not a lossy matrix)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = mat4_identity()
    if axis == "x":
        m[1][1], m[1][2], m[2][1], m[2][2] = c, -s, s, c
    elif axis == "y":
        m[0][0], m[0][2], m[2][0], m[2][2] = c, s, -s, c
    elif axis == "z":
        m[0][0], m[0][1], m[1][0], m[1][1] = c, -s, s, c
    else:
        raise ValueError(f"axis must be x/y/z, got {axis!r}")
    return m


def mat4_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def mat4_inverse_affine(m):
    """Inverse of a rigid/affine transform (rotation+translation, no shear needed here): invert the
    3x3 block and recompute the translation -- exact for the affine transforms this module builds
    (translate/rotate/compose), which is all eval_warp.py needs (world point -> local point)."""
    R = [[m[i][j] for j in range(3)] for i in range(3)]
    t = [m[i][3] for i in range(3)]
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
           - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    if abs(det) < 1e-15:
        raise ValueError("singular transform, cannot invert")
    inv = [[0.0] * 3 for _ in range(3)]
    cof = lambda r, c: (R[(r + 1) % 3][(c + 1) % 3] * R[(r + 2) % 3][(c + 2) % 3]
                         - R[(r + 1) % 3][(c + 2) % 3] * R[(r + 2) % 3][(c + 1) % 3])
    for i in range(3):
        for j in range(3):
            inv[j][i] = cof(i, j) / det
    itx = -(inv[0][0] * t[0] + inv[0][1] * t[1] + inv[0][2] * t[2])
    ity = -(inv[1][0] * t[0] + inv[1][1] * t[1] + inv[1][2] * t[2])
    itz = -(inv[2][0] * t[0] + inv[2][1] * t[1] + inv[2][2] * t[2])
    out = mat4_identity()
    for i in range(3):
        for j in range(3):
            out[i][j] = inv[i][j]
    out[0][3], out[1][3], out[2][3] = itx, ity, itz
    return out


# --------------------------------------------------------------------------------------------- node types
# Every node is a plain dict {"op": <str>, ...params}. This IS the serialization format (no separate
# to-dict step needed) -- json.dumps(node) is the file. Kept as plain dicts (not classes with __dict__
# quirks) so JSON round-trip is trivially lossless and eval_warp.py's flattening pass can walk it
# uniformly regardless of node type.

def halfspace(normal, offset):
    """Plane primitive: signed distance = dot(p, normal) - offset (normal must be unit; offset in mm
    along normal from origin). Inside = negative side."""
    nx, ny, nz = normal
    nrm = math.sqrt(nx * nx + ny * ny + nz * nz)
    assert nrm > 1e-12
    return {"op": "halfspace", "normal": [nx / nrm, ny / nrm, nz / nrm], "offset": float(offset)}


def sphere(radius, center=(0.0, 0.0, 0.0)):
    return {"op": "sphere", "radius": float(radius), "center": list(map(float, center))}


def box(half_extents, center=(0.0, 0.0, 0.0)):
    """Axis-aligned box, EXACT (not smoothed-corner) SDF: standard Chamfer/Bloomenthal box distance."""
    return {"op": "box", "half_extents": list(map(float, half_extents)), "center": list(map(float, center))}


def cylinder(radius, height, axis="z", center=(0.0, 0.0, 0.0)):
    """Finite cylinder, axis-aligned along `axis` through `center`, half-height = height/2 (so `height`
    is the full extent along axis -- matches build123d.Cylinder's `height` convention for the round-trip
    proof in proofs.py)."""
    return {"op": "cylinder", "radius": float(radius), "height": float(height), "axis": axis,
            "center": list(map(float, center))}


def cone(radius1, radius2, height, axis="z", center=(0.0, 0.0, 0.0)):
    """Truncated cone (frustum), radius1 at -height/2, radius2 at +height/2 along `axis`."""
    return {"op": "cone", "radius1": float(radius1), "radius2": float(radius2), "height": float(height),
            "axis": axis, "center": list(map(float, center))}


def rect_frustum(z0, z1, hx0, hy0, hx1, hy1, cx0=0.0, cy0=0.0, cx1=0.0, cy1=0.0):
    """Tapered rectangular frustum between two AXIS-ALIGNED rectangular profiles at z0/z1 (local frame,
    caller wraps in transform() to place/orient it): an exact zero set for an affinely related
    rect-to-rect loft segment, built as the intersection of nine half-spaces. Half-extents hx(z)/hy(z)
    interpolate linearly between (hx0,hy0) at z0 and (hx1,hy1) at z1; the centre (cx(z),cy(z))
    interpolates the same way (default 0, i.e. profiles centred on the sketch-plane origin). Each of the
    6 bounding
    faces (2 flat z-caps + 4 tilted sides) is a TRUE PLANE (a straight ruled line between two parallel
    profile edges is planar by construction) so max() of the 6 per-face SIGNED PLANE DISTANCES is an
    EXACT zero-set, Lipschitz-1 underestimate near edges only (same convention box() above documents --
    this is a stacked-consecutive-loft's atomic segment; a 3-profile loft = union() of 2 rect_frustum
    segments, each valid only within its own z-slab thanks to the z-cap halfspaces)."""
    assert z1 > z0, "rect_frustum requires z1 > z0"
    return {"op": "rect_frustum", "z0": float(z0), "z1": float(z1),
            "hx0": float(hx0), "hy0": float(hy0), "hx1": float(hx1), "hy1": float(hy1),
            "cx0": float(cx0), "cy0": float(cy0), "cx1": float(cx1), "cy1": float(cy1)}


def gridfield(npz_path, key="occupancy", mode="occupancy", iso=0.5, pitch_override=None):
    """Leaf that wraps an already-grown voxel grid (existing .npz occupancy/SDF maps) as a trilinearly-
    sampled SDF primitive -- NO re-derivation of the field, only a sampling contract on top of it.
    mode="occupancy": binary/float occupancy in [0,1], iso-value `iso` is treated as the zero-crossing
      (value - iso, so >iso reads OUTSIDE per this module's sign convention: SDF<0=inside means we store
      (iso - value) so occupancy=1 (fully inside source) -> negative -> inside). mode="sdf": array already
      stores a true signed distance in mm, used as-is.
    Requires keys `global_origin_mm` (3,) and `pitch_mm` (scalar) in the npz."""
    return {"op": "gridfield", "npz_path": npz_path, "key": key, "mode": mode, "iso": float(iso),
            "pitch_override": pitch_override}


def transform(child, mat4):
    return {"op": "transform", "child": child, "mat4": mat4}


def translate(child, tx, ty, tz):
    return transform(child, mat4_translate(tx, ty, tz))


def rotate(child, axis, deg, about=(0.0, 0.0, 0.0)):
    ax, ay, az = about
    m = mat4_mul(mat4_translate(ax, ay, az), mat4_mul(mat4_rot_axis(axis, deg), mat4_translate(-ax, -ay, -az)))
    return transform(child, m)


def axis_project(child, axis, ref):
    """Freezes ONE world coordinate (axis in {0:x,1:y,2:z}) to a fixed value `ref` before sampling
    `child` -- turns a solid that has a UNIFORM cross-section along `axis` (extrude/box/cylinder-shaped,
    built by this translator's own `extrude` op) into an infinite prism of that same cross-section,
    sampled at the interior reference plane p[axis]=ref. EXACT for any child whose SDF genuinely does not
    depend on p[axis] within the region being queried (true for box/cylinder/extrude-composited trees
    that have not been tapered) -- added for shell's open-face construction (sdf_shell_v1): the "cap"
    region removed by an open shell's `faces_to_remove` face is exactly
    intersect(axis_project(base, axis, ref), <slab at the removed face>), because sampling the base solid
    at an interior z gives the same footprint as the removed cap would have had, letting subtract() carve
    the opening. NOT exact for a tapered child (loft/rect_frustum) -- declared, not silently used, by the
    caller (recept_till_ikarus_v1.py only invokes this when build_meta proves the base is untapered along
    the requested axis)."""
    assert axis in (0, 1, 2), f"axis_project: axis must be 0/1/2 (x/y/z), got {axis!r}"
    return {"op": "axis_project", "child": child, "axis": int(axis), "ref": float(ref)}


def offset(child, delta):
    """Isotropic offset (wall thickness / dilation): SDF - delta grows the solid by delta (positive
    delta = dilate outward, negative = erode). Exact for exact-SDF primitives; approximate (Lipschitz-1
    per-sample) once a gridfield/smooth-union child is in the subtree -- eval_warp.py documents the bound."""
    return {"op": "offset", "child": child, "delta": float(delta)}


def scale(child, s):
    """Uniform scale about the origin: composes with a coordinate-space transform (1/s on the sampled
    point) + s on the returned distance so the result stays a true SDF (Lipschitz-1)."""
    return {"op": "scale", "child": child, "s": float(s)}


def union(a, b, k=0.0):
    """k=0: hard union (min). k>0: smooth union (mm-scale blend radius, Quilez polynomial smooth-min)."""
    return {"op": "union", "a": a, "b": b, "k": float(k)}


def intersect(a, b, k=0.0):
    return {"op": "intersect", "a": a, "b": b, "k": float(k)}


def subtract(a, b, k=0.0):
    """A minus B = intersect(A, complement(B)) = max(A, -B) (hard) or the smooth analogue."""
    return {"op": "subtract", "a": a, "b": b, "k": float(k)}


# --------------------------------------------------------------------------------------------- serialize
IKARUS_VERSION = "1.0"   # bumped whenever the node schema (op set / field names) changes; the
                          # sphere-tracing renderer consumes files by this tag rather than guessing.


def save(node, path):
    """Wraps the tree in a versioned envelope {"ikarus_version": ..., "tree": ...} -- load() also accepts
    a bare (unversioned) node dict for backward-compat with files written before this envelope existed."""
    with open(path, "w") as f:
        json.dump({"ikarus_version": IKARUS_VERSION, "tree": node}, f, indent=1)
    return path


def load(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "tree" in data and "ikarus_version" in data:
        return data["tree"]
    return data   # bare node (pre-versioning file, or a node passed straight through)


def node_count(node):
    """Count nodes in the tree (needed by queries.py's cost-dispatch: expr size is one of the two axes,
    alongside query-point count, that decide the evaluation method)."""
    if not isinstance(node, dict):
        return 0
    n = 1
    for k in ("child", "a", "b"):
        if k in node:
            n += node_count(node[k])
    return n


def leaf_ops(node):
    """Set of leaf-op types present in the tree (queries.py logs this: a tree with a gridfield leaf can't
    use the exact-primitive precision path)."""
    out = set()

    def walk(n):
        op = n["op"]
        if op in ("halfspace", "sphere", "box", "cylinder", "cone", "rect_frustum", "gridfield"):
            out.add(op)
        for k in ("child", "a", "b"):
            if k in n:
                walk(n[k])
    walk(node)
    return out


# --------------------------------------------------------------------------------------------- pure-python reference eval (ground truth for the Warp kernel's validation gate)
def eval_sdf_py(node, p):
    """Reference (slow, pure-Python, no Warp) SDF evaluator -- the independent ground truth eval_warp.py's
    validation gate cross-checks against (discipline #14: machine cross-check vs a truth you did NOT use
    to build the thing being checked). p = (x, y, z) tuple, mm."""
    op = node["op"]
    if op == "halfspace":
        nx, ny, nz = node["normal"]
        return p[0] * nx + p[1] * ny + p[2] * nz - node["offset"]
    if op == "sphere":
        cx, cy, cz = node["center"]
        d = math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2)
        return d - node["radius"]
    if op == "box":
        cx, cy, cz = node["center"]
        hx, hy, hz = node["half_extents"]
        qx, qy, qz = abs(p[0] - cx) - hx, abs(p[1] - cy) - hy, abs(p[2] - cz) - hz
        outside = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2)
        inside = min(max(qx, max(qy, qz)), 0.0)
        return outside + inside
    if op == "cylinder":
        cx, cy, cz = node["center"]
        axis = node["axis"]
        lx, ly, lz = p[0] - cx, p[1] - cy, p[2] - cz
        if axis == "z":
            rad_v, ax_v = math.sqrt(lx * lx + ly * ly), lz
        elif axis == "y":
            rad_v, ax_v = math.sqrt(lx * lx + lz * lz), ly
        else:
            rad_v, ax_v = math.sqrt(ly * ly + lz * lz), lx
        qx, qy = rad_v - node["radius"], abs(ax_v) - node["height"] / 2.0
        outside = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2)
        inside = min(max(qx, qy), 0.0)
        return outside + inside
    if op == "cone":
        cx, cy, cz = node["center"]
        axis = node["axis"]
        lx, ly, lz = p[0] - cx, p[1] - cy, p[2] - cz
        if axis == "z":
            rad_v, ax_v = math.sqrt(lx * lx + ly * ly), lz
        elif axis == "y":
            rad_v, ax_v = math.sqrt(lx * lx + lz * lz), ly
        else:
            rad_v, ax_v = math.sqrt(ly * ly + lz * lz), lx
        h = node["height"]
        t = (ax_v + h / 2.0) / h
        t = min(max(t, 0.0), 1.0)
        r_at = node["radius1"] + (node["radius2"] - node["radius1"]) * t
        # approximate frustum SDF (radial-minus-linear-interp, correct on-axis + exact at caps; adequate
        # for this module's use as a housing/relief primitive, not claimed exact off-axis near the cap edges)
        qy = abs(ax_v) - h / 2.0
        qx = rad_v - r_at
        outside = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2)
        inside = min(max(qx, qy), 0.0)
        return outside + inside
    if op == "rect_frustum":
        z0, z1 = node["z0"], node["z1"]
        t = (p[2] - z0) / (z1 - z0)
        hx = node["hx0"] + (node["hx1"] - node["hx0"]) * t
        hy = node["hy0"] + (node["hy1"] - node["hy0"]) * t
        cx = node["cx0"] + (node["cx1"] - node["cx0"]) * t
        cy = node["cy0"] + (node["cy1"] - node["cy0"]) * t
        kx = (node["hx1"] - node["hx0"]) / (z1 - z0)
        ky = (node["hy1"] - node["hy0"]) / (z1 - z0)
        kcx = (node["cx1"] - node["cx0"]) / (z1 - z0)
        kcy = (node["cy1"] - node["cy0"]) / (z1 - z0)
        # 6 true-plane signed distances (each already exact Euclidean plane distance after /norm);
        # max() = exact zero-set intersection (see rect_frustum()'s docstring above).
        d_zlo = z0 - p[2]
        d_zhi = p[2] - z1
        # +x face: x - (kx+kcx)*z - (hx0 - (kx+kcx)*z0) <= 0 ... expand with center drift kcx too
        def _plane(pv, nx, nz, c):
            n = math.sqrt(nx * nx + nz * nz)
            return (nx * pv + nz * p[2] - c) / n
        d_xhi = _plane(p[0], 1.0, -(kx + kcx), node["hx0"] + node["cx0"] - (kx + kcx) * z0)
        d_xlo = _plane(p[0], -1.0, -(kx - kcx), node["hx0"] - node["cx0"] - (kx - kcx) * z0)
        d_yhi = _plane(p[1], 1.0, -(ky + kcy), node["hy0"] + node["cy0"] - (ky + kcy) * z0)
        d_ylo = _plane(p[1], -1.0, -(ky - kcy), node["hy0"] - node["cy0"] - (ky - kcy) * z0)
        return max(d_zlo, d_zhi, d_xhi, d_xlo, d_yhi, d_ylo)
    if op == "gridfield":
        raise NotImplementedError("gridfield leaf sampling is Warp-side only (see eval_warp.py); no "
                                   "pure-python path is provided (would require re-implementing trilinear "
                                   "sampling twice) -- the validation gate for gridfield trees instead "
                                   "checks trilinear-sample agreement at a handful of grid-aligned points "
                                   "directly against the raw npz array (exact, no interpolation needed there).")
    if op == "transform":
        inv = mat4_inverse_affine(node["mat4"])
        x, y, z, w = p[0], p[1], p[2], 1.0
        lp = (inv[0][0] * x + inv[0][1] * y + inv[0][2] * z + inv[0][3],
              inv[1][0] * x + inv[1][1] * y + inv[1][2] * z + inv[1][3],
              inv[2][0] * x + inv[2][1] * y + inv[2][2] * z + inv[2][3])
        return eval_sdf_py(node["child"], lp)
    if op == "axis_project":
        lp = list(p)
        lp[node["axis"]] = node["ref"]
        return eval_sdf_py(node["child"], tuple(lp))
    if op == "offset":
        return eval_sdf_py(node["child"], p) - node["delta"]
    if op == "scale":
        s = node["s"]
        lp = (p[0] / s, p[1] / s, p[2] / s)
        return eval_sdf_py(node["child"], lp) * s
    if op == "union":
        da, db = eval_sdf_py(node["a"], p), eval_sdf_py(node["b"], p)
        k = node["k"]
        if k <= 0.0:
            return min(da, db)
        h = max(k - abs(da - db), 0.0) / k
        return min(da, db) - h * h * k * 0.25
    if op == "intersect":
        da, db = eval_sdf_py(node["a"], p), eval_sdf_py(node["b"], p)
        k = node["k"]
        if k <= 0.0:
            return max(da, db)
        h = max(k - abs(da - db), 0.0) / k
        return max(da, db) + h * h * k * 0.25
    if op == "subtract":
        da, db = eval_sdf_py(node["a"], p), eval_sdf_py(node["b"], p)
        k = node["k"]
        if k <= 0.0:
            return max(da, -db)
        h = max(k - abs(da + db), 0.0) / k
        return max(da, -db) + h * h * k * 0.25
    raise ValueError(f"unknown op {op!r}")


if __name__ == "__main__":
    # smoke self-test: build a tiny tree, round-trip through JSON, confirm identical eval
    c = cylinder(radius=12.5, height=40.0)
    b = box(half_extents=(8.0, 8.0, 8.0), center=(0.0, 0.0, 15.0))
    tree = subtract(c, translate(b, 0.0, 0.0, 0.0))
    p = (3.0, 1.0, 10.0)
    v0 = eval_sdf_py(tree, p)
    js = json.dumps(tree)
    tree2 = json.loads(js)
    v1 = eval_sdf_py(tree2, p)
    assert v0 == v1, (v0, v1)
    print("expr.py self-test OK:", v0, "node_count", node_count(tree), "leaf_ops", leaf_ops(tree))
