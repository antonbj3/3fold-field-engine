#!/usr/bin/env python3
"""Warp evaluator for expr.py SDF trees: codegen one kernel per tree, evaluate point batches.

Method: codegen, not a switch-loop interpreter. An interpreter kernel needs either a fixed-depth stack in
per-thread registers with a runtime dispatch switch per node (branch divergence plus worst-case register
pressure paid by every thread), or recursion, which wp.func does not support arbitrarily. Codegen emits
one wp.func per node, bottom-up, with primitive parameters baked in as compile-time literals; a new kernel
per distinct tree, cached by structural hash. That trades a recompile on tree change for zero interpreter
overhead at launch.

Gridfield leaves are packed into one flat float32 array (all leaves concatenated) plus per-leaf metadata
arrays, so the generated kernel has a fixed small signature regardless of leaf count.

The 2.5D band tree does not go through literal baking: the plan (zs/z0/z1 per band plus axis) is passed as
kernel arguments, so one kernel serves every plan of every length (see BandTreeEvaluator below).

Entry points: eval_batch(node, points_mm) -> (N,) distances in mm; gradient_batch(node, points_mm) ->
(N,3) central-difference normals; selftest() cross-checks the generated kernel against expr.eval_sdf_py.
Run `python eval_warp.py selftest`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expr as E

wp.init()
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

_KERNEL_CACHE = {}   # structural-hash -> compiled (module_dict, kernel, n_grid_leaves)


# --------------------------------------------------------------------------------------------- shared grid-sample func (used by every codegen'd gridfield leaf)
_GRID_SAMPLE_SRC = '''
@wp.func
def _sample_grid_leaf(px: float, py: float, pz: float, leaf_id: int,
                       grid_data: wp.array(dtype=wp.float32),
                       grid_offset: wp.array(dtype=wp.int32),
                       grid_nx: wp.array(dtype=wp.int32), grid_ny: wp.array(dtype=wp.int32), grid_nz: wp.array(dtype=wp.int32),
                       grid_ox: wp.array(dtype=wp.float32), grid_oy: wp.array(dtype=wp.float32), grid_oz: wp.array(dtype=wp.float32),
                       grid_pitch: wp.array(dtype=wp.float32), grid_iso: wp.array(dtype=wp.float32),
                       grid_is_sdf: wp.array(dtype=wp.int32)):
    off = grid_offset[leaf_id]
    nx = grid_nx[leaf_id]
    ny = grid_ny[leaf_id]
    nz = grid_nz[leaf_id]
    pitch = grid_pitch[leaf_id]
    fx = (px - grid_ox[leaf_id]) / pitch
    fy = (py - grid_oy[leaf_id]) / pitch
    fz = (pz - grid_oz[leaf_id]) / pitch
    OUTSIDE = float(1.0e6)
    if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx > float(nx - 1) or fy > float(ny - 1) or fz > float(nz - 1):
        return OUTSIDE
    ix = int(wp.floor(fx)); iy = int(wp.floor(fy)); iz = int(wp.floor(fz))
    ix1 = wp.min(ix + 1, nx - 1); iy1 = wp.min(iy + 1, ny - 1); iz1 = wp.min(iz + 1, nz - 1)
    tx = fx - float(ix); ty = fy - float(iy); tz = fz - float(iz)

    def idx(i: int, j: int, k: int, nx2: int, ny2: int, nz2: int, off2: int):
        return off2 + (i * ny2 + j) * nz2 + k
    c000 = grid_data[idx(ix, iy, iz, nx, ny, nz, off)]
    c100 = grid_data[idx(ix1, iy, iz, nx, ny, nz, off)]
    c010 = grid_data[idx(ix, iy1, iz, nx, ny, nz, off)]
    c110 = grid_data[idx(ix1, iy1, iz, nx, ny, nz, off)]
    c001 = grid_data[idx(ix, iy, iz1, nx, ny, nz, off)]
    c101 = grid_data[idx(ix1, iy, iz1, nx, ny, nz, off)]
    c011 = grid_data[idx(ix, iy1, iz1, nx, ny, nz, off)]
    c111 = grid_data[idx(ix1, iy1, iz1, nx, ny, nz, off)]
    c00 = wp.lerp(c000, c100, tx); c10 = wp.lerp(c010, c110, tx)
    c01 = wp.lerp(c001, c101, tx); c11 = wp.lerp(c011, c111, tx)
    c0 = wp.lerp(c00, c10, ty); c1 = wp.lerp(c01, c11, ty)
    val = wp.lerp(c0, c1, tz)
    if grid_is_sdf[leaf_id] == 1:
        return val
    else:
        return (grid_iso[leaf_id] - val) * pitch
'''
# NOTE: nested wp.func-local `idx` above is illustrative pseudo-code; Warp does not support nested
# function defs inside a @wp.func body, so the ACTUAL emitted source (below, _grid_sample_real_src)
# inlines the index arithmetic at each of the 8 corners instead. Kept the annotated version above in the
# module for readability/documentation of intent; the real codegen uses the inlined form.
_GRID_SAMPLE_SRC_REAL = '''
@wp.func
def _sample_grid_leaf(px: float, py: float, pz: float, leaf_id: int,
                       grid_data: wp.array(dtype=wp.float32),
                       grid_offset: wp.array(dtype=wp.int32),
                       grid_nx: wp.array(dtype=wp.int32), grid_ny: wp.array(dtype=wp.int32), grid_nz: wp.array(dtype=wp.int32),
                       grid_ox: wp.array(dtype=wp.float32), grid_oy: wp.array(dtype=wp.float32), grid_oz: wp.array(dtype=wp.float32),
                       grid_pitch: wp.array(dtype=wp.float32), grid_iso: wp.array(dtype=wp.float32),
                       grid_is_sdf: wp.array(dtype=wp.int32)):
    off = grid_offset[leaf_id]
    nx = grid_nx[leaf_id]
    ny = grid_ny[leaf_id]
    nz = grid_nz[leaf_id]
    pitch = grid_pitch[leaf_id]
    fx = (px - grid_ox[leaf_id]) / pitch
    fy = (py - grid_oy[leaf_id]) / pitch
    fz = (pz - grid_oz[leaf_id]) / pitch
    OUTSIDE = float(1.0e6)
    if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx > float(nx - 1) or fy > float(ny - 1) or fz > float(nz - 1):
        return OUTSIDE
    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    iz = int(wp.floor(fz))
    ix1 = wp.min(ix + 1, nx - 1)
    iy1 = wp.min(iy + 1, ny - 1)
    iz1 = wp.min(iz + 1, nz - 1)
    tx = fx - float(ix)
    ty = fy - float(iy)
    tz = fz - float(iz)
    i000 = off + (ix * ny + iy) * nz + iz
    i100 = off + (ix1 * ny + iy) * nz + iz
    i010 = off + (ix * ny + iy1) * nz + iz
    i110 = off + (ix1 * ny + iy1) * nz + iz
    i001 = off + (ix * ny + iy) * nz + iz1
    i101 = off + (ix1 * ny + iy) * nz + iz1
    i011 = off + (ix * ny + iy1) * nz + iz1
    i111 = off + (ix1 * ny + iy1) * nz + iz1
    c000 = grid_data[i000]
    c100 = grid_data[i100]
    c010 = grid_data[i010]
    c110 = grid_data[i110]
    c001 = grid_data[i001]
    c101 = grid_data[i101]
    c011 = grid_data[i011]
    c111 = grid_data[i111]
    c00 = wp.lerp(c000, c100, tx)
    c10 = wp.lerp(c010, c110, tx)
    c01 = wp.lerp(c001, c101, tx)
    c11 = wp.lerp(c011, c111, tx)
    c0 = wp.lerp(c00, c10, ty)
    c1 = wp.lerp(c01, c11, ty)
    val = wp.lerp(c0, c1, tz)
    if grid_is_sdf[leaf_id] == 1:
        return val
    else:
        return (grid_iso[leaf_id] - val) * pitch
'''

_GRID_SIG = ("grid_data: wp.array(dtype=wp.float32), grid_offset: wp.array(dtype=wp.int32), "
             "grid_nx: wp.array(dtype=wp.int32), grid_ny: wp.array(dtype=wp.int32), grid_nz: wp.array(dtype=wp.int32), "
             "grid_ox: wp.array(dtype=wp.float32), grid_oy: wp.array(dtype=wp.float32), grid_oz: wp.array(dtype=wp.float32), "
             "grid_pitch: wp.array(dtype=wp.float32), grid_iso: wp.array(dtype=wp.float32), grid_is_sdf: wp.array(dtype=wp.int32)")
_GRID_ARGS = "grid_data, grid_offset, grid_nx, grid_ny, grid_nz, grid_ox, grid_oy, grid_oz, grid_pitch, grid_iso, grid_is_sdf"


def _mat_to_literal(m):
    return "[" + ", ".join("[" + ", ".join(repr(float(x)) for x in row) + "]" for row in m) + "]"


class _Codegen:
    """Post-order-emits one wp.func per node. fname(node_id) -> str."""

    def __init__(self):
        self.lines = []
        self.grid_leaves = []   # list of grid leaf node dicts, in first-seen order -> leaf_id
        self._n = 0

    def _new_id(self):
        self._n += 1
        return self._n

    def emit(self, node):
        op = node["op"]
        if op == "halfspace":
            nid = self._new_id()
            nx, ny, nz = node["normal"]
            off = node["offset"]
            self.lines.append(f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                               f"    return px*{nx!r} + py*{ny!r} + pz*{nz!r} - {off!r}\n")
            return nid
        if op == "sphere":
            nid = self._new_id()
            cx, cy, cz = node["center"]
            r = node["radius"]
            self.lines.append(f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                               f"    dx = px - {cx!r}\n    dy = py - {cy!r}\n    dz = pz - {cz!r}\n"
                               f"    return wp.sqrt(dx*dx + dy*dy + dz*dz) - {r!r}\n")
            return nid
        if op == "box":
            nid = self._new_id()
            cx, cy, cz = node["center"]
            hx, hy, hz = node["half_extents"]
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                f"    qx = wp.abs(px - {cx!r}) - {hx!r}\n"
                f"    qy = wp.abs(py - {cy!r}) - {hy!r}\n"
                f"    qz = wp.abs(pz - {cz!r}) - {hz!r}\n"
                f"    ox = wp.max(qx, float(0.0)); oy = wp.max(qy, float(0.0)); oz = wp.max(qz, float(0.0))\n"
                f"    outside = wp.sqrt(ox*ox + oy*oy + oz*oz)\n"
                f"    inside = wp.min(wp.max(qx, wp.max(qy, qz)), float(0.0))\n"
                f"    return outside + inside\n")
            return nid
        if op in ("cylinder", "cone"):
            nid = self._new_id()
            cx, cy, cz = node["center"]
            axis = node["axis"]
            h = node["height"]
            if axis == "z":
                rad_expr, ax_expr = "wp.sqrt(lx*lx + ly*ly)", "lz"
            elif axis == "y":
                rad_expr, ax_expr = "wp.sqrt(lx*lx + lz*lz)", "ly"
            else:
                rad_expr, ax_expr = "wp.sqrt(ly*ly + lz*lz)", "lx"
            if op == "cylinder":
                r = node["radius"]
                self.lines.append(
                    f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                    f"    lx = px - {cx!r}; ly = py - {cy!r}; lz = pz - {cz!r}\n"
                    f"    rad_v = {rad_expr}; ax_v = {ax_expr}\n"
                    f"    qx = rad_v - {r!r}\n    qy = wp.abs(ax_v) - {h / 2.0!r}\n"
                    f"    ox = wp.max(qx, float(0.0)); oy = wp.max(qy, float(0.0))\n"
                    f"    outside = wp.sqrt(ox*ox + oy*oy)\n"
                    f"    inside = wp.min(wp.max(qx, qy), float(0.0))\n"
                    f"    return outside + inside\n")
            else:
                r1, r2 = node["radius1"], node["radius2"]
                self.lines.append(
                    f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                    f"    lx = px - {cx!r}; ly = py - {cy!r}; lz = pz - {cz!r}\n"
                    f"    rad_v = {rad_expr}; ax_v = {ax_expr}\n"
                    f"    t = (ax_v + {h / 2.0!r}) / {h!r}\n"
                    f"    t = wp.clamp(t, float(0.0), float(1.0))\n"
                    f"    r_at = {r1!r} + ({r2!r} - {r1!r}) * t\n"
                    f"    qy = wp.abs(ax_v) - {h / 2.0!r}\n    qx = rad_v - r_at\n"
                    f"    ox = wp.max(qx, float(0.0)); oy = wp.max(qy, float(0.0))\n"
                    f"    outside = wp.sqrt(ox*ox + oy*oy)\n"
                    f"    inside = wp.min(wp.max(qx, qy), float(0.0))\n"
                    f"    return outside + inside\n")
            return nid
        if op == "gridfield":
            leaf_id = len(self.grid_leaves)
            self.grid_leaves.append(node)
            nid = self._new_id()
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                f"    return _sample_grid_leaf(px, py, pz, {leaf_id}, {_GRID_ARGS})\n")
            return nid
        if op == "transform":
            cid = self.emit(node["child"])
            inv = E.mat4_inverse_affine(node["mat4"])
            nid = self._new_id()
            m = inv
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                f"    lx = {m[0][0]!r}*px + {m[0][1]!r}*py + {m[0][2]!r}*pz + {m[0][3]!r}\n"
                f"    ly = {m[1][0]!r}*px + {m[1][1]!r}*py + {m[1][2]!r}*pz + {m[1][3]!r}\n"
                f"    lz = {m[2][0]!r}*px + {m[2][1]!r}*py + {m[2][2]!r}*pz + {m[2][3]!r}\n"
                f"    return _f{cid}(lx, ly, lz, {_GRID_ARGS})\n")
            return nid
        if op == "offset":
            cid = self.emit(node["child"])
            nid = self._new_id()
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                f"    return _f{cid}(px, py, pz, {_GRID_ARGS}) - {node['delta']!r}\n")
            return nid
        if op == "axis_project":
            # sdf_shell_v1: freezes p[axis] to `ref` before sampling child -- see expr.axis_project's
            # docstring (shell's open-face construction). Literal-baked here like every other op this
            # codegen emits; `ref` is included in _codegen_shape's hashed fields (below) so a differing
            # ref never hits a stale cached kernel with the OLD baked value.
            cid = self.emit(node["child"])
            nid = self._new_id()
            axis, ref = node["axis"], node["ref"]
            if axis == 0:
                call = f"_f{cid}({ref!r}, py, pz, {_GRID_ARGS})"
            elif axis == 1:
                call = f"_f{cid}(px, {ref!r}, pz, {_GRID_ARGS})"
            else:
                call = f"_f{cid}(px, py, {ref!r}, {_GRID_ARGS})"
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                f"    return {call}\n")
            return nid
        if op == "scale":
            cid = self.emit(node["child"])
            s = node["s"]
            nid = self._new_id()
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                f"    v = _f{cid}(px/{s!r}, py/{s!r}, pz/{s!r}, {_GRID_ARGS})\n"
                f"    return v * {s!r}\n")
            return nid
        if op in ("union", "intersect", "subtract"):
            aid = self.emit(node["a"])
            bid = self.emit(node["b"])
            k = node["k"]
            nid = self._new_id()
            if op == "union":
                hard = "wp.min(da, db)"
                smooth = ("h = wp.max(float(0.0), {k!r} - wp.abs(da - db)) / {k!r}\n"
                          "    return wp.min(da, db) - h*h*{k!r}*float(0.25)").format(k=k)
            elif op == "intersect":
                hard = "wp.max(da, db)"
                smooth = ("h = wp.max(float(0.0), {k!r} - wp.abs(da - db)) / {k!r}\n"
                          "    return wp.max(da, db) + h*h*{k!r}*float(0.25)").format(k=k)
            else:
                hard = "wp.max(da, -db)"
                smooth = ("h = wp.max(float(0.0), {k!r} - wp.abs(da + db)) / {k!r}\n"
                          "    return wp.max(da, -db) + h*h*{k!r}*float(0.25)").format(k=k)
            if k <= 0.0:
                body = f"    return {hard}\n"
            else:
                body = f"    {smooth}\n"
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_GRID_SIG}):\n"
                f"    da = _f{aid}(px, py, pz, {_GRID_ARGS})\n"
                f"    db = _f{bid}(px, py, pz, {_GRID_ARGS})\n" + body)
            return nid
        raise ValueError(f"unknown op {op!r}")


def _structural_hash(node):
    """Stable hash of the tree SHAPE+PARAMS (used as the kernel-cache key -- two calls with the same
    expression, even rebuilt independently, hit the same compiled kernel)."""
    return hashlib.sha256(json.dumps(node, sort_keys=True).encode()).hexdigest()[:16]


def _codegen_shape(node):
    """Strips gridfield leaves down to a CANONICAL placeholder before hashing for the KERNEL-cache key
    (v1.1 fix, found this cell's scene-sweep benchmark: EVERY intrusion_volume() call between two
    DIFFERENT gridfield leaves was recompiling a fresh Warp kernel, ~330-340ms each, even though the
    generated SOURCE is byte-identical -- gridfield codegen (see _Codegen.emit's "gridfield" branch)
    only ever emits `_sample_grid_leaf(px,py,pz,{leaf_id},...)`, where leaf_id is the leaf's FIRST-SEEN
    POSITION in the tree, NOT its npz_path/key/mode/iso (those are runtime data threaded through the
    packed grid_* arrays _pack_gridfields uploads separately, keyed by their OWN cache in
    _GRID_UPLOAD_CACHE) -- so any two trees with the SAME shape (e.g. intersect(gridfield, gridfield))
    compile to the SAME kernel regardless of which specific leaves they reference. The original
    _structural_hash hashed the FULL node dict (npz_path included), so a 285-solid scene sweep with N
    distinct AABB-overlap candidate pairs paid N full recompiles instead of ONE -- for the measured
    25-pair sample this cost 25*335ms=8.4s of the 9.0s total sweep wall time (93%), the dominant term
    by far, dwarfing actual GPU eval cost. Any node dict field NOT read by _Codegen.emit is stripped
    here before hashing; every OTHER param (primitive radii/centers/matrices/k-blend factors) still
    participates, since those genuinely change the emitted source."""
    op = node["op"]
    if op == "gridfield":
        return {"op": "gridfield"}   # leaf identity does not affect codegen, only leaf ORDER (which
                                       # walk() below preserves via list order, not dict content)
    out = {"op": op}
    for k in ("normal", "offset", "radius", "center", "half_extents", "height", "axis",
              "radius1", "radius2", "mat4", "delta", "s", "k", "ref"):
        if k in node:
            out[k] = node[k]
    for k in ("child", "a", "b"):
        if k in node:
            out[k] = _codegen_shape(node[k])
    return out


def _kernel_cache_key(node):
    return hashlib.sha256(json.dumps(_codegen_shape(node), sort_keys=True).encode()).hexdigest()[:16]


def _collect_grid_leaves(node):
    """Walks `node` in the SAME left-to-right, post-order sequence _Codegen.emit visits it in, collecting
    gridfield leaf dicts in traversal order (matching the leaf_id each one would be assigned during a
    fresh emit). Pure-python, no codegen/compile -- exists so compile_expr's kernel-cache hit path (below)
    can return the CALLER's actual leaves instead of the leaves baked into a differently-parameterized
    tree that happened to compile the cached kernel first (see compile_expr's docstring, RENDER_V2 bug
    find)."""
    op = node["op"]
    if op == "gridfield":
        return [node]
    out = []
    for k in ("child", "a", "b"):
        if k in node:
            out.extend(_collect_grid_leaves(node[k]))
    return out


def compile_expr(node):
    """Codegen + wp.kernel compile for `node`. Returns (kernel, grid_leaves list, module_ns). The KERNEL
    is cached by CODEGEN-SHAPE hash (v1.1 fix, see _codegen_shape's docstring) -- NOT the full structural
    hash -- so two DIFFERENT gridfield-leaf pairs with the SAME tree shape (e.g. every intrusion_volume()
    call in a scene sweep, which always builds intersect(leafA, leafB)) hit the SAME compiled kernel; only
    a genuinely different tree SHAPE (different op arrangement, different primitive params, a different
    NUMBER of gridfield leaves) forces a recompile.

    The grid_leaves list is never taken from the cache on a shape-hash hit: it is data (which specific
    leaves this node references), not codegen shape, so two different single-gridfield trees would
    otherwise both hash to {"op":"gridfield"} and the second call would get the first call's leaf. It is
    always recomputed for the node passed in; only the compiled kernel, module and source are reused.
    """
    h = _kernel_cache_key(node)
    if h in _KERNEL_CACHE:
        kernel, _stale_grid_leaves, mod, src = _KERNEL_CACHE[h]
        return kernel, _collect_grid_leaves(node), mod, src

    cg = _Codegen()
    root_id = cg.emit(node)
    src_lines = ["import warp as wp", "", _GRID_SAMPLE_SRC_REAL, ""] + cg.lines
    src_lines.append(
        f"@wp.kernel\n"
        f"def eval_kernel(pts: wp.array(dtype=wp.vec3), {_GRID_SIG}, out: wp.array(dtype=wp.float32)):\n"
        f"    tid = wp.tid()\n"
        f"    p = pts[tid]\n"
        f"    out[tid] = _f{root_id}(p[0], p[1], p[2], {_GRID_ARGS})\n")
    src = "\n".join(src_lines)

    # Warp inspects source via inspect.getsourcelines, which needs a real file on disk (exec() of a
    # string is unsupported by Warp's codegen), so the generated module is written to the cache dir and
    # imported via importlib, keyed by the structural hash.
    cache_dir = os.path.join(HERE, "_gen_cache")
    os.makedirs(cache_dir, exist_ok=True)
    mod_path = os.path.join(cache_dir, f"ikarus_expr_{h}.py")
    if not os.path.exists(mod_path):
        with open(mod_path, "w") as f:
            f.write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"ikarus_expr_{h}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    result = (mod.eval_kernel, cg.grid_leaves, mod, src)
    _KERNEL_CACHE[h] = result
    return result


_GRID_UPLOAD_CACHE = {}   # (tuple of (npz_path, key, mode, iso), device) -> packed wp.array tuple


def _pack_gridfields(grid_leaves, device):
    """Load each gridfield leaf's .npz, concatenate into one flat float32 array + per-leaf metadata
    arrays, uploaded once to `device`. Returns the 11 wp.array args _sample_grid_leaf expects.

    Cached by (leaf identity tuple, device): a per-call reload and re-upload otherwise dominates wall
    time, and leaf data is immutable once built, so one upload per (leaf, device) is reused by every
    query against it.
    """
    cache_key = (tuple((lf["npz_path"], lf["key"], lf["mode"], lf["iso"]) for lf in grid_leaves), device)
    if cache_key in _GRID_UPLOAD_CACHE:
        return _GRID_UPLOAD_CACHE[cache_key]

    n = max(len(grid_leaves), 1)
    flat_chunks = []
    offset_v, nx_v, ny_v, nz_v = [], [], [], []
    ox_v, oy_v, oz_v, pitch_v, iso_v, is_sdf_v = [], [], [], [], [], []
    cur_off = 0
    for leaf in grid_leaves:
        d = np.load(leaf["npz_path"])
        arr = np.asarray(d[leaf["key"]], dtype=np.float32)
        nxv, nyv, nzv = arr.shape
        origin = d["global_origin_mm"] if "global_origin_mm" in d else np.zeros(3)
        pitch = leaf["pitch_override"] or float(d["pitch_mm"]) if "pitch_mm" in d else 1.0
        flat_chunks.append(arr.reshape(-1))
        offset_v.append(cur_off)
        nx_v.append(nxv); ny_v.append(nyv); nz_v.append(nzv)
        ox_v.append(float(origin[0])); oy_v.append(float(origin[1])); oz_v.append(float(origin[2]))
        pitch_v.append(float(pitch)); iso_v.append(float(leaf["iso"]))
        is_sdf_v.append(1 if leaf["mode"] == "sdf" else 0)
        cur_off += arr.size
    if not flat_chunks:
        flat = np.zeros(1, dtype=np.float32)
        offset_v, nx_v, ny_v, nz_v = [0], [1], [1], [1]
        ox_v, oy_v, oz_v, pitch_v, iso_v, is_sdf_v = [0.0], [0.0], [0.0], [1.0], [0.5], [0]
    else:
        flat = np.concatenate(flat_chunks)

    mk_i = lambda v: wp.array(np.asarray(v, dtype=np.int32), dtype=wp.int32, device=device)
    mk_f = lambda v: wp.array(np.asarray(v, dtype=np.float32), dtype=wp.float32, device=device)
    packed = (wp.array(flat, dtype=wp.float32, device=device), mk_i(offset_v), mk_i(nx_v), mk_i(ny_v), mk_i(nz_v),
              mk_f(ox_v), mk_f(oy_v), mk_f(oz_v), mk_f(pitch_v), mk_f(iso_v), mk_i(is_sdf_v))
    _GRID_UPLOAD_CACHE[cache_key] = packed
    return packed


def eval_batch(node, points_mm, device=DEVICE):
    """Batch SDF evaluation: points_mm (N,3) float array -> (N,) SDF values (mm, fp32). This is the ONE
    primitive queries.py's dispatcher builds every higher-level query (inside/outside, distance, volume,
    clearance) on top of."""
    kernel, grid_leaves, ns, src = compile_expr(node)
    grid_args = _pack_gridfields(grid_leaves, device)
    n = points_mm.shape[0]
    pts = wp.array(points_mm.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(kernel, dim=n, inputs=[pts, *grid_args], outputs=[out], device=device)
    wp.synchronize()
    return out.numpy()


# --------------------------------------------------------------------- one kernel launch for many pairs
# Calling eval_batch() once per candidate pair pays N separate launch+synchronize round trips, and each
# launch has a fixed overhead (dispatch bookkeeping, host-device sync, array conversion) that dominates a
# scene sweep once the kernel cache removes the recompile. Concatenating every pair's coarse/refine grid
# points into one flat buffer plus a per-point pair-id index collapses N launches into one, so the fixed
# overhead is paid once for the whole sweep. This covers the common case only (both operands gridfield
# leaves, hard intersect, k=0); the general per-op codegen path above is unchanged and still used for
# everything else. It is an additive fast path, not a replacement.
_MULTI_PAIR_KERNEL_CACHE = {}
MULTI_PAIR_SHAPE_KEY = "multi_pair_gridfield_hard_intersect_v1_2"


def compile_multi_pair_gridfield_kernel(device=DEVICE):
    """Compile (once, cached) the generic batched-pair kernel: out[tid] = max(sdfA, sdfB) where A/B are
    selected PER-POINT via pair_of_point[tid] -> (leaf_a_of_pair, leaf_b_of_pair). Fixed kernel shape
    (does not depend on how many pairs or which leaves) -- compiled exactly once per device for the
    entire process lifetime, unlike compile_expr's per-tree-shape cache."""
    if device in _MULTI_PAIR_KERNEL_CACHE:
        return _MULTI_PAIR_KERNEL_CACHE[device]

    src_lines = ["import warp as wp", "", _GRID_SAMPLE_SRC_REAL, "",
        f"@wp.kernel\n"
        f"def eval_kernel_multi_pair(pts: wp.array(dtype=wp.vec3),\n"
        f"                            pair_of_point: wp.array(dtype=wp.int32),\n"
        f"                            leaf_a_of_pair: wp.array(dtype=wp.int32),\n"
        f"                            leaf_b_of_pair: wp.array(dtype=wp.int32),\n"
        f"                            {_GRID_SIG},\n"
        f"                            out: wp.array(dtype=wp.float32)):\n"
        f"    tid = wp.tid()\n"
        f"    p = pts[tid]\n"
        f"    pid = pair_of_point[tid]\n"
        f"    la = leaf_a_of_pair[pid]\n"
        f"    lb = leaf_b_of_pair[pid]\n"
        f"    da = _sample_grid_leaf(p[0], p[1], p[2], la, {_GRID_ARGS})\n"
        f"    db = _sample_grid_leaf(p[0], p[1], p[2], lb, {_GRID_ARGS})\n"
        f"    out[tid] = wp.max(da, db)\n"]
    src = "\n".join(src_lines)

    cache_dir = os.path.join(HERE, "_gen_cache")
    os.makedirs(cache_dir, exist_ok=True)
    mod_path = os.path.join(cache_dir, "ikarus_multi_pair_kernel_v1_2.py")
    if not os.path.exists(mod_path):
        with open(mod_path, "w") as f:
            f.write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("ikarus_multi_pair_kernel_v1_2", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MULTI_PAIR_KERNEL_CACHE[device] = mod.eval_kernel_multi_pair
    return mod.eval_kernel_multi_pair


def eval_batch_multi_pair(points_mm, pair_of_point, leaf_a_of_pair, leaf_b_of_pair, grid_leaves_all,
                           device=DEVICE):
    """Batch SDF evaluation across MANY (leafA,leafB) pairs in ONE kernel launch.
    points_mm: (N,3) concatenated points from every pair's grid pass.
    pair_of_point: (N,) int32, which pair (0..n_pairs-1) each point belongs to.
    leaf_a_of_pair/leaf_b_of_pair: (n_pairs,) int32 indices into grid_leaves_all.
    grid_leaves_all: the full list of distinct gridfield-leaf node dicts referenced by any pair (reuses
    _pack_gridfields' own (leaf-identity, device) cache, so a leaf already uploaded for a prior call is
    not re-uploaded).
    Returns (N,) float32 SDF values of max(sdfA, sdfB) (hard intersect) at each point."""
    kernel = compile_multi_pair_gridfield_kernel(device)
    grid_args = _pack_gridfields(grid_leaves_all, device)
    n = points_mm.shape[0]
    pts = wp.array(points_mm.astype(np.float32), dtype=wp.vec3, device=device)
    pop = wp.array(np.asarray(pair_of_point, dtype=np.int32), dtype=wp.int32, device=device)
    laf = wp.array(np.asarray(leaf_a_of_pair, dtype=np.int32), dtype=wp.int32, device=device)
    lbf = wp.array(np.asarray(leaf_b_of_pair, dtype=np.int32), dtype=wp.int32, device=device)
    out = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(kernel, dim=n, inputs=[pts, pop, laf, lbf, *grid_args], outputs=[out], device=device)
    wp.synchronize()
    return out.numpy()


# ------------------------------------------------------------------- the band plan as a kernel ARGUMENT
# The 2.5D band tree
#     union_i( intersect( axis_project(leaf, axis, zs_i), intersect(hs(+e,z1_i), hs(-e,-z0_i)) ) )
# has 7*n_bands-1 nodes, and compile_expr() bakes EVERY band plane (zs_i, z0_i, z1_i) in as a
# compile-time literal inside its own wp.func, so the emitted source grows with the band count and the
# compiler cost grows superlinearly in it (measured: 8 bands/55 nodes 4.85 s, 16/111 23.7 s, 28/195
# 355.2 s for the first launch). The kernel cache is keyed on the codegen shape, and the plan is part of
# that shape, so a loop that moves the band planes every round misses the cache every time.
#
# This is precisely the move this file has already made TWICE for gridfield leaves:
#   (1) _pack_gridfields: leaf VOXELS are runtime data in flat wp.arrays, not baked source, so the kernel
#       signature is fixed regardless of leaf count/size;
#   (2) _codegen_shape: leaf IDENTITY is stripped before hashing, so a new leaf never forces a recompile.
# The band plan is the same species of thing: DATA that was living in the source. Here it moves into three
# float32 arrays (band_zs/band_z0/band_z1) plus two ints (axis, n_bands), and the union/intersect/
# axis_project/halfspace algebra becomes a runtime loop over those arrays. Result: ONE fixed kernel shape,
# compiled once per device per process (and once per machine into ~/.cache/warp), reusable for ANY plan of
# ANY length -- the 39.9x warm win becomes reachable on the FIRST real round instead of never.
#
# EXACTNESS (why this is a refactor, not an approximation): the emitted arithmetic is the same expression
# in the same association order as the literal path.
#   * hs(+e, z1) = px*1.0 + py*0.0 + pz*0.0 - z1  ==  p[axis] - z1   (exact in fp: y*0.0 = 0.0, x+0.0 = x)
#   * hs(-e,-z0) = px*-1.0 + ... - (-z0)          ==  z0 - p[axis]   (fl(-a - -b) = fl(b - a))
#   * intersect  = wp.max(a, b), same operand order; union = wp.min(a, b), left-folded == a running min
#     (min is exact and associative, so fold order cannot change a bit)
#   * axis_project freezes p[axis] := zs_i; the literal path bakes repr(float64) which nvrtc rounds to the
#     nearest float32, the array path uploads np.float32(zs_i) -- the SAME rounding of the SAME double.
# So the two paths must agree BIT-FOR-BIT on the same plan, and eval_band_tree_selftest() below measures
# exactly that instead of asserting it.
_BAND_TREE_KERNEL_CACHE = {}
BAND_TREE_SHAPE_KEY = "band_tree_axis_project_slab_union_v1_3"


def compile_band_tree_kernel(device=DEVICE):
    """Compile (once, cached per device) the fixed-shape band-tree kernel. The band plan arrives as
    ARGUMENTS (band_zs/band_z0/band_z1 arrays + axis + n_bands), never as baked literals, so a plan
    change costs an array upload instead of an nvrtc run."""
    if device in _BAND_TREE_KERNEL_CACHE:
        return _BAND_TREE_KERNEL_CACHE[device]

    src_lines = ["import warp as wp", "", _GRID_SAMPLE_SRC_REAL, "",
        f"@wp.kernel\n"
        f"def eval_kernel_band_tree(pts: wp.array(dtype=wp.vec3),\n"
        f"                           axis: int,\n"
        f"                           n_bands: int,\n"
        f"                           leaf_id: int,\n"
        f"                           band_zs: wp.array(dtype=wp.float32),\n"
        f"                           band_z0: wp.array(dtype=wp.float32),\n"
        f"                           band_z1: wp.array(dtype=wp.float32),\n"
        f"                           {_GRID_SIG},\n"
        f"                           out: wp.array(dtype=wp.float32)):\n"
        f"    tid = wp.tid()\n"
        f"    p = pts[tid]\n"
        f"    pax = p[0]\n"
        f"    if axis == 1:\n"
        f"        pax = p[1]\n"
        f"    if axis == 2:\n"
        f"        pax = p[2]\n"
        f"    best = float(1.0e30)\n"
        f"    for i in range(n_bands):\n"
        f"        qx = p[0]\n"
        f"        qy = p[1]\n"
        f"        qz = p[2]\n"
        f"        if axis == 0:\n"
        f"            qx = band_zs[i]\n"
        f"        if axis == 1:\n"
        f"            qy = band_zs[i]\n"
        f"        if axis == 2:\n"
        f"            qz = band_zs[i]\n"
        f"        d_body = _sample_grid_leaf(qx, qy, qz, leaf_id, {_GRID_ARGS})\n"
        f"        d_slab = wp.max(pax - band_z1[i], band_z0[i] - pax)\n"
        f"        d = wp.max(d_body, d_slab)\n"
        f"        best = wp.min(best, d)\n"
        f"    out[tid] = best\n"]
    src = "\n".join(src_lines)

    cache_dir = os.path.join(HERE, "_gen_cache")
    os.makedirs(cache_dir, exist_ok=True)
    mod_path = os.path.join(cache_dir, "ikarus_band_tree_kernel_v1_3.py")
    if not os.path.exists(mod_path) or open(mod_path).read() != src:
        with open(mod_path, "w") as f:
            f.write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("ikarus_band_tree_kernel_v1_3", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _BAND_TREE_KERNEL_CACHE[device] = mod.eval_kernel_band_tree
    return mod.eval_kernel_band_tree


class BandTreeEvaluator:
    """A leaf + the compiled band-tree kernel, with the PLAN swappable per call.

    Holds the (immutable) gridfield leaf upload and the compiled kernel; `eval(points, plan)` uploads only
    the plan's three small float32 arrays. Deliberately an object rather than a function so the loop can
    keep the leaf hot across rungs -- the leaf upload, not the kernel, is the only per-leaf cost left."""

    def __init__(self, leaf_node, axis, device=DEVICE):
        self.kernel = compile_band_tree_kernel(device)
        self.leaf = leaf_node
        self.axis = int(axis)
        self.device = device
        self.grid_args = _pack_gridfields([leaf_node], device)
        self._plan_cache = {}

    def _plan_arrays(self, bands):
        key = tuple((float(b["zs"]), float(b["z0"]), float(b["z1"])) for b in bands)
        got = self._plan_cache.get(key)
        if got is None:
            zs = np.array([b["zs"] for b in bands], dtype=np.float32)
            z0 = np.array([b["z0"] for b in bands], dtype=np.float32)
            z1 = np.array([b["z1"] for b in bands], dtype=np.float32)
            got = (wp.array(zs, dtype=wp.float32, device=self.device),
                   wp.array(z0, dtype=wp.float32, device=self.device),
                   wp.array(z1, dtype=wp.float32, device=self.device),
                   len(bands))
            self._plan_cache[key] = got
        return got

    def eval(self, points_mm, bands):
        a_zs, a_z0, a_z1, n_bands = self._plan_arrays(bands)
        n = points_mm.shape[0]
        pts = wp.array(points_mm.astype(np.float32), dtype=wp.vec3, device=self.device)
        out = wp.zeros(n, dtype=wp.float32, device=self.device)
        wp.launch(self.kernel, dim=n,
                  inputs=[pts, self.axis, n_bands, 0, a_zs, a_z0, a_z1, *self.grid_args],
                  outputs=[out], device=self.device)
        wp.synchronize()
        return out.numpy()


def band_tree_expr(leaf_node, axis, bands):
    """The literal-baked reference tree (what compile_expr compiles) for a band plan.

    Takes a leaf node, an axis index and a list of band dicts (zs/z0/z1); returns the expression tree,
    so the argument path can be bit-compared against it.
    """
    def slab(z0, z1):
        n = [0.0, 0.0, 0.0]; n[axis] = 1.0
        nm = [0.0, 0.0, 0.0]; nm[axis] = -1.0
        return E.intersect(E.halfspace(n, z1), E.halfspace(nm, -z0))
    node = None
    for b in bands:
        piece = E.intersect(E.axis_project(leaf_node, axis, b["zs"]), slab(b["z0"], b["z1"]))
        node = piece if node is None else E.union(node, piece)
    return node


# --------------------------------------------------------------------------------------------- gradient (v2 sphere-tracer contract, item (b))
def gradient_batch(node, points_mm, h=0.05, device=DEVICE):
    """Central-difference surface normal/gradient at each point: ONE extra eval_batch call over 6N offset
    points (not 6 separate launches) -- works for ANY node (primitive OR gridfield leaf) since it only
    needs eval_batch, not per-op analytic derivative code. This is the numeric baseline the v2 sphere-
    tracer's shading needs (surface normal = normalize(gradient)); an analytic-gradient codegen pass
    (differentiate each wp.func alongside its value, doubling the emitted source) is a legitimate future
    perf upgrade over this central-difference version, not required for v2 to start working.
    Returns (N,3) gradient array (mm SDF units per mm)."""
    n = points_mm.shape[0]
    offsets = np.array([[h, 0, 0], [-h, 0, 0], [0, h, 0], [0, -h, 0], [0, 0, h], [0, 0, -h]])
    stacked = (points_mm[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    vals = eval_batch(node, stacked, device=device).reshape(6, n)
    gx = (vals[0] - vals[1]) / (2 * h)
    gy = (vals[2] - vals[3]) / (2 * h)
    gz = (vals[4] - vals[5]) / (2 * h)
    return np.stack([gx, gy, gz], axis=1)


# --------------------------------------------------------------------------------------------- self-test / validation gate
def selftest():
    """Cross-check: for a tree of analytic primitives only, the generated kernel must match expr.py's
    pure-python eval_sdf_py at random points to fp32 rounding.

    Takes no arguments; returns a dict with the max/mean absolute error, the gate verdict and the
    first-call vs cached-call timings.
    """
    import random
    random.seed(3)
    c = E.cylinder(radius=12.5, height=40.0)
    b = E.box(half_extents=(8.0, 8.0, 8.0), center=(3.0, 0.0, 5.0))
    s = E.sphere(radius=6.0, center=(-4.0, 2.0, -3.0))
    tree = E.union(E.subtract(c, b, k=0.0), E.rotate(s, "y", 37.0), k=2.0)

    pts = np.array([[random.uniform(-30, 30), random.uniform(-30, 30), random.uniform(-30, 30)]
                     for _ in range(4000)], dtype=np.float64)

    t0 = time.time()
    gpu_vals = eval_batch(tree, pts)
    t_gpu_first = time.time() - t0
    t0 = time.time()
    gpu_vals2 = eval_batch(tree, pts)   # second call: kernel-cache hit, measures steady-state
    t_gpu_cached = time.time() - t0

    cpu_vals = np.array([E.eval_sdf_py(tree, tuple(p)) for p in pts])

    err = np.abs(gpu_vals - cpu_vals)
    result = {
        "n_points": len(pts),
        "max_abs_err_mm": float(err.max()),
        "mean_abs_err_mm": float(err.mean()),
        "gate_pass": bool(err.max() < 1e-3),
        "t_gpu_first_call_s": t_gpu_first,
        "t_gpu_cached_call_s": t_gpu_cached,
        "cache_speedup_x": t_gpu_first / max(t_gpu_cached, 1e-9),
        "device": DEVICE,
    }
    return result


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if stage == "selftest":
        r = selftest()
        print(json.dumps(r, indent=2))
        assert r["gate_pass"], "eval_warp.py selftest FAILED vs pure-python ground truth"
        print("PASS")
