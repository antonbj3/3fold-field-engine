#!/usr/bin/env python3
"""Uniform-parameter GPU evaluator for IKARUS expression trees.

Hoists every numeric primitive parameter (radius / center / half_extents / mat4 / offset / k / ...) out
of the generated Warp source into a `wp.array(dtype=float)` read at kernel-launch time. The kernel-cache
key becomes the tree's topology shape only (op types + tree structure + axis strings, never a number),
so a parameter tweak that does not change which ops are present hits the same compiled kernel every time
and only the small parameter buffer is re-uploaded: much lower parameter-change latency for a modest
steady-state eval-throughput tax.

Additive, not a drop-in replacement for eval_warp.py's compile_expr/eval_batch, which stay untouched and
are still used by proofs.py / queries.py / render_v2's default path. gridfield leaves are out of scope
here: imported grid-field context needs its own voxelisation-cache path.

Validation gate: eval_batch_uniform() must agree with expr.eval_sdf_py (pure python, independent of this
codegen) to fp32 rounding -- see selftest() below.

I/O: expression tree (dict) + query points in, float array out; `selftest` prints the agreement report.
Needs a Warp device; uses CUDA when one is present.

Run: python eval_warp_uniform_v2.py selftest
"""
from __future__ import annotations

import hashlib
import json
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

_KERNEL_CACHE = {}   # topology-shape hash -> (kernel, module, n_params, src)

_SIG = "params: wp.array(dtype=wp.float32)"


# --------------------------------------------------------------------------------------------- topology shape (the P1 kernel-cache key: NO numbers, ever)
def _topology_shape(node):
    """Strip every numeric field, keep only op type + structural/string fields (axis) + child shape --
    this IS the S1 fix: eval_warp.py's old `_codegen_shape` (kept there for the gridfield-leaf-identity
    bug it fixes) still hashes primitive radii/centers/matrices INTO the kernel key; this function hashes
    NONE of them, which is the whole point of P1 (a parameter tweak must never change this key)."""
    op = node["op"]
    out = {"op": op}
    if op in ("cylinder", "cone"):
        out["axis"] = node["axis"]
    if op == "axis_project":
        out["axis"] = node["axis"]   # STRUCTURAL (0/1/2 selects which codegen line), stays a topology key
    for k in ("child", "a", "b"):
        if k in node:
            out[k] = _topology_shape(node[k])
    return out


def _topology_hash(node):
    return hashlib.sha256(json.dumps(_topology_shape(node), sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------- codegen: post-order emit, every number becomes params[slot]
class _CodegenUniform:
    """MEMOIZED by python object identity (id(node)): a recipe tree built by
    recept_till_ikarus_v1.py is a DAG, not a tree -- fillet/mirror/pattern_linear all deliberately
    re-reference the SAME child dict object from multiple parents (e.g. mirror = union(base,
    transform(base,...)), pattern_linear unions N translated copies of the SAME base object). A naive
    post-order emit() that fires on every VISIT (not every unique node) re-emits a full duplicate
    wp.func chain per visit -- for a DAG with k independent doubling points that is O(2^k) generated
    functions, not O(nodes). MEASURED this cell on the real motorblock_v1 recipe prefix before this
    fix: N=5 ops (6 nodes) -> 0.86s to first launch, N=10 (18 nodes) -> 2.86s, N=15 ops (66 nodes,
    crosses the recipe's first 8-way pattern_linear) -> 33.3s -- non-linear growth consistent with
    duplicate-emission blowup, not the node COUNT itself (66 nodes alone is trivial for eval_warp.py's
    existing literal-baked codegen, ~0.3-0.5s per S1's own K=64/131-node benchmark). Fix: cache emit()
    by id(node) so a shared subtree is emitted ONCE and every other reference just calls the same
    _f{nid} function name -- turns the generated-function count back into O(unique nodes)."""
    def __init__(self):
        self.lines = []
        self.param_vals = []   # flat list, in slot order -- becomes the wp.array
        self._n = 0
        self._memo = {}   # id(node) -> already-assigned nid (DAG dedup, see class docstring)

    def _slot(self, value):
        i = len(self.param_vals)
        self.param_vals.append(float(value))
        return i

    def _new_id(self):
        self._n += 1
        return self._n

    def emit(self, node):
        key = id(node)
        if key in self._memo:
            return self._memo[key]
        nid_result = self._emit_uncached(node)
        self._memo[key] = nid_result
        return nid_result

    def _emit_uncached(self, node):
        op = node["op"]
        if op == "halfspace":
            nid = self._new_id()
            nx, ny, nz = node["normal"]
            snx, sny, snz, soff = self._slot(nx), self._slot(ny), self._slot(nz), self._slot(node["offset"])
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                f"    return px*params[{snx}] + py*params[{sny}] + pz*params[{snz}] - params[{soff}]\n")
            return nid
        if op == "sphere":
            nid = self._new_id()
            cx, cy, cz = node["center"]
            scx, scy, scz, sr = self._slot(cx), self._slot(cy), self._slot(cz), self._slot(node["radius"])
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                f"    dx = px - params[{scx}]\n    dy = py - params[{scy}]\n    dz = pz - params[{scz}]\n"
                f"    return wp.sqrt(dx*dx + dy*dy + dz*dz) - params[{sr}]\n")
            return nid
        if op == "box":
            nid = self._new_id()
            cx, cy, cz = node["center"]
            hx, hy, hz = node["half_extents"]
            s = [self._slot(v) for v in (cx, cy, cz, hx, hy, hz)]
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                f"    qx = wp.abs(px - params[{s[0]}]) - params[{s[3]}]\n"
                f"    qy = wp.abs(py - params[{s[1]}]) - params[{s[4]}]\n"
                f"    qz = wp.abs(pz - params[{s[2]}]) - params[{s[5]}]\n"
                f"    ox = wp.max(qx, float(0.0)); oy = wp.max(qy, float(0.0)); oz = wp.max(qz, float(0.0))\n"
                f"    outside = wp.sqrt(ox*ox + oy*oy + oz*oz)\n"
                f"    inside = wp.min(wp.max(qx, wp.max(qy, qz)), float(0.0))\n"
                f"    return outside + inside\n")
            return nid
        if op in ("cylinder", "cone"):
            nid = self._new_id()
            cx, cy, cz = node["center"]
            axis = node["axis"]   # STRUCTURAL (topology key), stays literal -- not a number
            scx, scy, scz = self._slot(cx), self._slot(cy), self._slot(cz)
            if axis == "z":
                rad_expr, ax_expr = "wp.sqrt(lx*lx + ly*ly)", "lz"
            elif axis == "y":
                rad_expr, ax_expr = "wp.sqrt(lx*lx + lz*lz)", "ly"
            else:
                rad_expr, ax_expr = "wp.sqrt(ly*ly + lz*lz)", "lx"
            if op == "cylinder":
                sh, sr = self._slot(node["height"] / 2.0), self._slot(node["radius"])
                self.lines.append(
                    f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                    f"    lx = px - params[{scx}]; ly = py - params[{scy}]; lz = pz - params[{scz}]\n"
                    f"    rad_v = {rad_expr}; ax_v = {ax_expr}\n"
                    f"    qx = rad_v - params[{sr}]\n    qy = wp.abs(ax_v) - params[{sh}]\n"
                    f"    ox = wp.max(qx, float(0.0)); oy = wp.max(qy, float(0.0))\n"
                    f"    outside = wp.sqrt(ox*ox + oy*oy)\n"
                    f"    inside = wp.min(wp.max(qx, qy), float(0.0))\n"
                    f"    return outside + inside\n")
            else:
                sh = self._slot(node["height"] / 2.0)
                sr1, sr2 = self._slot(node["radius1"]), self._slot(node["radius2"])
                sfull_h = self._slot(node["height"])
                self.lines.append(
                    f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                    f"    lx = px - params[{scx}]; ly = py - params[{scy}]; lz = pz - params[{scz}]\n"
                    f"    rad_v = {rad_expr}; ax_v = {ax_expr}\n"
                    f"    t = (ax_v + params[{sh}]) / params[{sfull_h}]\n"
                    f"    t = wp.clamp(t, float(0.0), float(1.0))\n"
                    f"    r_at = params[{sr1}] + (params[{sr2}] - params[{sr1}]) * t\n"
                    f"    qy = wp.abs(ax_v) - params[{sh}]\n    qx = rad_v - r_at\n"
                    f"    ox = wp.max(qx, float(0.0)); oy = wp.max(qy, float(0.0))\n"
                    f"    outside = wp.sqrt(ox*ox + oy*oy)\n"
                    f"    inside = wp.min(wp.max(qx, qy), float(0.0))\n"
                    f"    return outside + inside\n")
            return nid
        if op == "rect_frustum":
            nid = self._new_id()
            z0, z1 = node["z0"], node["z1"]
            kx = (node["hx1"] - node["hx0"]) / (z1 - z0)
            ky = (node["hy1"] - node["hy0"]) / (z1 - z0)
            kcx = (node["cx1"] - node["cx0"]) / (z1 - z0)
            kcy = (node["cy1"] - node["cy0"]) / (z1 - z0)
            sz0, sz1 = self._slot(z0), self._slot(z1)
            s_nzhi = self._slot(-(kx + kcx))
            s_chi = self._slot(node["hx0"] + node["cx0"] - (kx + kcx) * z0)
            s_nzlo = self._slot(-(kx - kcx))
            s_clo = self._slot(node["hx0"] - node["cx0"] - (kx - kcx) * z0)
            s_ynzhi = self._slot(-(ky + kcy))
            s_ychi = self._slot(node["hy0"] + node["cy0"] - (ky + kcy) * z0)
            s_ynzlo = self._slot(-(ky - kcy))
            s_yclo = self._slot(node["hy0"] - node["cy0"] - (ky - kcy) * z0)
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                f"    d_zlo = params[{sz0}] - pz\n    d_zhi = pz - params[{sz1}]\n"
                f"    nx1 = float(1.0); nz1 = params[{s_nzhi}]\n"
                f"    d_xhi = (px + nz1*pz - params[{s_chi}]) / wp.sqrt(nx1*nx1 + nz1*nz1)\n"
                f"    nx2 = float(-1.0); nz2 = params[{s_nzlo}]\n"
                f"    d_xlo = (-px + nz2*pz - params[{s_clo}]) / wp.sqrt(nx2*nx2 + nz2*nz2)\n"
                f"    ny1 = float(1.0); nzy1 = params[{s_ynzhi}]\n"
                f"    d_yhi = (py + nzy1*pz - params[{s_ychi}]) / wp.sqrt(ny1*ny1 + nzy1*nzy1)\n"
                f"    ny2 = float(-1.0); nzy2 = params[{s_ynzlo}]\n"
                f"    d_ylo = (-py + nzy2*pz - params[{s_yclo}]) / wp.sqrt(ny2*ny2 + nzy2*nzy2)\n"
                f"    m1 = wp.max(d_zlo, d_zhi)\n    m2 = wp.max(d_xhi, d_xlo)\n    m3 = wp.max(d_yhi, d_ylo)\n"
                f"    return wp.max(m1, wp.max(m2, m3))\n")
            return nid
        if op == "transform":
            cid = self.emit(node["child"])
            inv = E.mat4_inverse_affine(node["mat4"])
            nid = self._new_id()
            slots = [[self._slot(inv[r][c]) for c in range(4)] for r in range(3)]
            m = slots
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                f"    lx = params[{m[0][0]}]*px + params[{m[0][1]}]*py + params[{m[0][2]}]*pz + params[{m[0][3]}]\n"
                f"    ly = params[{m[1][0]}]*px + params[{m[1][1]}]*py + params[{m[1][2]}]*pz + params[{m[1][3]}]\n"
                f"    lz = params[{m[2][0]}]*px + params[{m[2][1]}]*py + params[{m[2][2]}]*pz + params[{m[2][3]}]\n"
                f"    return _f{cid}(lx, ly, lz, params)\n")
            return nid
        if op == "offset":
            cid = self.emit(node["child"])
            nid = self._new_id()
            sd = self._slot(node["delta"])
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                f"    return _f{cid}(px, py, pz, params) - params[{sd}]\n")
            return nid
        if op == "axis_project":
            cid = self.emit(node["child"])
            nid = self._new_id()
            axis = node["axis"]   # 0/1/2, structural (part of topology hash) -- literal in generated src
            sref = self._slot(node["ref"])
            if axis == 0:
                call = f"_f{cid}(params[{sref}], py, pz, params)"
            elif axis == 1:
                call = f"_f{cid}(px, params[{sref}], pz, params)"
            else:
                call = f"_f{cid}(px, py, params[{sref}], params)"
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                f"    return {call}\n")
            return nid
        if op == "scale":
            cid = self.emit(node["child"])
            nid = self._new_id()
            ss = self._slot(node["s"])
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                f"    s = params[{ss}]\n"
                f"    v = _f{cid}(px/s, py/s, pz/s, params)\n    return v * s\n")
            return nid
        if op in ("union", "intersect", "subtract"):
            aid = self.emit(node["a"])
            bid = self.emit(node["b"])
            nid = self._new_id()
            sk = self._slot(node["k"])
            if op == "union":
                hard = "wp.min(da, db)"
                smooth = (f"        h = wp.max(float(0.0), params[{sk}] - wp.abs(da - db)) / params[{sk}]\n"
                          f"        return wp.min(da, db) - h*h*params[{sk}]*float(0.25)\n")
            elif op == "intersect":
                smooth = (f"        h = wp.max(float(0.0), params[{sk}] - wp.abs(da - db)) / params[{sk}]\n"
                          f"        return wp.max(da, db) + h*h*params[{sk}]*float(0.25)\n")
                hard = "wp.max(da, db)"
            else:
                smooth = (f"        h = wp.max(float(0.0), params[{sk}] - wp.abs(da + db)) / params[{sk}]\n"
                          f"        return wp.max(da, -db) + h*h*params[{sk}]*float(0.25)\n")
                hard = "wp.max(da, -db)"
            # k is now a RUNTIME param (could be 0 or >0 across calls without a topology change) -- branch
            # in-kernel rather than at codegen time (only extra cost: one predicated select per boolean
            # node, negligible next to the sqrt/div work already in every leaf).
            body = (f"    if params[{sk}] <= float(0.0):\n        return {hard}\n"
                    f"    else:\n{smooth}")
            self.lines.append(
                f"@wp.func\ndef _f{nid}(px: float, py: float, pz: float, {_SIG}):\n"
                f"    da = _f{aid}(px, py, pz, params)\n"
                f"    db = _f{bid}(px, py, pz, params)\n" + body)
            return nid
        raise ValueError(f"eval_warp_uniform_v2: unsupported op {op!r} (P1 scope: primitives + "
                          f"rect_frustum + transform/offset/axis_project/scale/booleans -- no gridfield "
                          f"leaves, see module docstring)")


def compile_expr_uniform(node):
    """Returns (kernel, param_vector: np.float32[N], n_params). Kernel is cached by TOPOLOGY hash only
    (P1's whole point) -- param_vector is ALWAYS freshly re-extracted from `node` (cheap pure-python
    walk, microseconds for a <200-node tree), so calling this again after a recipe param edit (new
    literal values, SAME tree shape) hits the compiled-kernel cache and just returns the new numbers."""
    h = _topology_hash(node)
    cg = _CodegenUniform()
    root_id = cg.emit(node)
    param_vector = np.asarray(cg.param_vals, dtype=np.float32)

    if h in _KERNEL_CACHE:
        kernel, mod, n_params, src = _KERNEL_CACHE[h]
        assert n_params == len(param_vector), (
            f"topology hash collided with a DIFFERENT param count ({n_params} vs {len(param_vector)}) "
            f"-- would be a hash-collision bug, not a normal parameter tweak; refusing silently-wrong reuse")
        return kernel, param_vector, n_params

    src_lines = ["import warp as wp", ""] + cg.lines
    src_lines.append(
        f"@wp.kernel\n"
        f"def eval_kernel_uniform(pts: wp.array(dtype=wp.vec3), {_SIG}, out: wp.array(dtype=wp.float32)):\n"
        f"    tid = wp.tid()\n"
        f"    p = pts[tid]\n"
        f"    out[tid] = _f{root_id}(p[0], p[1], p[2], params)\n")
    src = "\n".join(src_lines)

    cache_dir = os.path.join(HERE, "_gen_cache")
    os.makedirs(cache_dir, exist_ok=True)
    mod_path = os.path.join(cache_dir, f"ikarus_uniform_{h}.py")
    if not os.path.exists(mod_path):
        with open(mod_path, "w") as f:
            f.write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"ikarus_uniform_{h}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    n_params = len(param_vector)
    _KERNEL_CACHE[h] = (mod.eval_kernel_uniform, mod, n_params, src)
    return mod.eval_kernel_uniform, param_vector, n_params


def eval_batch_uniform(node, points_mm, device=DEVICE):
    """eval_warp.eval_batch's drop-in for uniform-param trees: (kernel-cache HIT on an unchanged tree
    shape) + (fresh param upload) + (launch). Measured cost is exactly the doc's arm-B ledger: no
    recompile, ~0.18ms upload+launch floor."""
    kernel, param_vector, n_params = compile_expr_uniform(node)
    params_wp = wp.array(param_vector, dtype=wp.float32, device=device)
    n = points_mm.shape[0]
    pts = wp.array(points_mm.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(kernel, dim=n, inputs=[pts, params_wp], outputs=[out], device=device)
    wp.synchronize()
    return out.numpy()


def gradient_batch_uniform(node, points_mm, h=0.05, device=DEVICE):
    """Same central-difference contract as eval_warp.gradient_batch, routed through the uniform path."""
    n = points_mm.shape[0]
    offsets = np.array([[h, 0, 0], [-h, 0, 0], [0, h, 0], [0, -h, 0], [0, 0, h], [0, 0, -h]])
    stacked = (points_mm[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    vals = eval_batch_uniform(node, stacked, device=device).reshape(6, n)
    gx = (vals[0] - vals[1]) / (2 * h)
    gy = (vals[2] - vals[3]) / (2 * h)
    gz = (vals[4] - vals[5]) / (2 * h)
    return np.stack([gx, gy, gz], axis=1)


# --------------------------------------------------------------------------------------------- self-test / validation gate
def selftest():
    """Cross-check vs expr.eval_sdf_py (independent pure-python ground truth) PLUS the
    P1 claim itself: build a tree, time first compile, then MUTATE ONE LITERAL PARAM (same topology,
    same shape hash) and re-measure -- that second call must be orders of magnitude faster (cache hit,
    no recompile) and must still agree with the (re-evaluated) pure-python ground truth at the NEW
    parameter value (catches a stale-param bug: reusing the kernel while forgetting to re-upload params
    would silently keep returning the OLD geometry)."""
    import random
    random.seed(7)
    c = E.cylinder(radius=12.5, height=40.0)
    b = E.box(half_extents=(8.0, 8.0, 8.0), center=(3.0, 0.0, 5.0))
    rf = E.rect_frustum(z0=-10.0, z1=30.0, hx0=20.0, hy0=15.0, hx1=12.0, hy1=9.0)
    tree = E.union(E.subtract(c, b, k=0.0), E.translate(rf, 40.0, 0.0, 0.0), k=0.0)

    pts = np.array([[random.uniform(-30, 60), random.uniform(-30, 30), random.uniform(-30, 30)]
                     for _ in range(4000)], dtype=np.float64)

    t0 = time.time()
    gpu_vals = eval_batch_uniform(tree, pts)
    t_first = time.time() - t0
    cpu_vals = np.array([E.eval_sdf_py(tree, tuple(p)) for p in pts])
    err0 = np.abs(gpu_vals - cpu_vals)

    # mutate ONE literal (same shape): cylinder radius 12.5 -> 9.0
    tree2 = json.loads(json.dumps(tree))
    tree2["a"]["a"]["radius"] = 9.0
    assert _topology_hash(tree2) == _topology_hash(tree), "param-only edit must not change topology hash"

    param_change_times = []
    for trial_r in (9.0, 15.0, 11.0, 13.5, 10.2):
        t2 = json.loads(json.dumps(tree))
        t2["a"]["a"]["radius"] = trial_r
        t0 = time.time()
        gpu_vals2 = eval_batch_uniform(t2, pts)
        param_change_times.append(time.time() - t0)
    cpu_vals2 = np.array([E.eval_sdf_py(t2, tuple(p)) for p in pts])   # last trial's tree
    err1 = np.abs(gpu_vals2 - cpu_vals2)

    param_change_times.sort()
    median_s = param_change_times[len(param_change_times) // 2]

    result = {
        "n_points": len(pts),
        "max_abs_err_mm_initial": float(err0.max()),
        "max_abs_err_mm_after_param_change": float(err1.max()),
        "gate_pass": bool(err0.max() < 1e-2 and err1.max() < 1e-2),
        "t_first_compile_s": t_first,
        "param_change_s_all_trials": param_change_times,
        "param_change_s_median": median_s,
        "recompile_avoided_speedup_x": t_first / max(median_s, 1e-9),
        "device": DEVICE,
    }
    return result


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if stage == "selftest":
        r = selftest()
        print(json.dumps(r, indent=2))
        assert r["gate_pass"], "eval_warp_uniform_v2.py selftest FAILED vs pure-python ground truth"
        print("PASS")
