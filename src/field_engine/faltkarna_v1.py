#!/usr/bin/env python3
"""Block-sparse GPU-resident SDF field kernel.

Structure (two-level, NanoVDB class): the domain is split into blocks of BLOCK^3 voxels. Each block is
classified by a Lipschitz-safe 8-corner evaluation (a signed distance field is 1-Lipschitz under
min/max CSG, so a block whose 8 corners all share a sign and whose smallest |value| exceeds the block
space diagonal sqrt(3)*BLOCK*pitch cannot contain a zero crossing anywhere in its volume):
  -1 INTERIOR   (background constant -BIG, no tile stored)
  +1 EXTERIOR   (background constant +BIG, no tile stored)
   0 NEAR-SURFACE (a full BLOCK^3 tile of exact per-voxel values is stored)
Memory therefore scales with interface area, not volume.

Operations:
  klassificera_och_evaluera_testfalt(...)  rasterise an analytic field into the sparse structure
                                           (two Warp passes: classify, then evaluate active tiles)
  csg_union/csg_intersect/csg_subtract     min/max CSG (union=min, intersect=max,
                                           subtract(A,B)=max(A,-B))
  rekonstruera_tat(...) / rekonstruera_tat_gpu(...)
                                           scatter back to a dense array for marching cubes (export only)
  konservering_check(...)                  volume delta against an analytic reference, inside a declared band
  determinism_cert(...)                    N runs of the same operation must be bit-identical

Inputs are grid metadata plus analytic field parameters; outputs are a SparseField (block kinds, active
block ids, exact tiles) and the gate dictionaries above. Requires a Warp device; the GPU kernels have no
numpy fallback.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import warp as wp
    wp.init()
    HAVE_CUDA = "cuda:0" in [str(d) for d in wp.get_devices()]
except Exception as e:  # pragma: no cover
    wp = None
    HAVE_CUDA = False
    print("faltkarna_v1: warp unavailable, there is no numpy fallback for the GPU kernels:", e)

BIG = 1.0e6           # background constant (INTERIOR = -BIG, EXTERIOR = +BIG), larger than any mm scale here
FIXPT_SCALE = 1.0e6    # int64 fixed-point scale for determinism-safe accumulation (mm^3 -> integer)


@dataclass
class GridMeta:
    x0: float; y0: float; z0: float
    pitch: float
    nx: int; ny: int; nz: int
    block: int = 8

    def __post_init__(self):
        self.n_bx = -(-self.nx // self.block)
        self.n_by = -(-self.ny // self.block)
        self.n_bz = -(-self.nz // self.block)
        self.n_blocks = self.n_bx * self.n_by * self.n_bz
        # Lipschitz margin: the full space diagonal of a block, the provably safe condition for
        # "no zero crossing possible inside this block" for a 1-Lipschitz SDF.
        self.margin_mm = math.sqrt(3.0) * self.block * self.pitch


@dataclass
class SparseField:
    meta: GridMeta
    kind: np.ndarray            # (n_blocks,) int32: -1/0/+1
    active_ids: np.ndarray      # (n_active,) int32 block-linear-id, kind==0
    tiles: np.ndarray           # (n_active, block,block,block) float32, exact values
    corner_min_abs: np.ndarray  # (n_blocks,) float32, classification diagnostic
    stats: dict = field(default_factory=dict)


def _block_ijk(lin, meta: GridMeta):
    n_by, n_bz = meta.n_by, meta.n_bz
    bi = lin // (n_by * n_bz)
    bj = (lin // n_bz) % n_by
    bk = lin % n_bz
    return bi, bj, bk


# ============================================================================================
# Generic min/max CSG primitives (wp.func), Hart/Bloomenthal convention (< 0 = inside).
# Same formulas as ikarus_v1/expr.py and the pocket SDF in tillverkningsfalt_v1_kernel.py.
# ============================================================================================
if wp is not None:
    @wp.func
    def sdf_box(x: wp.float32, y: wp.float32, z: wp.float32,
                cx: wp.float32, cy: wp.float32, cz: wp.float32,
                hx: wp.float32, hy: wp.float32, hz: wp.float32) -> wp.float32:
        qx = wp.abs(x - cx) - hx
        qy = wp.abs(y - cy) - hy
        qz = wp.abs(z - cz) - hz
        outside = wp.sqrt(wp.max(qx, 0.0) ** 2.0 + wp.max(qy, 0.0) ** 2.0 + wp.max(qz, 0.0) ** 2.0)
        inside = wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)
        return outside + inside

    @wp.func
    def sdf_cyl_axis_z(x: wp.float32, y: wp.float32, cx: wp.float32, cy: wp.float32,
                        r: wp.float32) -> wp.float32:
        return wp.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) - r

    @wp.func
    def csg_union(a: wp.float32, b: wp.float32) -> wp.float32:
        return wp.min(a, b)

    @wp.func
    def csg_intersect(a: wp.float32, b: wp.float32) -> wp.float32:
        return wp.max(a, b)

    @wp.func
    def csg_subtract(a: wp.float32, b: wp.float32) -> wp.float32:
        return wp.max(a, -b)

    # ------------------------------------------------------------------ test field (box minus cyl)
    # Used by planted-fault test (iii) and by the generic self-test; its exact volume is known
    # analytically.
    @wp.func
    def testfalt(x: wp.float32, y: wp.float32, z: wp.float32,
                 hx: wp.float32, hy: wp.float32, hz: wp.float32, r: wp.float32) -> wp.float32:
        b = sdf_box(x, y, z, 0.0, 0.0, 0.0, hx, hy, hz)
        c = sdf_cyl_axis_z(x, y, 0.0, 0.0, r)
        return csg_subtract(b, c)

    @wp.kernel
    def k_classify_testfalt(n_bx: int, n_by: int, n_bz: int, block: int,
                             x0: float, y0: float, z0: float, pitch: float,
                             hx: float, hy: float, hz: float, r: float,
                             margin: float,
                             kind: wp.array(dtype=wp.int32),
                             corner_min_abs: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        bk = tid % n_bz
        bj = (tid // n_bz) % n_by
        bi = tid // (n_bz * n_by)
        ox = x0 + float(bi * block) * pitch
        oy = y0 + float(bj * block) * pitch
        oz = z0 + float(bk * block) * pitch
        side = float(block) * pitch
        min_abs = wp.float32(1.0e30)
        pos_seen = False
        neg_seen = False
        for c in range(8):
            dx = float(c & 1) * side
            dy = float((c >> 1) & 1) * side
            dz = float((c >> 2) & 1) * side
            v = testfalt(ox + dx, oy + dy, oz + dz, hx, hy, hz, r)
            min_abs = wp.min(min_abs, wp.abs(v))
            if v >= 0.0:
                pos_seen = True
            else:
                neg_seen = True
        corner_min_abs[tid] = min_abs
        if pos_seen and neg_seen:
            kind[tid] = 0
        elif min_abs <= margin:
            kind[tid] = 0
        elif neg_seen:
            kind[tid] = -1
        else:
            kind[tid] = 1

    @wp.kernel
    def k_eval_tile_testfalt(active_ids: wp.array(dtype=wp.int32), n_by: int, n_bz: int, block: int,
                              x0: float, y0: float, z0: float, pitch: float,
                              hx: float, hy: float, hz: float, r: float,
                              corrupt_index: int,
                              out: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        b3 = block * block * block
        a_idx = tid // b3
        local = tid % b3
        lin = active_ids[a_idx]
        if corrupt_index != 0:
            # Planted fault (i): corrupt the block indexing on purpose, so material lands at the
            # wrong world position; the conservation check (volume vs analytic reference) must fail.
            lin = lin + 1
        bk = lin % n_bz
        bi = lin // (n_bz * n_by)
        bj = (lin // n_bz) % n_by
        li = local // (block * block)
        lj = (local // block) % block
        lk = local % block
        wx = x0 + float(bi * block + li) * pitch
        wy = y0 + float(bj * block + lj) * pitch
        wz = z0 + float(bk * block + lk) * pitch
        out[tid] = testfalt(wx, wy, wz, hx, hy, hz, r)

    @wp.kernel
    def k_fill_background(kind: wp.array(dtype=wp.int32), n_by: int, n_bz: int, block: int,
                           nx: int, ny: int, nz: int, big: float,
                           out: wp.array(dtype=wp.float32)):
        """Fill the padded dense array with the per-block background constant, one thread per voxel."""
        tid = wp.tid()
        b3 = block * block * block
        block_lin = tid // b3
        local = tid % b3
        bk = block_lin % n_bz
        bi = block_lin // (n_bz * n_by)
        bj = (block_lin // n_bz) % n_by
        li = local // (block * block)
        lj = (local // block) % block
        lk = local % block
        vx = bi * block + li
        vy = bj * block + lj
        vz = bk * block + lk
        # the padded array is (n_bx*block, n_by*block, n_bz*block); flat C-order index:
        idx = (vx * ny + vy) * nz + vz
        if kind[block_lin] < 0:
            out[idx] = -big
        else:
            out[idx] = big

    @wp.kernel
    def k_scatter_tiles(active_ids: wp.array(dtype=wp.int32), tiles: wp.array(dtype=wp.float32),
                         n_by: int, n_bz: int, block: int, ny: int, nz: int,
                         out: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        b3 = block * block * block
        a_idx = tid // b3
        local = tid % b3
        lin = active_ids[a_idx]
        bk = lin % n_bz
        bi = lin // (n_bz * n_by)
        bj = (lin // n_bz) % n_by
        li = local // (block * block)
        lj = (local // block) % block
        lk = local % block
        vx = bi * block + li
        vy = bj * block + lj
        vz = bk * block + lk
        idx = (vx * ny + vy) * nz + vz
        out[idx] = tiles[tid]

    @wp.kernel
    def k_reduce_checksum_f32_unsafe(vals: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        # Summing the actual SDF values (varying magnitude) is the sharp case: float32 atomic_add
        # is order-dependent and jitters across separate processes, while the int64 fixed-point
        # accumulation below is bit-identical. Accumulating a uniform constant per thread does not
        # expose the jitter and would give a false-negative self-test.
        tid = wp.tid()
        v = vals[tid]
        if v < 0.0:
            wp.atomic_add(out, 0, v)

    @wp.kernel
    def k_reduce_checksum_i64_safe(vals: wp.array(dtype=wp.float32), scale: float,
                                    out: wp.array(dtype=wp.int64)):
        tid = wp.tid()
        v = vals[tid]
        if v < 0.0:
            wp.atomic_add(out, 0, wp.int64(wp.float64(v) * wp.float64(scale)))


def klassificera_och_evaluera_testfalt(meta: GridMeta, hx, hy, hz, r, corrupt_index=False,
                                        device="cuda:0" if HAVE_CUDA else "cpu") -> SparseField:
    """Rasterise the test field (box minus cylinder) into a SparseField.

    Two Warp passes: block classification, then exact tile evaluation over the compacted active list.
    corrupt_index=True injects planted fault (i). Returns a SparseField.
    """
    assert wp is not None, "warp is required; the GPU kernels have no CPU fallback"
    with wp.ScopedDevice(device):
        kind_w = wp.zeros(meta.n_blocks, dtype=wp.int32)
        cma_w = wp.zeros(meta.n_blocks, dtype=wp.float32)
        wp.launch(k_classify_testfalt, dim=meta.n_blocks,
                  inputs=[meta.n_bx, meta.n_by, meta.n_bz, meta.block,
                          meta.x0, meta.y0, meta.z0, meta.pitch,
                          hx, hy, hz, r, meta.margin_mm],
                  outputs=[kind_w, cma_w])
        wp.synchronize()
        kind = kind_w.numpy()
        cma = cma_w.numpy()
        active_ids = np.where(kind == 0)[0].astype(np.int32)
        n_active = len(active_ids)
        b3 = meta.block ** 3
        if n_active > 0:
            active_w = wp.array(active_ids, dtype=wp.int32)
            out_w = wp.zeros(n_active * b3, dtype=wp.float32)
            wp.launch(k_eval_tile_testfalt, dim=n_active * b3,
                      inputs=[active_w, meta.n_by, meta.n_bz, meta.block,
                              meta.x0, meta.y0, meta.z0, meta.pitch,
                              hx, hy, hz, r, 1 if corrupt_index else 0],
                      outputs=[out_w])
            wp.synchronize()
            tiles = out_w.numpy().reshape(n_active, meta.block, meta.block, meta.block)
        else:
            tiles = np.zeros((0, meta.block, meta.block, meta.block), dtype=np.float32)
    n_interior = int(np.sum(kind == -1))
    n_exterior = int(np.sum(kind == 1))
    stats = dict(n_blocks=int(meta.n_blocks), n_active=int(n_active),
                 n_interior=n_interior, n_exterior=n_exterior,
                 active_voxels=int(n_active * b3), dense_voxels=int(meta.nx * meta.ny * meta.nz),
                 margin_mm=meta.margin_mm)
    return SparseField(meta=meta, kind=kind, active_ids=active_ids, tiles=tiles,
                        corner_min_abs=cma, stats=stats)


def rekonstruera_tat(sf: SparseField) -> np.ndarray:
    """Build a dense array from a SparseField (export only).

    Scatters the exact near-surface tiles and fills the background constant (-BIG/+BIG) for
    interior/exterior blocks. The zero crossing is guaranteed to lie inside near-surface blocks
    (see the Lipschitz margin). Fully vectorised: pad to a block multiple, reshape to
    (n_bx,n_by,n_bz,B,B,B) and fancy-index-assign all active blocks in one call.
    Returns a dense (nx,ny,nz) float32 array.
    """
    meta = sf.meta
    b = meta.block
    px, py, pz = meta.n_bx * b, meta.n_by * b, meta.n_bz * b
    kind_grid = sf.kind.reshape(meta.n_bx, meta.n_by, meta.n_bz)
    background = np.where(kind_grid < 0, -BIG, BIG).astype(np.float32)
    padded = np.empty((px, py, pz), dtype=np.float32)
    block_view = padded.reshape(meta.n_bx, b, meta.n_by, b, meta.n_bz, b).transpose(0, 2, 4, 1, 3, 5)
    block_view[:] = background[:, :, :, None, None, None]
    if len(sf.active_ids) > 0:
        # block_view is a true view (reshape+transpose of a contiguous array never copies), so
        # fancy-index assignment on the first three axes writes straight into padded's memory.
        # Note: .reshape() on the transposed view would force a silent copy.
        bi, bj, bk = np.unravel_index(sf.active_ids, (meta.n_bx, meta.n_by, meta.n_bz))
        block_view[bi, bj, bk] = sf.tiles
    return padded[:meta.nx, :meta.ny, :meta.nz]


def rekonstruera_tat_gpu(sf: SparseField, device="cuda:0" if HAVE_CUDA else "cpu") -> np.ndarray:
    """Same as rekonstruera_tat but GPU-resident (two Warp scatter kernels).

    The only host transfer is the final .numpy() copy, required by skimage marching_cubes.
    Returns a dense (nx,ny,nz) float32 array.
    """
    meta = sf.meta
    b = meta.block
    pnx, pny, pnz = meta.n_bx * b, meta.n_by * b, meta.n_bz * b
    b3 = b ** 3
    with wp.ScopedDevice(device):
        out_w = wp.zeros(pnx * pny * pnz, dtype=wp.float32)
        kind_w = wp.array(sf.kind.astype(np.int32), dtype=wp.int32)
        wp.launch(k_fill_background, dim=meta.n_blocks * b3,
                  inputs=[kind_w, meta.n_by, meta.n_bz, b, pnx, pny, pnz, BIG], outputs=[out_w])
        if len(sf.active_ids) > 0:
            active_w = wp.array(sf.active_ids.astype(np.int32), dtype=wp.int32)
            tiles_w = wp.array(sf.tiles.reshape(-1).astype(np.float32), dtype=wp.float32)
            wp.launch(k_scatter_tiles, dim=len(sf.active_ids) * b3,
                      inputs=[active_w, tiles_w, meta.n_by, meta.n_bz, b, pny, pnz], outputs=[out_w])
        wp.synchronize()
        padded = out_w.numpy().reshape(pnx, pny, pnz)
    return padded[:meta.nx, :meta.ny, :meta.nz]


def konservering_check(volym_matt_mm3: float, volym_facit_mm3: float, tol_frac: float) -> dict:
    diff_frac = abs(volym_matt_mm3 - volym_facit_mm3) / max(abs(volym_facit_mm3), 1e-9)
    return dict(volym_matt_mm3=volym_matt_mm3, volym_facit_mm3=volym_facit_mm3,
                diff_frac=diff_frac, tol_frac=tol_frac, GRON=bool(diff_frac <= tol_frac))


def determinism_cert(varden: list) -> dict:
    """Gate hook: N runs of the same operation must give a bit-identical result.

    Takes the list of per-run values, returns a dict with n_korningar, varden and bit_identiska.
    """
    identiska = all(v == varden[0] for v in varden)
    return dict(n_korningar=len(varden), varden=varden, bit_identiska=bool(identiska))


def reduktion_checksum_single_process(tiles_npy_path: str, device="cuda:0" if HAVE_CUDA else "cpu"):
    """One run, meant to be called as a standalone subprocess, of both the float32-unsafe and the
    int64 fixed-point SDF checksum over material<0 voxels.

    Takes the path of a .npy of tile values, returns (float32_sum, int64_fixedpoint_sum).
    """
    vals = np.load(tiles_npy_path).reshape(-1).astype(np.float32)
    with wp.ScopedDevice(device):
        vals_w = wp.array(vals, dtype=wp.float32)
        out32 = wp.zeros(1, dtype=wp.float32)
        wp.launch(k_reduce_checksum_f32_unsafe, dim=len(vals), inputs=[vals_w], outputs=[out32])
        wp.synchronize()
        r32 = float(out32.numpy()[0])

        out64 = wp.zeros(1, dtype=wp.int64)
        wp.launch(k_reduce_checksum_i64_safe, dim=len(vals), inputs=[vals_w, FIXPT_SCALE], outputs=[out64])
        wp.synchronize()
        r64 = float(out64.numpy()[0]) / FIXPT_SCALE
    return r32, r64


def volym_via_gpu_reduktion(sf: SparseField, n_korningar=6, tmp_dir="/tmp"):
    """Planted fault (ii): the SDF checksum computed in two modes, float32 without fixed point
    (non-deterministic across separate processes) and int64 fixed point (bit-identical).

    Runs n_korningar standalone subprocesses (a loop inside one warm process does not expose the
    jitter). Returns a dict with a determinism_cert for each mode.
    """
    import subprocess
    import sys
    import tempfile
    tiles_path = os.path.join(tmp_dir, "faltkarna_v1_reduktion_tiles.npy")
    np.save(tiles_path, sf.tiles.astype(np.float32))
    here = os.path.dirname(os.path.abspath(__file__))
    f32_totals, i64_totals = [], []
    for _ in range(n_korningar):
        r = subprocess.run([sys.executable, "-c",
                             "import sys; sys.path.insert(0,%r); import faltkarna_v1 as FK; "
                             "r32,r64 = FK.reduktion_checksum_single_process(%r); "
                             "print(repr(r32), repr(r64))" % (here, tiles_path)],
                            capture_output=True, text=True, cwd=here, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"subprocess reduction failed: {r.stderr[-2000:]}")
        parts = r.stdout.strip().splitlines()[-1].split()
        f32_totals.append(float(parts[0]))
        i64_totals.append(float(parts[1]))
    return dict(f32_unsafe=determinism_cert(f32_totals), i64_fixpunkt_safe=determinism_cert(i64_totals),
                n_active_voxels=int(sf.tiles.size), n_processer=n_korningar)
