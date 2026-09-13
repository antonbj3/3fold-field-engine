#!/usr/bin/env python3
"""Mesh to SDF: turn a triangle mesh into a signed field and a sparse block-classified occupancy.

Pipeline per part:
  1. inside/outside per voxel centre, by default from a +Z ray winding number against the triangle
     soup (see the block comment below for why this replaced the earlier flood fill);
  2. signed euclidean distance from two distance transforms of the solid mask itself,
     sd = EDT(~solid) - EDT(solid). Taking the transform against the surface-point mask instead gives
     a surface voxel distance exactly 0.0 to a mask it is itself part of, so sd becomes -0.0, and
     -0.0 < 0.0 is false in IEEE754: for a thin part whose whole body is surface voxels every material
     voxel is then misclassified as void. The correct construction gives any solid voxel with a
     non-solid neighbour a strictly negative value (at least one pitch), so the sparse substrate's
     exact values are real mm distances;
  3. the generic block classifier: the same Lipschitz-safe 8-corner logic as the analytic test field,
     but reading a precomputed dense array instead of an analytic formula;
  4. margin dilation on the reconstructed occupancy.

A watertightness gate runs independently of the signing and either flags or raises (see the block
comment at vattentathetsgrind). Main entry point: mesh_to_sdf_del(wp, V, T, pitch, margin_mm, lo,
device) -> dict with the local window origin, shape, final occupancy, the sparse field counts, the
signing diagnostics and the gate verdict.

Run `python faltkarna_v1_mesh_to_sdf.py` for a self-test on analytic bodies.
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BLOCK = 8


def _import_warp():
    import warp as wp
    wp.init()
    have_cuda = "cuda:0" in [str(d) for d in wp.get_devices()]
    return wp, have_cuda


# ================================================================================================
# Steps 1-3: inside/outside classification and the signed distance transform (host, numpy/scipy).
# ================================================================================================
# Why the default is a ray winding number and not the shell flood fill.
#
# The old path (metod="skal_floodfill" below) rasterises the surface as points with np.rint() and
# flood-fills with the 6-neighbourhood of ndimage.label. A closed analytic 60x40x30 mm box came out
# hollow: the solid mask was bit-identical to the surface mask (74% of the volume missing at 2 mm
# pitch, 95% at 1 mm), and three real parts lost 49/19/94% against their mesh volume. Mechanism: the
# point raster is sparse. Subdividing to pitch/2 density plus rint does not guarantee a 6-separating
# shell, and on axis-aligned faces the points land on exact half voxel steps where rint rounds to
# even. The outside label then leaks straight through the shell and what is left as interior is empty.
# A regression gate that compared the device path against the same algorithm on the host could not
# catch this: the reference was as hollow as the thing it judged, bit-identical and both wrong.
#
# The fix, correct first and fast second: no flood fill. Inside/outside is decided geometrically per
# voxel centre by a +Z ray against the triangle soup (metod="raypar_vindning", the default):
#   * each column (i,j) of the grid is one ray, and each triangle contributes only to the columns
#     inside its XY bounding box, so the work is the sum of the triangles' column boxes, not
#     nx*ny*n_triangles;
#   * the hit is solved in 2D barycentric coordinates on the XY projection and z is interpolated
#     exactly; the sign of the 2D signed area is the sign of the triangle normal's z component;
#   * regel="vindning" (default): a nonzero sum of hit normal signs above the point means inside.
#     That is the ray winding number, and it is correct for nested and overlapping bodies too (one
#     test part consists of 7934 separate bodies). Plain even/odd parity cancels two overlapping
#     bodies and would leave a hole where they meet;
#   * regel="paritet" remains as an independent anchor (same hits, different rule) and the difference
#     between the two is always reported in the diagnostics: two methods that agree on a
#     well-oriented closed mesh and differ when the normals are broken;
#   * half-open convention in z: hits with z_hit <= z_centre do not count as above, so a surface
#     lying exactly in a voxel centre gets a deterministic answer rather than a coin flip on
#     floating-point noise;
#   * XY ties (a voxel centre exactly on an edge shared by two triangles, common on axis-aligned
#     boxes) are broken with a deterministic subvoxel jitter of the ray's XY position, 1e-4 * pitch
#     along a golden-ratio direction. The classification is therefore "the point displaced
#     infinitesimally": a declared choice, identical between runs.
# The signed distance is then built with two distance transforms of the corrected solid mask,
# sd = EDT(~solid) - EDT(solid). Downstream semantics are unchanged (sd < 0 is material).
#
# A second path is implemented and measured: metod="winding_gpu" reads sd directly from Warp's
# mesh_query_point_sign_winding_number (BVH plus generalised winding number), giving exact mm
# distances instead of a voxel-quantised transform, and no distance transform at all. It needs a GPU.
# The default is the host path because it is deterministic without a GPU and because the signing also
# runs in worker processes without a CUDA context.
#
# The old path is kept behind metod="skal_floodfill" (env FALTKARNA_SDF_METOD) so a benchmark can
# show before and after in one run.
# ================================================================================================
METOD_STANDARD = os.environ.get("FALTKARNA_SDF_METOD", "raypar_vindning")
METODER = ("raypar_vindning", "raypar_paritet", "winding_gpu", "skal_floodfill")
_GYLLENE = (0.3819660112501051, 0.6180339887498949, 0.2360679774997897)  # jitter direction
JITTER_FRAC = 1.0e-4   # fraction of a pitch; subvoxel, breaks XY ties without moving the geometry
# Sample position inside a voxel: 0.0 means voxel i represents the point lo + i*pitch, not the cell
# centre lo + (i+0.5)*pitch. Downstream consumers read the array with the same convention (marching
# cubes with spacing=pitch places the surface at index*pitch, and analytic masks are centred on an
# index), so using the cell centre introduces a systematic half-voxel shift against all of them
# (measured: 2.44% mask deviation against an analytic circle anchor instead of 0.05%). The half-open
# z rule plus the XY jitter keep the point convention exact even when a surface lies exactly in a
# sample point (measured: axis-aligned box, relative error 0.000).
VOXEL_PROVPUNKT = 0.0
# Sub-voxel surface correction of the distance-transform field.
# sd = EDT(~solid) - EDT(solid) measures the distance from a voxel to the nearest voxel of the other
# class, not to the surface between them. The nearest such voxel is one pitch away, while the surface
# it stands for lies halfway, so every value is one half pitch too large in magnitude, everywhere and
# in the same direction. Measured on the hole plate: sampling the field exactly on a hole wall, where
# the true signed distance is 0, gives max |sd| = 1.000 * pitch at every radius and every pitch
# (1.0 mm at pitch 1.0, 0.5 mm at pitch 0.5), while the zero crossing of the same field along the
# radius lies only 0.5 * pitch off the true wall -- the level set is in the right place, the values
# around it are twice too large. sd - sign(sd) * pitch/2 removes exactly that offset: it moves no
# zero crossing (the crossing of the interpolated field is unchanged), it flips no sign (|sd| >= pitch
# for every voxel of the transform), so the delivered occupancy is bit-identical, and it makes
# |sd| / R a measure of the geometry instead of a measure of the pitch.
# It is on by default: the delivered occupancy is provably unchanged (for any voxel one of the two
# transforms is 0 and the other at least one pitch, so |sd| >= pitch > pitch/2 and no sign can flip),
# and the module's own selftest reproduces every number bit-identically with it on and off. The one
# thing it does change is the block classifier's census, which may call more blocks active for the
# same Lipschitz margin -- conservative, never wrong, not free. Set FALTKARNA_SDF_YTKORREKTION=0 or
# pass ytkorrektion=False to get the raw transform back.
YTKORREKTION_STANDARD = os.environ.get("FALTKARNA_SDF_YTKORREKTION", "1") == "1"


def _fonster(V, pitch, lo, global_shape=None, margin_vox_pad=2):
    """Local grid window in global grid indices, clamped against the global window when one is given.

    Takes the vertices, the pitch, the grid origin and optionally the global shape; returns
    (gmin, shape_l).
    """
    Vf = np.asarray(V, dtype=np.float64)
    gmin = np.floor((Vf.min(0) - lo) / pitch).astype(np.int64) - margin_vox_pad
    gmax = np.ceil((Vf.max(0) - lo) / pitch).astype(np.int64) + margin_vox_pad + 1
    if global_shape is not None:
        gmin = np.maximum(gmin, 0)
        gmax = np.minimum(gmax, np.array(global_shape, dtype=np.int64))
    gmax = np.maximum(gmax, gmin)
    return gmin, tuple(int(x) for x in (gmax - gmin))


def _kolumnbatchar(i0, i1, j0, j1, cap):
    """Batches the (triangle -> column box) pairs so no intermediate exceeds `cap` entries.

    Triangles are sorted by box size, which bounds the waste inside a batch.
    """
    ext_i = i1 - i0 + 1
    ext_j = j1 - j0 + 1
    idx = np.where((ext_i > 0) & (ext_j > 0))[0]
    if len(idx) == 0:
        return
    kost = (ext_i[idx] * ext_j[idx]).astype(np.int64)
    ordn = np.argsort(kost, kind="stable")
    idx, kost = idx[ordn], kost[ordn]
    n = len(idx)
    start = 0
    while start < n:
        slut = min(n, start + max(1, int(cap // max(int(kost[start]), 1))))
        while slut > start + 1 and (slut - start) * int(kost[slut - 1]) > cap:
            slut = start + max(1, int(cap // max(int(kost[slut - 1]), 1)))
        yield idx[start:slut]
        start = slut


def _kolumntraffar(V, T, pitch, origin_l, nx, ny, jitter_frac=JITTER_FRAC, cap=3_000_000):
    """All intersections between the grid's +Z rays (one per column, through the sample point plus
    jitter) and the triangle soup.

    Returns (column_linear_index, z_hit, sign_of_nz) as three 1D arrays. Triangles whose XY projection
    is degenerate (faces parallel to Z) are skipped: a +Z ray cannot pass through them.
    """
    Vf = np.asarray(V, dtype=np.float64)
    Ti = np.asarray(T, dtype=np.int64)
    A, B, C = Vf[Ti[:, 0]], Vf[Ti[:, 1]], Vf[Ti[:, 2]]
    a2 = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    skala = max(float(np.abs(Vf[:, :2]).max()), 1.0)
    lev = np.abs(a2) > 1e-12 * skala * skala
    A, B, C, a2 = A[lev], B[lev], C[lev], a2[lev]
    tom = (np.zeros(0, np.int64), np.zeros(0, np.float64), np.zeros(0, np.int8))
    if len(a2) == 0:
        return tom
    jx = pitch * jitter_frac * _GYLLENE[0]
    jy = pitch * jitter_frac * _GYLLENE[1]
    xmin = np.minimum.reduce([A[:, 0], B[:, 0], C[:, 0]])
    xmax = np.maximum.reduce([A[:, 0], B[:, 0], C[:, 0]])
    ymin = np.minimum.reduce([A[:, 1], B[:, 1], C[:, 1]])
    ymax = np.maximum.reduce([A[:, 1], B[:, 1], C[:, 1]])
    # column i carries the ray x = origin_l[0] + (i+0.5)*pitch + jx
    i0 = np.clip(np.ceil((xmin - jx - origin_l[0]) / pitch - VOXEL_PROVPUNKT).astype(np.int64), 0, nx)
    i1 = np.clip(np.floor((xmax - jx - origin_l[0]) / pitch - VOXEL_PROVPUNKT).astype(np.int64), -1, nx - 1)
    j0 = np.clip(np.ceil((ymin - jy - origin_l[1]) / pitch - VOXEL_PROVPUNKT).astype(np.int64), 0, ny)
    j1 = np.clip(np.floor((ymax - jy - origin_l[1]) / pitch - VOXEL_PROVPUNKT).astype(np.int64), -1, ny - 1)
    tecken = np.sign(a2).astype(np.int8)
    ut_kol, ut_z, ut_tk = [], [], []
    for sel in _kolumnbatchar(i0, i1, j0, j1, cap):
        ei = int((i1[sel] - i0[sel] + 1).max())
        ej = int((j1[sel] - j0[sel] + 1).max())
        di, dj = np.meshgrid(np.arange(ei, dtype=np.int64), np.arange(ej, dtype=np.int64), indexing="ij")
        di, dj = di.ravel(), dj.ravel()
        ii = i0[sel][:, None] + di[None, :]
        jj = j0[sel][:, None] + dj[None, :]
        giltig = (ii <= i1[sel][:, None]) & (jj <= j1[sel][:, None])
        tt = np.broadcast_to(np.arange(len(sel), dtype=np.int64)[:, None], ii.shape)[giltig]
        ii, jj = ii[giltig], jj[giltig]
        if len(ii) == 0:
            continue
        s = sel[tt]
        px = origin_l[0] + (ii + VOXEL_PROVPUNKT) * pitch + jx
        py = origin_l[1] + (jj + VOXEL_PROVPUNKT) * pitch + jy
        ax, ay = A[s, 0], A[s, 1]
        bx, by = B[s, 0], B[s, 1]
        cx, cy = C[s, 0], C[s, 1]
        l0 = (bx - px) * (cy - py) - (by - py) * (cx - px)   # weight for A
        l1 = (cx - px) * (ay - py) - (cy - py) * (ax - px)   # weight for B
        l2 = (ax - px) * (by - py) - (ay - py) * (bx - px)   # weight for C
        d = a2[s]
        sgn = np.sign(d)
        inne = (l0 * sgn >= 0) & (l1 * sgn >= 0) & (l2 * sgn >= 0)
        if not inne.any():
            continue
        l0, l1, l2, d = l0[inne], l1[inne], l2[inne], d[inne]
        s_in = s[inne]
        z = (l0 * A[s_in, 2] + l1 * B[s_in, 2] + l2 * C[s_in, 2]) / d
        ut_kol.append(ii[inne] * ny + jj[inne])
        ut_z.append(z)
        ut_tk.append(tecken[s_in])
    if not ut_kol:
        return tom
    return (np.concatenate(ut_kol), np.concatenate(ut_z), np.concatenate(ut_tk))


def solid_via_stralvindning(V, T, pitch, origin_l, shape_l, regel="vindning",
                            jitter_frac=JITTER_FRAC, chunk_vox=2_000_000, cap=3_000_000):
    """Inside/outside per voxel from the +Z ray hits.

    Takes the mesh, the pitch, the local origin and shape and the rule ("vindning" or "paritet");
    returns (solid_bool[shape_l], diagnostics).
    """
    nx, ny, nz = (int(shape_l[0]), int(shape_l[1]), int(shape_l[2]))
    solid = np.zeros((nx, ny, nz), dtype=bool)
    diag = dict(metod="raypar", regel=regel, jitter_frac=jitter_frac, n_traffar=0,
                n_kolumner_med_traff=0, n_vindning_vs_paritet_diff=None,
                andel_vindning_vs_paritet=None)
    if nx == 0 or ny == 0 or nz == 0 or len(T) == 0:
        return solid, diag
    kol, z, tk = _kolumntraffar(V, T, pitch, origin_l, nx, ny, jitter_frac=jitter_frac, cap=cap)
    diag["n_traffar"] = int(len(kol))
    if len(kol) == 0:
        return solid, diag
    ordn = np.lexsort((z, kol))
    kol, z, tk = kol[ordn], z[ordn], tk[ordn]
    ncols = nx * ny
    antal = np.bincount(kol, minlength=ncols)
    slut = np.cumsum(antal)
    cum = np.concatenate([[0], np.cumsum(tk.astype(np.int64))])
    zmin, zmax = float(z.min()), float(z.max())
    span = max(zmax - zmin, 1e-12)
    # composite monotone key: the integer part is the column and the fractional part lies in
    # [0.25, 0.75] for hits, so a voxel centre below or above every hit in its column is clamped to
    # [0, 0.999) and never leaks into the neighbouring column's key interval.
    nycklar = kol.astype(np.float64) + 0.25 + 0.5 * (z - zmin) / span
    zc_all = origin_l[2] + (np.arange(nz, dtype=np.float64) + VOXEL_PROVPUNKT) * pitch
    aktiva = np.where(antal > 0)[0]
    diag["n_kolumner_med_traff"] = int(len(aktiva))
    flat = solid.reshape(ncols, nz)
    per = max(1, chunk_vox // max(nz, 1))
    n_diff = 0
    for c0 in range(0, len(aktiva), per):
        kolset = aktiva[c0:c0 + per]
        cc = np.repeat(kolset, nz)
        zz = np.tile(zc_all, len(kolset))
        kc = cc.astype(np.float64) + np.clip(0.25 + 0.5 * (zz - zmin) / span, 0.0, 0.999)
        n_le = np.searchsorted(nycklar, kc, side="right")
        e = slut[cc]
        inne_par = ((e - n_le) & 1) == 1
        inne_vin = (cum[e] - cum[n_le]) != 0
        n_diff += int(np.count_nonzero(inne_par != inne_vin))
        flat[kolset] = (inne_par if regel == "paritet" else inne_vin).reshape(len(kolset), nz)
    diag["n_vindning_vs_paritet_diff"] = n_diff
    diag["andel_vindning_vs_paritet"] = n_diff / float(nx * ny * nz)
    return solid, diag


_WINDKERNEL = {}


def _bygg_winding_kernel(wp):
    if "k" in _WINDKERNEL:
        return _WINDKERNEL["k"]

    @wp.kernel
    def k_sdf_winding(mesh: wp.uint64, ox: float, oy: float, oz: float, pitch: float, prov: float,
                      jx: float, jy: float, jz: float,
                      ny: int, nz: int, lin0: int, n: int, maxdist: float, acc: float, tr: float,
                      out: wp.array(dtype=wp.float32)):
        t = wp.tid()
        if t >= n:
            return
        lin = lin0 + t
        k = lin % nz
        j = (lin // nz) % ny
        i = lin // (nz * ny)
        # the same deterministic subvoxel jitter as the host path, but on all three axes: the
        # generalised winding number is exactly 0.5 on the surface, so a flat face lying exactly in
        # the sample grid would otherwise be classified as outside at both ends and the body loses a
        # voxel layer per axis (measured on the 60x40x30 box: 2.9% relative error at 1 mm pitch and
        # 6.3% at 2 mm, against 0.000 for the host path).
        p = wp.vec3(ox + (float(i) + prov) * pitch + jx,
                    oy + (float(j) + prov) * pitch + jy,
                    oz + (float(k) + prov) * pitch + jz)
        q = wp.mesh_query_point_sign_winding_number(mesh, p, maxdist, acc, tr)
        if q.result:
            cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            out[t] = q.sign * wp.length(p - cp)
        else:
            out[t] = wp.float32(1.0e9)

    _WINDKERNEL["k"] = k_sdf_winding
    return k_sdf_winding


def sd_via_winding_gpu(wp, V, T, pitch, origin_l, shape_l, device="cuda:0", chunk=4_000_000,
                       noggrannhet=2.0, troskel=0.5):
    """Exact signed distance per voxel from Warp's BVH and generalised winding number.

    Returns (sd_float32[shape_l], solid_bool, diagnostics). No distance transform and no flood fill:
    sd holds real mm distances to the nearest surface rather than voxel-quantised ones.
    """
    K = _bygg_winding_kernel(wp)
    nx, ny, nz = (int(shape_l[0]), int(shape_l[1]), int(shape_l[2]))
    ntot = nx * ny * nz
    sd = np.empty(ntot, dtype=np.float32)
    diag = dict(metod="winding_gpu", noggrannhet=noggrannhet, troskel=troskel, chunk=chunk)
    if ntot == 0:
        return sd.reshape(nx, ny, nz), np.zeros((nx, ny, nz), bool), diag
    Vf = np.ascontiguousarray(np.asarray(V, dtype=np.float32))
    Ti = np.ascontiguousarray(np.asarray(T, dtype=np.int32).reshape(-1))
    maxd = float(np.linalg.norm(np.ptp(np.asarray(V, dtype=np.float64), axis=0)) * 4.0 + 10.0 * pitch)
    jx = pitch * JITTER_FRAC * _GYLLENE[0]
    jy = pitch * JITTER_FRAC * _GYLLENE[1]
    jz = pitch * JITTER_FRAC * _GYLLENE[2]
    with wp.ScopedDevice(device):
        mesh = wp.Mesh(points=wp.array(Vf, dtype=wp.vec3), indices=wp.array(Ti, dtype=wp.int32))
        buf = wp.zeros(min(chunk, ntot), dtype=wp.float32)
        for lin0 in range(0, ntot, chunk):
            n = min(chunk, ntot - lin0)
            wp.launch(K, dim=n, inputs=[mesh.id, float(origin_l[0]), float(origin_l[1]),
                                        float(origin_l[2]), float(pitch), float(VOXEL_PROVPUNKT),
                                        jx, jy, jz, ny, nz, lin0, n,
                                        maxd, float(noggrannhet), float(troskel)], outputs=[buf])
            wp.synchronize()
            sd[lin0:lin0 + n] = buf.numpy()[:n]
    sd = sd.reshape(nx, ny, nz)
    return sd, (sd < 0.0), diag

def _solid_via_skal_floodfill(V, T, pitch, lo, global_shape=None, margin_vox_pad=2):
    """Legacy shell-and-flood-fill signing, kept for before/after comparisons.

    Takes the mesh, the pitch, the grid origin and optionally the global shape; returns
    (gmin, shape_l, surface_mask, solid_no_margin, sd_mm). gmin is in global grid indices relative to
    lo, using the same (P-lo)/pitch formula throughout, so the local window can be written straight
    into a global occupancy array with no separate floor(lo/pitch) recomputation. When global_shape is
    given, gmin/gmax are clamped against it before the fill is built: without that clamp a part whose
    geometry sticks out of the global window gets a larger local flood domain and a different outer
    topology than a clamped reference.
    """
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = B - A, C - A
    emax = np.maximum.reduce([np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1),
                              np.linalg.norm(C - B, axis=1)])
    n_sub = np.clip(np.ceil(emax / (pitch * 0.5)).astype(int), 1, 24)
    P = []
    for k in np.unique(n_sub):
        sel = n_sub == k
        uu, vv = np.meshgrid(np.linspace(0, 1, k + 1), np.linspace(0, 1, k + 1))
        msk = (uu + vv) <= 1.0
        uu, vv = uu[msk], vv[msk]
        P.append((A[sel][:, None, :] + uu[None, :, None] * e1[sel][:, None, :]
                  + vv[None, :, None] * e2[sel][:, None, :]).reshape(-1, 3))
    P = np.concatenate(P, axis=0)
    gmin = np.floor((P.min(0) - lo) / pitch).astype(np.int64) - margin_vox_pad
    gmax = np.ceil((P.max(0) - lo) / pitch).astype(np.int64) + margin_vox_pad + 1
    if global_shape is not None:
        gmin = np.maximum(gmin, 0)
        gmax = np.minimum(gmax, np.array(global_shape, dtype=np.int64))
    shape_l = tuple(gmax - gmin)
    yta = np.zeros(shape_l, dtype=bool)
    idx = np.rint((P - lo) / pitch).astype(np.int64) - gmin
    ok = np.all((idx >= 0) & (idx < np.array(shape_l)), axis=1)
    idx = idx[ok]
    yta[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    fri = ~yta
    lbl, _ = ndimage.label(fri)
    rand = set(np.unique(np.concatenate([
        lbl[0, :, :].ravel(), lbl[-1, :, :].ravel(), lbl[:, 0, :].ravel(),
        lbl[:, -1, :].ravel(), lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()])).tolist())
    rand.discard(0)
    utsida = np.isin(lbl, list(rand)) if rand else np.zeros(shape_l, dtype=bool)
    solid = yta | (~utsida & ~yta)
    # Two distance transforms of the solid mask itself, not of the surface-point mask: at a surface
    # voxel the surface mask is already the zero set, so the outside transform reads 0.0 there and
    # sd becomes -0.0, and -0.0 < 0.0 is false in IEEE754. For a thin part whose whole body is
    # surface voxels that misclassifies every material voxel as void. Against the solid mask, a solid
    # voxel with at least one non-solid neighbour is at least one pitch from the nearest non-solid
    # position, never exactly zero, so the sign is strictly negative as it should be.
    d_out = ndimage.distance_transform_edt(~solid, sampling=(pitch, pitch, pitch)).astype(np.float32)
    d_in = ndimage.distance_transform_edt(solid, sampling=(pitch, pitch, pitch)).astype(np.float32)
    sd = (d_out - d_in).astype(np.float32)
    return gmin, shape_l, yta, solid, sd



def surface_raster_and_flood(V, T, pitch, lo, global_shape=None, margin_vox_pad=2,
                             metod=None, wp=None, device="cuda:0", diag_ut=None,
                             ytkorrektion=None):
    """Signing selector. Returns (gmin, shape_l, surface, solid, sd).

    `metod` chooses how inside/outside is decided:
      "raypar_vindning" (default) -- +Z ray winding number per voxel, sd from two distance transforms
      "raypar_paritet"            -- same hits, even/odd rule (independent anchor)
      "winding_gpu"               -- Warp mesh_query_point_sign_winding_number, exact sd, needs a GPU
      "skal_floodfill"            -- the legacy path, kept for before/after comparisons
    For the ray paths `surface` is the material's surface layer (solid minus its erosion) rather than
    the point raster, so the "solid is only the shell" diagnostic still works.
    `ytkorrektion` (default YTKORREKTION_STANDARD, on) subtracts the half pitch by which a distance
    transform of a voxel mask overstates the distance to the surface; it applies to the two
    distance-transform paths only, since winding_gpu already returns true distances.
    """
    metod = metod or METOD_STANDARD
    ytkorr = YTKORREKTION_STANDARD if ytkorrektion is None else bool(ytkorrektion)
    if metod not in METODER:
        raise ValueError(f"okand metod {metod!r}, valj bland {METODER}")
    if metod == "skal_floodfill":
        r = _solid_via_skal_floodfill(V, T, pitch, lo, global_shape=global_shape,
                                      margin_vox_pad=margin_vox_pad)
        if diag_ut is not None:
            diag_ut.update(dict(metod="skal_floodfill"))
        return r
    gmin, shape_l = _fonster(V, pitch, lo, global_shape, margin_vox_pad)
    origin_l = np.asarray(lo, dtype=np.float64) + gmin.astype(np.float64) * pitch
    if metod == "winding_gpu":
        if wp is None:
            wp, _have = _import_warp()
        sd, solid, diag = sd_via_winding_gpu(wp, V, T, pitch, origin_l, shape_l, device=device)
    else:
        solid, diag = solid_via_stralvindning(
            V, T, pitch, origin_l, shape_l,
            regel="paritet" if metod == "raypar_paritet" else "vindning")
        d_out = ndimage.distance_transform_edt(~solid, sampling=(pitch, pitch, pitch)).astype(np.float32)
        d_in = ndimage.distance_transform_edt(solid, sampling=(pitch, pitch, pitch)).astype(np.float32)
        sd = (d_out - d_in).astype(np.float32)
        if ytkorr:
            sd = (sd - np.sign(sd) * np.float32(0.5 * pitch)).astype(np.float32)
    diag["ytkorrektion"] = bool(ytkorr and metod != "winding_gpu")
    if solid.any():
        yta = solid & ~ndimage.binary_erosion(solid, ndimage.generate_binary_structure(3, 1))
    else:
        yta = np.zeros_like(solid)
    diag["metod_vald"] = metod
    if diag_ut is not None:
        diag_ut.update(diag)
    return gmin, shape_l, yta, solid, sd


def ray_parity_probe(V, T, points):
    """Independent anchor: ray-triangle parity (Moller-Trumbore, +Z) per probe point.

    Takes the mesh and the probe points; returns a boolean array (True = inside by parity). Used only
    to flag divergence from the delivered signing, never to produce it.
    """
    origin = points.astype(np.float64)
    d = np.array([0.0, 0.0, 1.0])
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = B - A, C - A
    inside_counts = np.zeros(len(points), dtype=np.int64)
    EPS = 1e-9
    for i, o in enumerate(origin):
        pvec = np.cross(np.broadcast_to(d, e2.shape), e2)
        det = np.einsum('ij,ij->i', e1, pvec)
        valid = np.abs(det) > EPS
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        tvec = o - A
        u = np.einsum('ij,ij->i', tvec, pvec) * inv_det
        m1 = valid & (u >= -1e-7) & (u <= 1 + 1e-7)
        qvec = np.cross(tvec, e1)
        v = np.einsum('ij,ij->i', np.broadcast_to(d, qvec.shape), qvec) * inv_det
        m2 = m1 & (v >= -1e-7) & ((u + v) <= 1 + 1e-7)
        t = np.einsum('ij,ij->i', e2, qvec) * inv_det
        m3 = m2 & (t > 1e-7)
        inside_counts[i] = int(np.sum(m3))
    return (inside_counts % 2) == 1


# ================================================================================================
# Step 4: generic block classifier over a precomputed dense array (the same Lipschitz-margin logic
# as faltkarna_v1.py's analytic kernels, with the formula replaced by an array lookup).
# ================================================================================================
_KERNELS_CACHE = {}


def _build_dense_kernels(wp):
    if "k" in _KERNELS_CACHE:
        return _KERNELS_CACHE["k"]

    @wp.kernel
    def k_classify_dense(n_bx: int, n_by: int, n_bz: int, block: int,
                          nx: int, ny: int, nz: int, margin: float,
                          dense: wp.array(dtype=wp.float32),
                          kind: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        bk = tid % n_bz
        bj = (tid // n_bz) % n_by
        bi = tid // (n_bz * n_by)
        vx0, vy0, vz0 = bi * block, bj * block, bk * block
        min_abs = wp.float32(1.0e30)
        pos_seen = False
        neg_seen = False
        for c in range(8):
            cx = wp.min(vx0 + (c & 1) * block, nx - 1)
            cy = wp.min(vy0 + ((c >> 1) & 1) * block, ny - 1)
            cz = wp.min(vz0 + ((c >> 2) & 1) * block, nz - 1)
            idx = (cx * ny + cy) * nz + cz
            v = dense[idx]
            min_abs = wp.min(min_abs, wp.abs(v))
            if v >= 0.0:
                pos_seen = True
            else:
                neg_seen = True
        if pos_seen and neg_seen:
            kind[tid] = 0
        elif min_abs <= margin:
            kind[tid] = 0
        elif neg_seen:
            kind[tid] = -1
        else:
            kind[tid] = 1

    @wp.kernel
    def k_eval_tile_dense(active_ids: wp.array(dtype=wp.int32), n_by: int, n_bz: int, block: int,
                           nx: int, ny: int, nz: int,
                           dense: wp.array(dtype=wp.float32),
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
        vx = wp.min(bi * block + li, nx - 1)
        vy = wp.min(bj * block + lj, ny - 1)
        vz = wp.min(bk * block + lk, nz - 1)
        idx = (vx * ny + vy) * nz + vz
        out[tid] = dense[idx]

    _KERNELS_CACHE["k"] = dict(k_classify_dense=k_classify_dense, k_eval_tile_dense=k_eval_tile_dense)
    return _KERNELS_CACHE["k"]


def klassificera_och_evaluera_fran_tatt_falt(wp, dense: np.ndarray, pitch: float, block: int,
                                              device: str):
    """Generic rasterisation op: takes an arbitrary precomputed dense SDF array and produces a sparse
    block classification by the same Lipschitz 8-corner rule as the analytic test field.

    Returns a dict with the block kinds, the active ids, the tiles and the block counts.
    """
    K = _build_dense_kernels(wp)
    nx, ny, nz = dense.shape
    n_bx, n_by, n_bz = -(-nx // block), -(-ny // block), -(-nz // block)
    n_blocks = n_bx * n_by * n_bz
    margin = math.sqrt(3.0) * block * pitch
    with wp.ScopedDevice(device):
        dense_w = wp.array(dense.astype(np.float32).reshape(-1), dtype=wp.float32)
        kind_w = wp.zeros(n_blocks, dtype=wp.int32)
        wp.launch(K["k_classify_dense"], dim=n_blocks,
                  inputs=[n_bx, n_by, n_bz, block, nx, ny, nz, margin, dense_w], outputs=[kind_w])
        wp.synchronize()
        kind = kind_w.numpy()
        active_ids = np.where(kind == 0)[0].astype(np.int32)
        n_active = len(active_ids)
        b3 = block ** 3
        if n_active > 0:
            active_w = wp.array(active_ids, dtype=wp.int32)
            out_w = wp.zeros(n_active * b3, dtype=wp.float32)
            wp.launch(K["k_eval_tile_dense"], dim=n_active * b3,
                      inputs=[active_w, n_by, n_bz, block, nx, ny, nz, dense_w], outputs=[out_w])
            wp.synchronize()
            tiles = out_w.numpy().reshape(n_active, block, block, block)
        else:
            tiles = np.zeros((0, block, block, block), dtype=np.float32)
    return dict(kind=kind, active_ids=active_ids, tiles=tiles, n_bx=n_bx, n_by=n_by, n_bz=n_bz,
                n_blocks=int(n_blocks), n_active=int(n_active), margin_mm=margin)


def reconstruct_solid_from_sparse(sf, nx, ny, nz, block):
    """Reconstructs a (nx,ny,nz) boolean array (True = material) from the block kinds and tiles.

    kind == -1 blocks are all inside, kind == +1 blocks are all outside, active blocks read tile < 0.
    """
    n_bx, n_by, n_bz = sf["n_bx"], sf["n_by"], sf["n_bz"]
    px, py, pz = n_bx * block, n_by * block, n_bz * block
    kind_grid = sf["kind"].reshape(n_bx, n_by, n_bz)
    background = kind_grid < 0
    padded = np.empty((px, py, pz), dtype=bool)
    block_view = padded.reshape(n_bx, block, n_by, block, n_bz, block).transpose(0, 2, 4, 1, 3, 5)
    block_view[:] = background[:, :, :, None, None, None]
    if len(sf["active_ids"]) > 0:
        bi, bj, bk = np.unravel_index(sf["active_ids"], (n_bx, n_by, n_bz))
        block_view[bi, bj, bk] = sf["tiles"] < 0.0
    return padded[:nx, :ny, :nz]


# ================================================================================================
# Orchestration: one part -> sparse SDF -> local occupancy (with margin dilation)
# ================================================================================================
# Watertightness gate: closing the fail-open hole in the signing.
# Measured: a mesh with 2181 triangles removed produced 70% of the closed volume with no flag at all,
# and meshes stored without vertex merging all read is_watertight=False yet passed straight through.
# The signing cannot detect this itself: a hole lets the outside label leak into the body and the
# result looks like a valid, smaller solid. The gate is therefore independent of the signing:
#   (a) is_watertight plus the number of edges not shared by two facets (the size of the hole, not a
#       tolerance);
#   (b) voxel volume against mesh volume (only meaningful when the mesh is closed: always reported,
#       but only decisive in the closed case);
#   (c) ray parity (the independent signing family) against the delivered classification at random
#       probe points: two methods that agree on a closed mesh and differ on an open one.
# The result always carries a "vattentathet" field with status OK / OPPEN_MESH / VOLYMAVVIKELSE /
# PARITETSDIVERGENS. Strict mode (grind="strikt" or env FALTKARNA_VATTENTATHET_STRIKT=1) raises
# ValueError instead. Repair (laga_oppen=True) fills holes before rasterisation and is declared in
# the result, never silent.
# ================================================================================================
VATTENTATHET_TOL_VOLYM = 0.02   # 2% voxel volume against mesh volume for a closed mesh
VATTENTATHET_TOL_PARITET = 0.02  # at most 2% of probe points where the two signing families disagree


def _trimesh_fakta(V, T, sla_ihop=True):
    """Mesh facts for the gate, computed on the vertex-merged mesh.

    A mesh stored with unmerged vertices has every triangle carrying its own corners, so no edge is
    shared and is_watertight reads False even for a provably closed body. Vertex merging is
    topological and lossless (it moves no point), so the gate judges the merged mesh and reports
    both; otherwise it flags a storage convention instead of a hole. Returns a dict of facts.
    """
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, dtype=np.float64), np.asarray(T, dtype=np.int64), process=False)
    if sla_ihop:
        m = m.copy()
        m.merge_vertices()
    kanter = m.edges_sorted
    _, antal = np.unique(kanter, axis=0, return_counts=True)
    return m, bool(m.is_watertight), int((antal != 2).sum()), float(m.volume)


def laga_oppen_mesh(V, T):
    """Fills holes on a copy of the mesh; returns (V2, T2, declaration).

    This changes geometry, so it is always opt-in and always declared in the result.
    """
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, dtype=np.float64), np.asarray(T, dtype=np.int64), process=False)
    n_tri_fore = int(len(m.faces))
    n_v_fore = int(len(m.vertices))
    fore_vattentat = bool(m.is_watertight)
    m.merge_vertices()   # topological and lossless: moves no point
    vattentat_efter_ihopslagning = bool(m.is_watertight)
    n_hal_fyllda = 0
    if not vattentat_efter_ihopslagning:
        n_tri_innan_fyll = int(len(m.faces))
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
        n_hal_fyllda = int(len(m.faces)) - n_tri_innan_fyll
    return (np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64),
            dict(lagning="merge_vertices" + ("+fill_holes+fix_normals" if n_hal_fyllda else ""),
                 vattentat_fore=fore_vattentat, vattentat_efter_ihopslagning=vattentat_efter_ihopslagning,
                 vattentat_efter=bool(m.is_watertight), n_trianglar_fore=n_tri_fore,
                 n_verts_fore=n_v_fore, n_verts_efter=int(len(m.vertices)),
                 n_trianglar_efter=int(len(m.faces)), n_trianglar_tillagda=int(len(m.faces)) - n_tri_fore,
                 n_trianglar_ur_halfyllning=n_hal_fyllda))


def vattentathetsgrind(V, T, solid, pitch, lo=None, gmin=None, n_prob=96, seed=20260902, sd=None):
    """The watertightness gate (see the block comment above).

    Takes the mesh, the delivered boolean occupancy on the local grid, the pitch, the grid origin lo
    and window origin gmin (so the probe points land on the right world positions), the probe count
    and optionally the signed field. Returns a dict with the status, the flags and every measured
    number behind them.
    """
    ut = dict(status="OK", flaggor=[], n_prob=0)
    try:
        _, vattentat, n_kanter_ej_par, vol_mesh = _trimesh_fakta(V, T, sla_ihop=True)
        _, vattentat_ratt, n_kanter_ratt, _ = _trimesh_fakta(V, T, sla_ihop=False)
        ut.update(vattentat_utan_sammanslagning=vattentat_ratt,
                  n_kanter_ej_par_utan_sammanslagning=n_kanter_ratt,
                  kravde_vertex_sammanslagning=bool(vattentat and not vattentat_ratt))
    except Exception as e:  # noqa: BLE001 -- a mesh-library failure must not kill the run, but must show
        ut.update(status="GRIND-BROKEN", fel=str(e)[:300], flaggor=["GRIND-BROKEN"])
        return ut
    vol_vox = float(np.count_nonzero(solid)) * pitch ** 3
    ut.update(vattentat=vattentat, n_kanter_ej_delade_av_2_trianglar=n_kanter_ej_par,
              volym_trimesh_mm3=vol_mesh, volym_voxel_mm3=vol_vox,
              rel_avvikelse_voxel_vs_trimesh=(abs(vol_vox - abs(vol_mesh)) / abs(vol_mesh)) if vol_mesh else None,
              volym_trimesh_giltig=vattentat)
    if not vattentat:
        ut["status"] = "OPPEN_MESH"
        ut["flaggor"].append("OPPEN_MESH")
    elif ut["rel_avvikelse_voxel_vs_trimesh"] is not None and ut["rel_avvikelse_voxel_vs_trimesh"] > VATTENTATHET_TOL_VOLYM:
        ut["status"] = "VOLYMAVVIKELSE"
        ut["flaggor"].append("VOLYMAVVIKELSE")
    # (c) independent parity anchor at random voxel centres, both solid and empty
    try:
        rng = np.random.default_rng(seed)
        idx_in = np.argwhere(solid)
        idx_ut = np.argwhere(~solid)
        if len(idx_in) and len(idx_ut):
            k = max(1, n_prob // 2)
            pick = np.vstack([idx_in[rng.choice(len(idx_in), size=min(k, len(idx_in)), replace=False)],
                              idx_ut[rng.choice(len(idx_ut), size=min(k, len(idx_ut)), replace=False)]])
            # grid origin: lo + gmin*pitch (gmin is in global grid indices relative to lo). Without
            # gmin the probes land hundreds of mm away and the "divergence" reads 34-52% on every
            # part: a measurement error, not a leak.
            lo_lokal = (np.asarray(lo, dtype=np.float64) + np.asarray(gmin, dtype=np.float64) * pitch
                        if lo is not None and gmin is not None
                        else np.asarray(V, dtype=np.float64).min(0) - 3 * pitch)
            P = lo_lokal + (pick + 0.5) * pitch
            paritet = ray_parity_probe(np.asarray(V, dtype=np.float64), np.asarray(T, dtype=np.int64), P)
            flood = solid[pick[:, 0], pick[:, 1], pick[:, 2]]
            oense = paritet != flood
            div = float(np.mean(oense))
            ut.update(n_prob=int(len(P)), paritet_divergens_frac=div)
            # Split the divergence. The two signing families are defined to agree inside and
            # outside the body, not on the surface. A probe point lying exactly in a face plane
            # (common: axis-aligned faces land in voxel centres) is genuinely ambiguous, and the two
            # families break that tie by different conventions (the ray winding with a declared
            # subvoxel jitter, the parity probe without). Measured on a flat disc part, every
            # divergent probe point lay in one face plane. The gate therefore judges deep probe
            # points only; a real leak shows up there, a convention difference never does. Both
            # numbers are reported.
            # Depth is measured geometrically, not from sd: the distance transform gives a voxel with
            # one non-solid neighbour exactly |sd| = pitch, so an "|sd| >= pitch" filter passes the
            # whole first surface layer. The distance to the nearest triangle is independent of both
            # signing families and of the voxel quantisation.
            try:
                import trimesh as _tm
                _m = _tm.Trimesh(np.asarray(V, dtype=np.float64), np.asarray(T, dtype=np.int64),
                                 process=False)
                _m.merge_vertices()
                _, dist_yta, _ = _tm.proximity.closest_point(_m, P)
                djup = np.asarray(dist_yta) >= pitch
                ut["avstand_till_yta_median_mm"] = float(np.median(dist_yta))
            except Exception as e:  # noqa: BLE001
                ut["djupmatt_fel"] = str(e)[:200]
                djup = (np.abs(np.asarray(sd)[pick[:, 0], pick[:, 1], pick[:, 2]]) >= 2.0 * pitch
                        if sd is not None else np.zeros(len(P), dtype=bool))
            n_djup = int(np.count_nonzero(djup))
            div_djup = float(np.mean(oense[djup])) if n_djup else None
            ut.update(n_prob_djup=n_djup, paritet_divergens_frac_djup=div_djup,
                      paritet_divergens_domande="djup" if n_djup >= 16 else "alla")
            div_dom = div_djup if (div_djup is not None and n_djup >= 16) else div
            if div_dom > VATTENTATHET_TOL_PARITET and ut["status"] == "OK":
                ut["status"] = "PARITETSDIVERGENS"
                ut["flaggor"].append("PARITETSDIVERGENS")
    except Exception as e:  # noqa: BLE001
        ut["paritet_fel"] = str(e)[:200]
    return ut


# ================================================================================================
# Small-feature resolution: choosing the pitch from the smallest feature radius in the mesh.
# ================================================================================================
# A through hole of radius R rasterised at pitch h is a cylinder wall sampled at h/R radians of arc
# per voxel, and every error around that wall scales with h/R and with nothing else (measured: the
# radial deviation of the zero level set reads 12.5 % at R/h = 8 for every radius from 1 to 16 mm).
# What degrades first is not the topology: on the measured part the wall is still one closed ring at
# two voxels per radius, and at one voxel per radius the hole is simply 27 % too large. It is the
# accuracy of the level set all round the hole, which is what makes a rendered hole look scalloped.
# The block classifier cannot repair that: its tiles carry the dense field exactly, so a feature that
# the dense field lost at pitch h is lost in the sparse field too. There is no per-block refinement in
# this substrate -- the blocks are BLOCK^3 voxels of one global pitch -- so the only lever is the
# pitch itself, and the parameter below turns "small features exist in this mesh" into a pitch.
#
# The smallest feature radius is measured on the mesh, not declared. For every pair of adjacent
# triangles whose dihedral angle is below the crease threshold (a sharp edge is a crease, not
# curvature) the local radius of curvature across the shared edge is
#     r = w / (2 * sin(theta / 2)),
# where theta is the dihedral angle and w is the smaller of the two triangles' extents measured
# perpendicular to the shared edge. For a cylinder tessellated into chords this is exact: w is the
# chord 2*R*sin(theta/2). The estimator deliberately does not use the triangle centroids: on a
# cylinder whose facets are split into two long triangles, the centroid offset across an edge is
# 2/3 of the facet's angular step and the centroid form of the same estimate reads 2/3*R (measured:
# 0.667, 1.333, 2.666, 5.333, 10.678 for holes of radius 1, 2, 4, 8, 16 mm).
# A low percentile rather than the raw minimum is taken, so one broken triangle cannot drive the
# whole grid, and both numbers are reported.
# The factor 6 is measured, not chosen: with the sub-voxel correction above, the zero level set of a
# hole of radius R rasterised at pitch h sits h/2 off the true wall, so sampling the field on the
# wall reads |sd| / R = h / (2R). The declared gate of 5 % at R = 2 mm therefore needs h <= 0.2 mm,
# and a part whose smallest feature is R_min = 1 mm needs h <= R_min / 5; 6 keeps a margin.
# h <= R_min / 4, the first thing one would try, leaves the wall one closed ring at every radius but
# reads 12 % at R = 2 mm (measured, see mesh_to_sdf_small_features_v1.py).
FEATURE_PITCH_FAKTOR = 6.0      # h <= R_min / 6, measured below: see the small-feature cell
FEATURE_CREASE_GRAD = 30.0      # dihedral angles above this are edges, not curvature
FEATURE_PERCENTIL = 1.0         # robust minimum over the per-edge radius estimates
FEATURE_MAX_VOXLAR = 40_000_000  # cost ceiling for the refined grid; hitting it is declared, not silent


def minsta_feature_radie(V, T, crease_grad=FEATURE_CREASE_GRAD, percentil=FEATURE_PERCENTIL):
    """Smallest radius of curvature in the mesh, from the dihedral angle across every smooth edge.

    Takes the mesh and the crease threshold; returns (r_min_mm or None, diagnostics). None means the
    mesh carries no curvature at all (a polyhedron): there is then no small round feature to resolve.
    """
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, dtype=np.float64), np.asarray(T, dtype=np.int64), process=False)
    m.merge_vertices()
    par, vinkel, kant = m.face_adjacency, m.face_adjacency_angles, m.face_adjacency_edges
    diag = dict(n_par=int(len(par)), crease_grad=crease_grad, percentil=percentil)
    if len(par) == 0:
        return None, dict(diag, n_mjuka_par=0)
    sel = (vinkel > 1.0e-4) & (vinkel < math.radians(crease_grad))
    diag["n_mjuka_par"] = int(np.count_nonzero(sel))
    if not sel.any():
        return None, diag
    par, vinkel, kant = par[sel], vinkel[sel], kant[sel]
    P = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    a = P[kant[:, 0]]
    u = P[kant[:, 1]] - a
    u = u / np.maximum(np.linalg.norm(u, axis=1), 1e-15)[:, None]

    def _bredd(fi):
        w = np.zeros(len(fi))
        for k in range(3):
            q = P[F[fi, k]] - a
            perp = q - np.einsum("ij,ij->i", q, u)[:, None] * u
            w = np.maximum(w, np.linalg.norm(perp, axis=1))
        return w

    w = np.minimum(_bredd(par[:, 0]), _bredd(par[:, 1]))
    r = w / (2.0 * np.sin(vinkel / 2.0))
    r_p = float(np.percentile(r, percentil))
    diag.update(r_min_mm=float(r.min()), r_percentil_mm=r_p, r_median_mm=float(np.median(r)))
    return r_p, diag


def valj_pitch_for_feature(V, T, pitch, feature_radius_min, faktor=FEATURE_PITCH_FAKTOR,
                           max_voxlar=FEATURE_MAX_VOXLAR):
    """Picks the pitch that resolves the smallest feature: h <= R_min / faktor.

    feature_radius_min is either a radius in mm or "auto" (measure it with minsta_feature_radie).
    The pitch is only ever refined, never coarsened, and never below the value at which the part's
    bounding box would exceed max_voxlar voxels -- that clamp is reported in the diagnostics rather
    than applied silently. Returns (pitch_effektiv, diagnostics).
    """
    diag = dict(begard=feature_radius_min, pitch_in=float(pitch), faktor=float(faktor),
                pitch_ut=float(pitch), r_min_mm=None, klamd_av_kostnadstak=False)
    if feature_radius_min is None:
        return float(pitch), diag
    if isinstance(feature_radius_min, str):
        if feature_radius_min != "auto":
            raise ValueError(f"feature_radius_min must be a radius in mm or 'auto', got {feature_radius_min!r}")
        r_min, d = minsta_feature_radie(V, T)
        diag["matning"] = d
    else:
        r_min = float(feature_radius_min)
    diag["r_min_mm"] = None if r_min is None else float(r_min)
    if r_min is None or r_min <= 0.0:
        return float(pitch), diag
    h = min(float(pitch), float(r_min) / float(faktor))
    ext = np.ptp(np.asarray(V, dtype=np.float64), axis=0)
    n_vox = float(np.prod(np.maximum(ext / h, 1.0) + 5.0))
    if n_vox > max_voxlar:
        h_tak = float(pitch)
        h_klamd = max(h, float((np.prod(np.maximum(ext, 1e-9)) / max_voxlar) ** (1.0 / 3.0)))
        h = min(h_tak, h_klamd)
        diag["klamd_av_kostnadstak"] = True
        diag["max_voxlar"] = int(max_voxlar)
    diag["pitch_ut"] = float(h)
    diag["n_voxlar_uppskattat"] = float(np.prod(np.maximum(ext / h, 1.0) + 5.0))
    return float(h), diag


def mesh_to_sdf_del(wp, V, T, pitch, marginal_mm, lo, device, global_shape=None, block=BLOCK,
                    grind="flagga", laga_oppen=False, grind_n_prob=96, metod=None,
                    feature_radius_min=None, ytkorrektion=None, returnera_falt=False):
    """One part: mesh -> signed field -> sparse classification -> local occupancy with margin.

    Takes a warp module, the mesh (V, T), the pitch, the margin in mm, the grid origin lo, the device,
    and optionally the global shape, the block size, the gate mode and the signing method.
    grind="flagga" (default) always puts the watertightness verdict in the result's "vattentathet"
    field; grind="strikt" (or env FALTKARNA_VATTENTATHET_STRIKT=1) raises instead; grind="av" skips
    it. laga_oppen=True fills holes before rasterisation and declares it.

    feature_radius_min (default None, off, so the numbers of every existing caller are unchanged)
    refines the pitch to resolve the smallest round feature: give a radius in mm, or "auto" to measure
    it from the mesh (see valj_pitch_for_feature). The pitch actually used is returned as
    "pitch_effektiv" and the window origin and shape are in units of that pitch, not of the requested
    one. returnera_falt=True additionally returns the dense signed field and the sparse field itself.

    Returns a dict with the window origin and shape, the final occupancy, the block counts, timings
    and the gate verdict.
    """
    lagning = None
    if laga_oppen:
        V, T, lagning = laga_oppen_mesh(V, T)
    pitch_begard = float(pitch)
    pitch, feature_diag = valj_pitch_for_feature(V, T, pitch, feature_radius_min)
    t0 = time.time()
    diag_sign = {}
    gmin, shape_l, yta, solid_no_margin, sd = surface_raster_and_flood(
        V, T, pitch, lo, global_shape=global_shape, metod=metod, wp=wp, device=device,
        diag_ut=diag_sign, ytkorrektion=ytkorrektion)
    t_flood = time.time() - t0
    t1 = time.time()
    sf = klassificera_och_evaluera_fran_tatt_falt(wp, sd, pitch, block, device)
    t_gpu = time.time() - t1
    nx, ny, nz = shape_l
    solid_gpu = reconstruct_solid_from_sparse(sf, nx, ny, nz, block)
    n_diff_pre_margin = int(np.sum(solid_gpu != solid_no_margin))
    m_vox = int(round(marginal_mm / pitch))
    solid_final = solid_gpu
    if m_vox > 0:
        solid_final = ndimage.binary_dilation(solid_gpu, ndimage.generate_binary_structure(3, 1),
                                              iterations=m_vox)
    strikt = grind == "strikt" or os.environ.get("FALTKARNA_VATTENTATHET_STRIKT") == "1"
    vt = None
    if grind != "av":
        t2 = time.time()
        vt = vattentathetsgrind(V, T, solid_no_margin, pitch, lo=lo, gmin=gmin, n_prob=grind_n_prob, sd=sd)
        vt["grind_wall_s"] = time.time() - t2
        vt["lagning"] = lagning
        if strikt and vt["status"] != "OK":
            raise ValueError(f"vattentathetsgrind: {vt['status']} {vt.get('flaggor')} -- "
                             f"vattentat={vt.get('vattentat')} kanter_ej_par={vt.get('n_kanter_ej_delade_av_2_trianglar')} "
                             f"paritet_divergens={vt.get('paritet_divergens_frac')}")
    ut = dict(gmin=gmin, shape_l=shape_l, solid_final=solid_final, signering=diag_sign,
              solid_no_margin_cpu_form=solid_no_margin, n_diff_pre_margin_gpu_vs_cpu=n_diff_pre_margin,
              t_flood_s=t_flood, t_gpu_classify_s=t_gpu, vattentathet=vt,
              n_active=sf["n_active"], n_blocks=sf["n_blocks"],
              pitch_begard=pitch_begard, pitch_effektiv=float(pitch), feature=feature_diag)
    if returnera_falt:
        ut.update(sd=sd, sf=sf, block=block)
    return ut


def _selftest():
    """Runs the pipeline on analytic bodies and on a deliberately opened mesh.

    A closed box and a closed sphere must reproduce their analytic volume within the discretisation
    band and pass the watertightness gate; the same box with facets removed must be flagged.
    Returns the result dict and raises SystemExit on a gate failure.
    """
    import trimesh
    wp, have_cuda = _import_warp()
    device = "cuda:0" if have_cuda else "cpu"
    pitch = 1.0
    out = {"device": device, "pitch_mm": pitch, "cases": []}

    box = trimesh.creation.box(extents=(60.0, 40.0, 30.0))
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    for namn, mesh, vol_analytic in (("box_60x40x30", box, 60.0 * 40.0 * 30.0),
                                     ("sphere_r20", sphere, 4.0 / 3.0 * math.pi * 20.0 ** 3)):
        V = np.asarray(mesh.vertices, dtype=np.float64)
        T = np.asarray(mesh.faces, dtype=np.int64)
        lo = V.min(axis=0) - 3.0 * pitch
        r = mesh_to_sdf_del(wp, V, T, pitch, 0.0, lo, device)
        vol_vox = float(np.sum(r["solid_final"])) * pitch ** 3
        rel = abs(vol_vox - vol_analytic) / vol_analytic
        out["cases"].append({
            "namn": namn, "vol_voxel_mm3": vol_vox, "vol_analytic_mm3": vol_analytic,
            "rel_err": rel, "vattentathet_status": r["vattentathet"]["status"],
            "n_blocks": r["n_blocks"], "n_active": r["n_active"],
            "gate_pass": bool(rel < 0.05 and r["vattentathet"]["status"] == "OK"),
        })

    # negative control: remove a patch of facets and require the gate to flag the open mesh
    V = np.asarray(box.vertices, dtype=np.float64)
    T = np.asarray(box.faces, dtype=np.int64)[2:]
    lo = V.min(axis=0) - 3.0 * pitch
    r_open = mesh_to_sdf_del(wp, V, T, pitch, 0.0, lo, device)
    out["cases"].append({
        "namn": "box_with_removed_facets", "vattentathet_status": r_open["vattentathet"]["status"],
        "vattentat": r_open["vattentathet"].get("vattentat"),
        "gate_pass": bool(r_open["vattentathet"]["status"] != "OK"),
    })
    out["ALL_PASS"] = all(c["gate_pass"] for c in out["cases"])
    return out


if __name__ == "__main__":
    res = _selftest()
    print(json.dumps(res, indent=2))
    if not res["ALL_PASS"]:
        raise SystemExit("mesh_to_sdf selftest FAILED")
    print("PASS")
