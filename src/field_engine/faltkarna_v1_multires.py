#!/usr/bin/env python3
"""Per-block pitch for the block-sparse SDF: a two-level structure where every active block carries
its own resolution.

The gap this closes. `faltkarna_v1.py` and the mesh path on top of it have one global pitch. A part
whose smallest round feature is a 1 mm hole forces `feature_radius_min="auto"` to pick a pitch that
resolves that hole everywhere, so a 200 mm plate is rasterised at 0.167 mm all over, including the
flat faces that a pitch four times as coarse would carry exactly as well. Here the brick lattice
stays one lattice, but each brick stores its samples at its own pitch out of {h, h/2, h/4}, chosen
from the local radius of curvature of the mesh inside that brick.

Structure. The lattice is the block lattice of the coarse field: bricks of BLOCK^3 coarse voxels,
world side L = BLOCK*h. A brick at level l holds (BLOCK*2^l + 1)^3 samples at pitch h/2^l over the
same world cube -- the "+1" is the one-sample overlap band, so a brick's own trilinear interpolant is
defined on the whole closed cube and never reads a neighbour. Inactive bricks store nothing and are
answered with the classifier's own Lipschitz bound (+/- margin), exactly as the single-pitch
reconstruction does.

Pitch per brick. The local feature radius is measured with `minsta_feature_radie` from
`faltkarna_v1_mesh_to_sdf.py` on the submesh of triangles that touch the brick, and the level is the
coarsest one with h_l <= r_local / MULTIRES_PITCH_FAKTOR. The factor is 10, not the 6 of the global
path, and that is a consequence of the measurement, not a preference: with the half-pitch surface
correction the zero level set of a hole of radius R rasterised at pitch h_l sits h_l/2 off the wall,
so sampling the field on the wall reads h_l/(2R); the declared 5 % gate therefore needs h_l <= R/10
at the same R. The global path gets away with 6 because the pitch it picks from the *smallest*
feature is far finer than R/10 for every larger feature; a per-block pitch has no such slack, since
each block is resolved from its own radius.

Continuity across a pitch change. The face between a fine brick and a coarser one is a sample plane
of both. The fine brick's face layer is overwritten by the coarse neighbour's field, interpolated
inside the coarse brick's own face plane (`_lagg_overlappsband`). A bilinear function restricted to a
sub-rectangle is bilinear, so resampling it on the refined face lattice and interpolating bilinearly
there reproduces it exactly: the two interpolants agree on the whole shared face, not only at the
coarse nodes, and the zero level set is continuous across the pitch change. The residual jump is
measured (`kontinuitetscheck`) by evaluating the delivered query a hair on each side of every such
face, before and after the band is applied.

Queries. `avstand(F, punkter)` resolves each point's brick and interpolates at that brick's pitch;
`gradient(F, punkter)` central-differences at the brick's own pitch, clamped inside the brick.

Run `python faltkarna_v1_multires.py --multires` for the three-way measurement (multires vs
global-fine vs global-coarse) on the holed plate and the synthetic bracket, the memory and build-time
comparison, the continuity check and the two-run determinism certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "parts"))

import faltkarna_v1_mesh_to_sdf as M2S            # noqa: E402
import mesh_to_sdf_small_features_v1 as SMF       # noqa: E402
import holed_plate_v1 as PART                     # noqa: E402

BLOCK = M2S.BLOCK
NIVAER = (0, 1, 2)                 # pitch h, h/2, h/4
MULTIRES_PITCH_FAKTOR = 10.0       # h_l <= r_local / 10, derived above from the 5 % deviation gate
KLUSTER_PAD_VOX = 16               # padding of a refinement window, in voxels of that level's pitch
CREASE_GRAD = M2S.FEATURE_CREASE_GRAD
ARTIFACTS = os.path.join(ROOT, "artifacts")
UT_JSON = os.path.join(ARTIFACTS, "faltkarna_v1_multires.json")

N_VINKLAR = SMF.N_VINKLAR
GATE_RADIE_MM = SMF.GATE_RADIE_MM      # 2 mm
GATE_AVVIKELSE = SMF.GATE_AVVIKELSE    # 5 %
GATE_MINNESKVOT = 0.30                 # multires active voxels <= 30 % of global-fine


@dataclass
class MultiresField:
    """Two-level block-sparse field with one pitch per block."""
    lo: np.ndarray                 # world origin of the brick lattice (the coarse window origin)
    h_grov: float                  # coarse pitch
    block: int
    brick_shape: tuple             # (n_bx, n_by, n_bz)
    kind: np.ndarray               # (n_bx,n_by,n_bz) int8: -1 interior, 0 active, +1 exterior
    niva: np.ndarray               # (n_bx,n_by,n_bz) int8: level per active brick, -1 elsewhere
    tiles: dict                    # brick linear id -> (n_l+1)^3 float32 samples, n_l = block*2^l
    margin_mm: float
    stats: dict = field(default_factory=dict)

    @property
    def L(self) -> float:
        return self.block * self.h_grov

    def pitch_of(self, lvl: int) -> float:
        return self.h_grov / float(1 << int(lvl))


# ================================================================================================
# Build
# ================================================================================================
def _krokta_trianglar(V, T, crease_grad=CREASE_GRAD):
    """Prefilter: the triangles that take part in a smooth (non-crease) face pair.

    A brick touched by none of them carries no curvature, so its level is the coarsest one and
    `minsta_feature_radie` need not be called for it at all. Returns a boolean mask over T.
    """
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, dtype=np.float64), np.asarray(T, dtype=np.int64), process=False)
    par, vinkel = m.face_adjacency, m.face_adjacency_angles
    mask = np.zeros(len(T), dtype=bool)
    if len(par) == 0:
        return mask
    sel = (vinkel > 1.0e-4) & (vinkel < math.radians(crease_grad))
    if sel.any():
        mask[np.unique(par[sel].reshape(-1))] = True
    return mask


def _tri_brickar(V, T, lo, L, brick_shape, halo_mm):
    """Maps each triangle to the bricks its bounding box (grown by halo_mm) overlaps.

    Returns a dict brick-linear-id -> list of triangle indices.
    """
    Vf = np.asarray(V, dtype=np.float64)
    Ti = np.asarray(T, dtype=np.int64)
    tmin = Vf[Ti].min(axis=1) - halo_mm
    tmax = Vf[Ti].max(axis=1) + halo_mm
    b0 = np.floor((tmin - lo) / L).astype(np.int64)
    b1 = np.floor((tmax - lo) / L).astype(np.int64)
    n_bx, n_by, n_bz = brick_shape
    b0 = np.clip(b0, 0, [n_bx - 1, n_by - 1, n_bz - 1])
    b1 = np.clip(b1, 0, [n_bx - 1, n_by - 1, n_bz - 1])
    ut = {}
    for t in range(len(Ti)):
        for bi in range(b0[t, 0], b1[t, 0] + 1):
            for bj in range(b0[t, 1], b1[t, 1] + 1):
                for bk in range(b0[t, 2], b1[t, 2] + 1):
                    ut.setdefault((bi * n_by + bj) * n_bz + bk, []).append(t)
    return ut


def _lokal_feature_radie(V, T, tri_idx):
    """Local radius of curvature from the submesh of the given triangles.

    The measurement itself is `minsta_feature_radie` from the mesh path, unchanged; only the mesh it
    is handed is local. Returns (r_mm or None, n_triangles).
    """
    Ti = np.asarray(T, dtype=np.int64)[np.asarray(tri_idx, dtype=np.int64)]
    vid, inv = np.unique(Ti.reshape(-1), return_inverse=True)
    Vs = np.asarray(V, dtype=np.float64)[vid]
    Ts = inv.reshape(-1, 3)
    r, _d = M2S.minsta_feature_radie(Vs, Ts)
    return r, int(len(Ts))


def _niva_for_radie(r_mm, h_grov, faktor=MULTIRES_PITCH_FAKTOR):
    """Coarsest level whose pitch resolves a feature of radius r_mm; clamps at the finest level."""
    if r_mm is None or r_mm <= 0.0:
        return 0, False
    for lvl in NIVAER:
        if h_grov / float(1 << lvl) <= r_mm / faktor:
            return lvl, False
    return NIVAER[-1], True


def _balansera(niva, aktiv):
    """2:1 balance plus one brick of level dilation.

    The dilation puts the fine pitch on the bricks the feature's wall reaches into, and the balance
    step leaves no face with a jump of more than one level, which is what the overlap band assumes
    one of (it works for a jump of two as well, but the field then has two different pitch changes
    inside one brick face). Returns the balanced level array.
    """
    n = niva.copy()
    n[~aktiv] = -1
    grown = ndimage.grey_dilation(np.where(aktiv, n, -1), size=(3, 3, 3), mode="nearest")
    n = np.where(aktiv, np.maximum(n, grown), -1).astype(np.int8)
    for _ in range(len(NIVAER)):
        omg = ndimage.grey_dilation(np.where(aktiv, n, -1), footprint=ndimage.generate_binary_structure(3, 1),
                                    mode="nearest")
        ny = np.where(aktiv, np.maximum(n, omg - 1), -1).astype(np.int8)
        if np.array_equal(ny, n):
            break
        n = ny
    return n


def _sd_pa_fonster(V, T, h, origin, shape, ytkorrektion=True):
    """Signed field on an explicit window at an explicit pitch, by the mesh path's own signing.

    Inside/outside comes from `solid_via_stralvindning` (whose +Z rays see the whole triangle soup,
    so a window is not a truncated body), the distance is the same two-transform construction as
    `surface_raster_and_flood`, and the half-pitch surface correction is applied at this window's own
    pitch. Returns the float32 array of `shape`.
    """
    solid, _diag = M2S.solid_via_stralvindning(V, T, h, np.asarray(origin, dtype=np.float64), shape)
    d_out = ndimage.distance_transform_edt(~solid, sampling=(h, h, h)).astype(np.float32)
    d_in = ndimage.distance_transform_edt(solid, sampling=(h, h, h)).astype(np.float32)
    sd = (d_out - d_in).astype(np.float32)
    if ytkorrektion:
        sd = (sd - np.sign(sd) * np.float32(0.5 * h)).astype(np.float32)
    return sd


def _lagg_overlappsband(F: MultiresField):
    """Overwrites every fine brick's face layer with the coarse neighbour's field on that face.

    Returns the number of faces corrected.
    """
    n_bx, n_by, n_bz = F.brick_shape
    n_faces = 0
    for lin, tile in F.tiles.items():
        bi = lin // (n_by * n_bz)
        bj = (lin // n_bz) % n_by
        bk = lin % n_bz
        lvl = int(F.niva[bi, bj, bk])
        if lvl == 0:
            continue
        n_l = F.block * (1 << lvl)
        b_origin = F.lo + np.array([bi, bj, bk], dtype=np.float64) * F.L
        for ax in range(3):
            for sida in (-1, +1):
                nb = [bi, bj, bk]
                nb[ax] += sida
                if not (0 <= nb[0] < n_bx and 0 <= nb[1] < n_by and 0 <= nb[2] < n_bz):
                    continue
                nlin = (nb[0] * n_by + nb[1]) * n_bz + nb[2]
                if F.kind[nb[0], nb[1], nb[2]] != 0:
                    continue
                lvl_n = int(F.niva[nb[0], nb[1], nb[2]])
                if lvl_n >= lvl:
                    continue
                # sample points of this brick's face layer
                idx = [np.arange(n_l + 1, dtype=np.float64)] * 3
                idx[ax] = np.array([0.0 if sida < 0 else float(n_l)])
                gi, gj, gk = np.meshgrid(*idx, indexing="ij")
                pts = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1) * F.pitch_of(lvl) + b_origin
                nb_origin = F.lo + np.array(nb, dtype=np.float64) * F.L
                lok = (pts - nb_origin) / F.pitch_of(lvl_n)
                v = ndimage.map_coordinates(F.tiles[nlin], lok.T, order=1, mode="nearest")
                sl = [slice(None)] * 3
                sl[ax] = 0 if sida < 0 else n_l
                tile[tuple(sl)] = v.reshape(tile[tuple(sl)].shape).astype(np.float32)
                n_faces += 1
    return n_faces


def bygg_multires(wp, V, T, h_grov, lo, device, block=BLOCK, faktor=MULTIRES_PITCH_FAKTOR,
                  overlappsband=True, sd_grov=None, gmin=None, shape_l=None):
    """Builds the per-block-pitch field.

    Takes the mesh, the coarse pitch, the grid origin and the device; `sd_grov`/`gmin`/`shape_l` let a
    caller hand in a coarse field it has already built (the coarse global run of the comparison).
    Returns (MultiresField, timings dict).
    """
    tid = {}
    t0 = time.time()
    if sd_grov is None:
        gmin, shape_l, _yta, _solid, sd_grov = M2S.surface_raster_and_flood(
            V, T, h_grov, lo, wp=wp, device=device)
    tid["t_grov_falt_s"] = time.time() - t0

    t0 = time.time()
    sf = M2S.klassificera_och_evaluera_fran_tatt_falt(wp, sd_grov, h_grov, block, device)
    tid["t_klassificering_s"] = time.time() - t0
    n_bx, n_by, n_bz = sf["n_bx"], sf["n_by"], sf["n_bz"]
    kind = sf["kind"].reshape(n_bx, n_by, n_bz).astype(np.int8)
    aktiv = kind == 0
    origin_l = np.asarray(lo, dtype=np.float64) + np.asarray(gmin, dtype=np.float64) * h_grov

    # level per active brick, from the local radius of curvature
    t0 = time.time()
    kroka = _krokta_trianglar(V, T)
    tri_idx = np.where(kroka)[0]
    L = block * h_grov
    b2t = _tri_brickar(V, np.asarray(T)[tri_idx], origin_l, L, (n_bx, n_by, n_bz), halo_mm=h_grov)
    niva = np.zeros((n_bx, n_by, n_bz), dtype=np.int8)
    niva[~aktiv] = -1
    n_klamda = 0
    r_per_brick = {}
    for lin, tl in b2t.items():
        bi = lin // (n_by * n_bz)
        bj = (lin // n_bz) % n_by
        bk = lin % n_bz
        if not aktiv[bi, bj, bk]:
            continue
        r, _n = _lokal_feature_radie(V, np.asarray(T)[tri_idx[np.asarray(tl)]], np.arange(len(tl)))
        lvl, klamd = _niva_for_radie(r, h_grov, faktor)
        n_klamda += int(klamd)
        niva[bi, bj, bk] = lvl
        r_per_brick[lin] = (None if r is None else float(r), int(lvl))
    niva = _balansera(niva, aktiv)
    tid["t_nivaval_s"] = time.time() - t0

    # level 0 tiles straight out of the coarse field (padded by one sample: the overlap band)
    t0 = time.time()
    nx, ny, nz = shape_l
    padded = np.empty((n_bx * block + 1, n_by * block + 1, n_bz * block + 1), dtype=np.float32)
    padded[:] = np.pad(sd_grov, ((0, padded.shape[0] - nx), (0, padded.shape[1] - ny),
                                 (0, padded.shape[2] - nz)), mode="edge")
    tiles = {}
    ids = np.argwhere(aktiv)
    for bi, bj, bk in ids:
        if niva[bi, bj, bk] != 0:
            continue
        lin = (int(bi) * n_by + int(bj)) * n_bz + int(bk)
        tiles[lin] = np.ascontiguousarray(
            padded[bi * block:bi * block + block + 1,
                   bj * block:bj * block + block + 1,
                   bk * block:bk * block + block + 1])
    tid["t_niva0_tiles_s"] = time.time() - t0

    # refined levels, one window per connected cluster of bricks at that level
    t0 = time.time()
    kluster_info = []
    for lvl in NIVAER[1:]:
        mask = (niva == lvl)
        if not mask.any():
            continue
        h_l = h_grov / float(1 << lvl)
        n_l = block * (1 << lvl)
        lab, n_lab = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
        for c in range(1, n_lab + 1):
            sel = np.argwhere(lab == c)
            b0 = sel.min(axis=0)
            b1 = sel.max(axis=0)
            win_origin = origin_l + b0.astype(np.float64) * L - KLUSTER_PAD_VOX * h_l
            ext = (b1 - b0 + 1) * n_l + 1 + 2 * KLUSTER_PAD_VOX
            shape_w = tuple(int(x) for x in ext)
            sd_w = _sd_pa_fonster(V, T, h_l, win_origin, shape_w)
            for bi, bj, bk in sel:
                lin = (int(bi) * n_by + int(bj)) * n_bz + int(bk)
                o = (np.array([bi, bj, bk]) - b0) * n_l + KLUSTER_PAD_VOX
                tiles[lin] = np.ascontiguousarray(
                    sd_w[o[0]:o[0] + n_l + 1, o[1]:o[1] + n_l + 1, o[2]:o[2] + n_l + 1])
            kluster_info.append(dict(niva=int(lvl), n_brickor=int(len(sel)), fonster=list(shape_w),
                                     fonster_voxlar=int(np.prod(shape_w))))
    tid["t_forfining_s"] = time.time() - t0

    F = MultiresField(lo=origin_l, h_grov=float(h_grov), block=block, brick_shape=(n_bx, n_by, n_bz),
                      kind=kind, niva=niva, tiles=tiles, margin_mm=float(sf["margin_mm"]))
    t0 = time.time()
    n_faces = _lagg_overlappsband(F) if overlappsband else 0
    tid["t_overlappsband_s"] = time.time() - t0
    tid["t_bygg_total_s"] = float(sum(tid.values()))

    per_niva = {int(l): int(np.count_nonzero(niva == l)) for l in NIVAER}
    F.stats = dict(n_brickor=int(niva.size), n_aktiva=int(aktiv.sum()),
                   n_per_niva=per_niva, n_klamda_brickor=int(n_klamda),
                   aktiva_voxlar=int(aktiva_voxlar(F)), n_overlappsfaces=int(n_faces),
                   kluster=kluster_info, h_grov=float(h_grov),
                   h_per_niva=[float(h_grov / (1 << l)) for l in NIVAER],
                   origin=[float(x) for x in origin_l], shape_l=[int(x) for x in shape_l],
                   dense_voxlar_grov=int(nx * ny * nz))
    return F, tid


def aktiva_voxlar(F: MultiresField) -> int:
    """Stored voxels of the structure: sum over active bricks of (block*2^level)^3.

    The overlap sample is not counted; it is the same voxel as the neighbour's first one and is the
    price of the continuity band, reported separately as `lagrade_sampel`.
    """
    tot = 0
    for l in NIVAER:
        tot += int(np.count_nonzero(F.niva == l)) * (F.block * (1 << l)) ** 3
    return tot


def lagrade_sampel(F: MultiresField) -> int:
    """Samples actually stored, overlap band included."""
    return int(sum(t.size for t in F.tiles.values()))


# ================================================================================================
# Queries
# ================================================================================================
def _brick_av_punkt(F: MultiresField, pts):
    """Brick index per point, preferring the active brick when a point sits exactly on a face."""
    rel = (np.asarray(pts, dtype=np.float64) - F.lo) / F.L
    b = np.floor(rel).astype(np.int64)
    shp = np.array(F.brick_shape, dtype=np.int64)
    utanfor = np.any((b < 0) | (b >= shp), axis=1)
    b = np.clip(b, 0, shp - 1)
    inaktiv = F.kind[b[:, 0], b[:, 1], b[:, 2]] != 0
    for ax in range(3):
        pa_plan = (rel[:, ax] == np.floor(rel[:, ax])) & (b[:, ax] > 0) & inaktiv & ~utanfor
        if not pa_plan.any():
            continue
        cand = b[pa_plan].copy()
        cand[:, ax] -= 1
        ok = F.kind[cand[:, 0], cand[:, 1], cand[:, 2]] == 0
        idx = np.where(pa_plan)[0][ok]
        b[idx, ax] -= 1
        inaktiv = F.kind[b[:, 0], b[:, 1], b[:, 2]] != 0
    return b, utanfor


def avstand(F: MultiresField, punkter):
    """Signed distance at world points, each resolved at its own brick's pitch.

    Points in a brick with no stored tile are answered with that brick's Lipschitz bound
    (+/- margin_mm), the strongest statement the classification makes about it. Returns mm values.
    """
    pts = np.atleast_2d(np.asarray(punkter, dtype=np.float64))
    b, utanfor = _brick_av_punkt(F, pts)
    n_by, n_bz = F.brick_shape[1], F.brick_shape[2]
    lin = (b[:, 0] * n_by + b[:, 1]) * n_bz + b[:, 2]
    kd = F.kind[b[:, 0], b[:, 1], b[:, 2]]
    ut = np.where(kd < 0, -F.margin_mm, F.margin_mm).astype(np.float64)
    ut[utanfor] = F.margin_mm
    akt = (kd == 0) & ~utanfor
    if akt.any():
        lin_a = lin[akt]
        uniq, inv = np.unique(lin_a, return_inverse=True)
        pts_a = pts[akt]
        vals = np.empty(len(pts_a), dtype=np.float64)
        for u_i, u in enumerate(uniq):
            sel = inv == u_i
            bi = u // (n_by * n_bz)
            bj = (u // n_bz) % n_by
            bk = u % n_bz
            lvl = int(F.niva[bi, bj, bk])
            o = F.lo + np.array([bi, bj, bk], dtype=np.float64) * F.L
            lok = (pts_a[sel] - o) / F.pitch_of(lvl)
            vals[sel] = ndimage.map_coordinates(F.tiles[u], lok.T, order=1, mode="nearest")
        ut[akt] = vals
    return ut


def gradient(F: MultiresField, punkter):
    """Central-difference gradient at each point's own brick pitch, clamped inside the brick.

    Returns an (n,3) array; the step is the brick's pitch and the two probe points are clamped into
    the brick, so the difference is always taken inside one resolution.
    """
    pts = np.atleast_2d(np.asarray(punkter, dtype=np.float64))
    b, _ut = _brick_av_punkt(F, pts)
    lvl = F.niva[b[:, 0], b[:, 1], b[:, 2]].astype(np.int64)
    h = F.h_grov / np.power(2.0, np.maximum(lvl, 0))
    o = F.lo + b.astype(np.float64) * F.L
    g = np.zeros((len(pts), 3), dtype=np.float64)
    for ax in range(3):
        d = np.zeros((len(pts), 3))
        d[:, ax] = h
        pa = np.minimum(pts + 0.5 * d, o + F.L)
        pb = np.maximum(pts - 0.5 * d, o)
        steg = pa[:, ax] - pb[:, ax]
        g[:, ax] = (avstand(F, pa) - avstand(F, pb)) / np.maximum(steg, 1e-12)
    return g


# ================================================================================================
# Continuity across a pitch change
# ================================================================================================
def kontinuitetscheck(F: MultiresField, eps_frac=1.0e-3, max_punkter_per_face=None):
    """Max jump of the delivered field across every block face where the pitch changes.

    Every such face is sampled at the finer side's own face lattice and the query is evaluated a hair
    (eps_frac of the finer pitch) on each side. The declared bound is the coarser pitch / 2: a
    discontinuity larger than that would be bigger than the coarse side's own quantisation.
    Returns a dict with the number of faces, the max and mean jump and the verdict.
    """
    n_bx, n_by, n_bz = F.brick_shape
    max_hopp = 0.0
    summa = 0.0
    n_pts = 0
    n_faces = 0
    varsta = None
    grans = 0.0
    for lin, _t in F.tiles.items():
        bi = lin // (n_by * n_bz)
        bj = (lin // n_bz) % n_by
        bk = lin % n_bz
        lvl = int(F.niva[bi, bj, bk])
        n_l = F.block * (1 << lvl)
        h_f = F.pitch_of(lvl)
        b_origin = F.lo + np.array([bi, bj, bk], dtype=np.float64) * F.L
        for ax in range(3):
            for sida in (-1, +1):
                nb = [bi, bj, bk]
                nb[ax] += sida
                if not (0 <= nb[0] < n_bx and 0 <= nb[1] < n_by and 0 <= nb[2] < n_bz):
                    continue
                if F.kind[nb[0], nb[1], nb[2]] != 0:
                    continue
                lvl_n = int(F.niva[nb[0], nb[1], nb[2]])
                if lvl_n >= lvl:
                    continue                      # the finer side owns the face
                idx = [np.arange(n_l + 1, dtype=np.float64)] * 3
                idx[ax] = np.array([0.0 if sida < 0 else float(n_l)])
                gi, gj, gk = np.meshgrid(*idx, indexing="ij")
                pts = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1) * h_f + b_origin
                if max_punkter_per_face is not None and len(pts) > max_punkter_per_face:
                    steg = int(math.ceil(len(pts) / max_punkter_per_face))
                    pts = pts[::steg]
                e = np.zeros(3)
                e[ax] = eps_frac * h_f
                v_in = avstand(F, pts - sida * e)     # inside the fine brick
                v_ut = avstand(F, pts + sida * e)     # inside the coarse neighbour
                hopp = np.abs(v_in - v_ut)
                n_faces += 1
                n_pts += len(pts)
                summa += float(hopp.sum())
                if float(hopp.max()) > max_hopp:
                    max_hopp = float(hopp.max())
                    varsta = dict(brick=[int(bi), int(bj), int(bk)], axel=ax, sida=sida,
                                  niva=lvl, niva_granne=lvl_n)
                grans = max(grans, 0.5 * F.pitch_of(lvl_n))
    return dict(n_pitchbyten_faces=n_faces, n_provpunkter=n_pts,
                max_hopp_mm=max_hopp, medel_hopp_mm=(summa / n_pts if n_pts else 0.0),
                grans_mm=grans, varsta=varsta, GRON=bool(n_faces == 0 or max_hopp <= grans))


# ================================================================================================
# Measurement: the note-11 table, evaluated through a query instead of a dense array
# ================================================================================================
def _prov_fran_tat(sd, origin_l, pitch):
    """Query wrapper for a single-pitch dense field (the reference configurations)."""
    return lambda pts: SMF._prov(sd, origin_l, pitch, pts)


def mat_hal_prov(prov, x_c, R, tjocklek, h_lokal, h_probe):
    """The note-11 measures for one hole, taken through a query function.

    radial deviation (|sd| on the wall / R), the zero crossing along the same rays, the one-ring test
    at R +/- h_lokal and the hole volume from the void area at mid-thickness on a probe lattice of
    pitch h_probe, which is the same lattice for every configuration so the three are comparable.
    Returns a dict with the same keys as the note-11 row.
    """
    vinklar = np.linspace(0.0, 2.0 * math.pi, N_VINKLAR, endpoint=False)
    cos, sin = np.cos(vinklar), np.sin(vinklar)

    def cirkel(r):
        return np.stack([x_c + r * cos, r * sin, np.zeros(N_VINKLAR)], axis=1)

    pa_vaggen = prov(cirkel(R))
    avvikelse = float(np.max(np.abs(pa_vaggen)) / R)

    rr = np.linspace(max(R - 2.0 * h_lokal, 0.1 * R), R + 2.0 * h_lokal, 81)
    dr, n_traff = [], 0
    for a in vinklar:
        v = prov(np.stack([x_c + rr * np.cos(a), rr * np.sin(a), np.zeros_like(rr)], axis=1))
        k = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
        if len(k):
            n_traff += 1
            i = k[0]
            w = v[i] / (v[i] - v[i + 1])
            dr.append(abs(rr[i] + w * (rr[i + 1] - rr[i]) - R))
    avvikelse_noll = float(max(dr) / R) if dr else None

    r_inre = R - h_lokal
    ring_matbar = r_inre > 0.25 * R
    if ring_matbar:
        inre = prov(cirkel(r_inre))
        yttre = prov(cirkel(R + h_lokal))
        byten = SMF._teckenbyten(inre) + SMF._teckenbyten(yttre)
        en_ring = bool(byten == 0 and np.all(inre > 0.0) and np.all(yttre < 0.0))
    else:
        byten, en_ring = None, None

    rad = R + 5.0
    ax = np.arange(-rad, rad + 0.5 * h_probe, h_probe)
    gx, gy = np.meshgrid(x_c + ax, ax, indexing="ij")
    inne = (gx - x_c) ** 2 + gy ** 2 <= rad * rad
    pts = np.stack([gx[inne], gy[inne], np.zeros(int(inne.sum()))], axis=1)
    v = prov(pts)
    area = float(np.count_nonzero(v > 0.0)) * h_probe * h_probe
    vol = area * tjocklek
    facit = math.pi * R * R * tjocklek
    return dict(R_mm=R, pitch_mm=h_lokal, R_per_h=R / h_lokal, radiell_avvikelse=avvikelse,
                radiell_avvikelse_nollgenomgang=avvikelse_noll, n_radier_med_nollgenomgang=n_traff,
                n_teckenbyten=byten, en_ring=en_ring, ring_matbar=bool(ring_matbar),
                vol_rekonstruerad_mm3=vol, vol_facit_mm3=facit, volym_fel=(vol - facit) / facit)


def _tabell(fall):
    """Prints the three-way hole table; returns the text."""
    rader = ["hole table: radial deviation of the zero level set (|sd|/R), one ring, hole volume error",
             "  R [mm] | " + " | ".join(f["namn"].center(24) for f in fall)]
    for k, R in enumerate(PART.HOLE_RADII_MM):
        celler = []
        for f in fall:
            d = f["rader"][k]
            ring = "-" if d["en_ring"] is None else ("1 ring" if d["en_ring"] else f"{d['n_teckenbyten']} flips")
            celler.append(f"{100 * d['radiell_avvikelse']:6.1f}% {ring:>8s} {100 * d['volym_fel']:+7.1f}%")
        rader.append(f"  {R:6.0f} | " + " | ".join(celler))
    text = "\n".join(rader)
    print(text)
    return text


def _niva_vid_hal(F: MultiresField, x_c, R):
    """Level of the brick that carries the hole wall at mid-thickness (the pitch that hole is seen at)."""
    p = np.array([[x_c + R, 0.0, 0.0]])
    b, _ = _brick_av_punkt(F, p)
    lvl = int(F.niva[b[0, 0], b[0, 1], b[0, 2]])
    return max(lvl, 0)


def _fingerprint(F: MultiresField):
    """Bit-level fingerprint of the whole structure, for the determinism certificate."""
    hsh = hashlib.sha256()
    hsh.update(F.kind.tobytes())
    hsh.update(F.niva.tobytes())
    for lin in sorted(F.tiles):
        hsh.update(np.int64(lin).tobytes())
        hsh.update(F.tiles[lin].tobytes())
    return hsh.hexdigest()


# ================================================================================================
# Runs
# ================================================================================================
def _global_fall(wp, device, V, T, h, lo, namn):
    """One single-pitch reference build; returns a dict with the field, the query and the counts."""
    t0 = time.time()
    gmin, shape_l, _yta, _solid, sd = M2S.surface_raster_and_flood(V, T, h, lo, wp=wp, device=device)
    t_falt = time.time() - t0
    t0 = time.time()
    sf = M2S.klassificera_och_evaluera_fran_tatt_falt(wp, sd, h, BLOCK, device)
    t_kl = time.time() - t0
    origin_l = np.asarray(lo, dtype=np.float64) + np.asarray(gmin, dtype=np.float64) * h
    sd_rek = SMF.tat_sd_fran_gles(sf, shape_l, BLOCK)
    return dict(namn=namn, pitch=float(h), origin=origin_l, shape_l=[int(x) for x in shape_l],
                n_blocks=int(sf["n_blocks"]), n_active=int(sf["n_active"]),
                aktiva_voxlar=int(sf["n_active"]) * BLOCK ** 3,
                t_bygg_s=float(t_falt + t_kl), t_falt_s=float(t_falt), t_klass_s=float(t_kl),
                prov=_prov_fran_tat(sd_rek, origin_l, h), sd=sd, gmin=gmin, sf=sf)


def kor_platta(wp, device, h_fin=None):
    """The three-way comparison on the holed plate. Returns the result dict."""
    V, T = PART.mesh()
    if h_fin is None:
        h_fin, _d = M2S.valj_pitch_for_feature(V, T, max(SMF.PITCHAR_MM), "auto")
    h_grov = 4.0 * h_fin
    lo = np.asarray(V, dtype=np.float64).min(axis=0) - 3.0 * h_grov
    h_probe = 0.5 * h_fin

    grov = _global_fall(wp, device, V, T, h_grov, lo, "global-coarse h")
    fin = _global_fall(wp, device, V, T, h_fin, lo, "global-fine h/4")

    F, tid = bygg_multires(wp, V, T, h_grov, lo, device,
                           sd_grov=grov["sd"], gmin=grov["gmin"], shape_l=grov["shape_l"])
    # the coarse field is handed in, so its cost is added back: the multires build time reported
    # here is the full path from the mesh, coarse field included.
    mr = dict(namn="multires {h,h/2,h/4}", pitch=None, aktiva_voxlar=aktiva_voxlar(F),
              t_bygg_s=float(tid["t_bygg_total_s"] + grov["t_bygg_s"]), tid=tid, stats=F.stats,
              prov=lambda pts: avstand(F, pts))

    fall = []
    for f in (mr, fin, grov):
        rader = []
        for x_c, R in zip(PART.hole_centres_mm(), PART.HOLE_RADII_MM):
            h_lokal = F.pitch_of(_niva_vid_hal(F, x_c, R)) if f is mr else f["pitch"]
            rader.append(mat_hal_prov(f["prov"], x_c, R, PART.THICKNESS_MM, h_lokal, h_probe))
        f["rader"] = rader
        fall.append(f)

    text = _tabell(fall)
    kont_med = kontinuitetscheck(F, max_punkter_per_face=2000)

    G, _tid2 = bygg_multires(wp, V, T, h_grov, lo, device, overlappsband=False,
                             sd_grov=grov["sd"], gmin=grov["gmin"], shape_l=grov["shape_l"])
    kont_utan = kontinuitetscheck(G, max_punkter_per_face=2000)

    grind_rader = []
    for d_m, d_f in zip(mr["rader"], fin["rader"]):
        if d_m["R_mm"] >= GATE_RADIE_MM:
            grind_rader.append(dict(R_mm=d_m["R_mm"],
                                    multires_avvikelse=d_m["radiell_avvikelse"],
                                    fin_avvikelse=d_f["radiell_avvikelse"],
                                    pass_avvikelse=bool(d_m["radiell_avvikelse"] <= GATE_AVVIKELSE),
                                    pass_ring=bool(d_m["en_ring"]),
                                    pass_volym_vs_fin=bool(abs(d_m["volym_fel"]) <=
                                                           max(abs(d_f["volym_fel"]) * 1.5, 0.01))))
    kvot = mr["aktiva_voxlar"] / float(fin["aktiva_voxlar"])
    gate = dict(rader=grind_rader, minneskvot=float(kvot), minneskvot_tak=GATE_MINNESKVOT,
                pass_minne=bool(kvot <= GATE_MINNESKVOT),
                pass_kontinuitet=bool(kont_med["GRON"]),
                pass_tabell=bool(all(g["pass_avvikelse"] and g["pass_ring"] for g in grind_rader)))
    gate["ALL_PASS"] = bool(gate["pass_minne"] and gate["pass_tabell"] and gate["pass_kontinuitet"])

    pts = np.stack([PART.hole_centres_mm()[4] + np.linspace(16.0, 16.0, 32),
                    np.zeros(32), np.linspace(-2.0, 2.0, 32)], axis=1)
    g = gradient(F, pts)
    grad_norm = float(np.median(np.linalg.norm(g, axis=1)))

    ut = dict(del_namn="holed_plate_v1", h_grov=float(h_grov), h_fin=float(h_fin),
              h_probe=float(h_probe), n_trianglar=int(len(T)),
              tabell=text,
              fall=[dict(namn=f["namn"], pitch=f.get("pitch"), aktiva_voxlar=int(f["aktiva_voxlar"]),
                         t_bygg_s=float(f["t_bygg_s"]), rader=f["rader"],
                         stats=f.get("stats"), tid=f.get("tid"),
                         n_active_blocks=f.get("n_active"), shape_l=f.get("shape_l"))
                    for f in fall],
              lagrade_sampel_multires=lagrade_sampel(F),
              minneskvot_multires_vs_fin=float(kvot),
              kontinuitet_med_band=kont_med, kontinuitet_utan_band=kont_utan,
              gradient_norm_median_pa_vagg=grad_norm,
              fingerprint=_fingerprint(F), gate=gate)
    return ut, F, (V, T, h_grov, lo)


def _bracket_mesh(sokvag=None):
    """Mesh of the synthetic bracket, tessellated from the recipe build. Returns (V, T)."""
    import tempfile
    import trimesh
    from build123d import export_stl
    sys.path.insert(0, os.path.join(ROOT, "examples", "parts"))
    import bracket_part_v1 as BP
    part = BP.build({"plate_w": 150.0, "plate_h": 100.0, "thk": 10.0, "hole_d": 9.0})
    p = sokvag or os.path.join(tempfile.gettempdir(), "faltkarna_v1_multires_bracket.stl")
    export_stl(part, p, tolerance=0.02, angular_tolerance=0.05)
    m = trimesh.load(p, process=False)
    m.merge_vertices()
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def kor_bracket(wp, device):
    """The same three configurations on the synthetic bracket.

    The bracket has no through-hole table of its own, so accuracy is measured where the mesh is: the
    field is sampled at the centroid of every triangle, where the true signed distance is 0, and the
    deviation is reported in mm and relative to the coarse pitch.
    """
    V, T = _bracket_mesh()
    h_fin, d = M2S.valj_pitch_for_feature(V, T, 2.0, "auto")
    h_grov = 4.0 * h_fin
    lo = np.asarray(V, dtype=np.float64).min(axis=0) - 3.0 * h_grov
    grov = _global_fall(wp, device, V, T, h_grov, lo, "global-coarse h")
    fin = _global_fall(wp, device, V, T, h_fin, lo, "global-fine h/4")
    F, tid = bygg_multires(wp, V, T, h_grov, lo, device,
                           sd_grov=grov["sd"], gmin=grov["gmin"], shape_l=grov["shape_l"])
    C = np.asarray(V)[np.asarray(T)].mean(axis=1)
    rng = np.random.default_rng(20260912)
    C = C[rng.choice(len(C), size=min(4000, len(C)), replace=False)]
    rows = []
    for namn, prov, h, vox, t_b in (
            ("multires {h,h/2,h/4}", lambda p: avstand(F, p), h_grov, aktiva_voxlar(F),
             tid["t_bygg_total_s"] + grov["t_bygg_s"]),
            ("global-fine h/4", fin["prov"], h_fin, fin["aktiva_voxlar"], fin["t_bygg_s"]),
            ("global-coarse h", grov["prov"], h_grov, grov["aktiva_voxlar"], grov["t_bygg_s"])):
        v = np.abs(prov(C))
        rows.append(dict(namn=namn, pitch_mm=float(h), aktiva_voxlar=int(vox), t_bygg_s=float(t_b),
                         yt_avvikelse_median_mm=float(np.median(v)),
                         yt_avvikelse_p95_mm=float(np.percentile(v, 95)),
                         yt_avvikelse_median_per_h_grov=float(np.median(v) / h_grov)))
    kont = kontinuitetscheck(F, max_punkter_per_face=2000)
    return dict(del_namn="bracket_part_v1", h_grov=float(h_grov), h_fin=float(h_fin),
                r_min_mm=d.get("r_min_mm"), n_trianglar=int(len(T)),
                n_provpunkter=int(len(C)), rader=rows,
                minneskvot_multires_vs_fin=float(aktiva_voxlar(F) / max(fin["aktiva_voxlar"], 1)),
                stats=F.stats, tid=tid, kontinuitet=kont, fingerprint=_fingerprint(F))


def kor(med_bracket=True):
    """Full measurement: plate table, memory, build time, continuity, determinism, bracket."""
    wp, have_cuda = M2S._import_warp()
    device = "cuda:0" if have_cuda else "cpu"
    platta, F, (V, T, h_grov, lo) = kor_platta(wp, device)

    F2, _t = bygg_multires(wp, V, T, h_grov, lo, device)
    det = dict(fingerprint_kor1=platta["fingerprint"], fingerprint_kor2=_fingerprint(F2),
               bit_identiska=bool(platta["fingerprint"] == _fingerprint(F2)))

    print()
    print("memory and build time (holed plate)")
    print("  configuration            | active voxels | ratio vs fine | build s")
    for f in platta["fall"]:
        print(f"  {f['namn']:<24s} | {f['aktiva_voxlar']:13d} | "
              f"{f['aktiva_voxlar'] / platta['fall'][1]['aktiva_voxlar']:13.3f} | {f['t_bygg_s']:7.2f}")
    k = platta["kontinuitet_med_band"]
    ku = platta["kontinuitet_utan_band"]
    print(f"  continuity at pitch changes: {k['n_pitchbyten_faces']} faces, {k['n_provpunkter']} points, "
          f"max jump {k['max_hopp_mm']:.6f} mm, bound {k['grans_mm']:.4f} mm "
          f"({'GREEN' if k['GRON'] else 'RED'}); without the overlap band "
          f"{ku['max_hopp_mm']:.6f} mm ({'GREEN' if ku['GRON'] else 'RED'})")
    print(f"  two builds bit-identical: {det['bit_identiska']}")

    bracket = kor_bracket(wp, device) if med_bracket else None
    if bracket:
        print()
        print("synthetic bracket: |sd| at triangle centroids (true value 0), active voxels, build s")
        for r in bracket["rader"]:
            print(f"  {r['namn']:<24s} | median {r['yt_avvikelse_median_mm']:.4f} mm "
                  f"| p95 {r['yt_avvikelse_p95_mm']:.4f} mm | {r['aktiva_voxlar']:9d} vox "
                  f"| {r['t_bygg_s']:6.2f} s")
        print(f"  memory ratio multires/fine {bracket['minneskvot_multires_vs_fin']:.3f}, "
              f"continuity max jump {bracket['kontinuitet']['max_hopp_mm']:.6f} mm "
              f"(bound {bracket['kontinuitet']['grans_mm']:.4f})")

    ut = dict(device=device, platta=platta, bracket=bracket, determinism=det,
              gate=dict(platta=platta["gate"],
                        bit_identiska=det["bit_identiska"],
                        ALL_PASS=bool(platta["gate"]["ALL_PASS"] and det["bit_identiska"])))
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(UT_JSON, "w") as f:
        json.dump(ut, f, indent=1, default=float)
    print(f"\n-> wrote {os.path.relpath(UT_JSON, ROOT)}")
    g = platta["gate"]
    print(f"gate: table {g['pass_tabell']}, memory ratio {g['minneskvot']:.3f} <= "
          f"{g['minneskvot_tak']} {g['pass_minne']}, continuity {g['pass_kontinuitet']}, "
          f"determinism {det['bit_identiska']}")
    print("ALL_PASS" if ut["gate"]["ALL_PASS"] else "GATE FAILED")
    return ut


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--multires", action="store_true", help="run the three-way measurement and the gate")
    ap.add_argument("--no-bracket", action="store_true", help="skip the bracket part of the run")
    a = ap.parse_args()
    res = kor(med_bracket=not a.no_bracket)
    if not res["gate"]["ALL_PASS"]:
        raise SystemExit("faltkarna_v1_multires gate FAILED")
