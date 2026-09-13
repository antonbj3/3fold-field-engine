#!/usr/bin/env python3
"""field_to_recipe_v1.py -- the reverse direction: a block-sparse SDF back to an op list.

The forward chain is closed (B-rep -> attributed adjacency graph -> recogniser -> recipe -> executed
solid -> field). The reverse was missing: an optimised field could not be turned into operations, so
a field result was not manufacturable through the recipe chain. This module supplies the one missing
link, mesh -> B-rep, and then hands the fitted B-rep to the EXISTING recogniser and synthesiser
(brep_recept_synth_v1.synth); no second synthesiser is written here and no op type or parameter is
added to cad_op_schema_v1.OP_SPECS.

Stages:
  1. block-sparse SDF (faltkarna_v1_mesh_to_sdf.klassificera_och_evaluera_fran_tatt_falt, the same
     structure the multires module stores per level) -> dense window (dense_fran_blocksparse);
  2. zero level set -> mesh (skimage marching cubes, nollnivamesh);
  3. region growing on triangle normals -> planar regions, then cylinder fits on what is left
     (passa_primitiver): every region carries its own residual, and the AREA FRACTION within
     tolerance of a fitted primitive is the reported coverage;
  4. the fitted planes and cylinders are assembled into a solid by half-space intersection plus
     cylinder subtraction (solid_fran_passning) -- a B-rep, not a mesh;
  5. that solid goes into brep_recept_synth_v1.synth, the recipe is validated by
     cad_op_schema_v1.validate_recipe and executed by cad_op_exec_v1.exec_ops.

API:
    falt_till_recept(V, T, pitch, namn) -> dict with the fit, the recipe, the rebuilt solid's volume
        and face mix, the volume deviation against the input solid and the rasterisation bound
    passa_primitiver(V, T, tol_mm) -> the fit, including tackning_area_frac (the coverage fraction)

A region that no plane and no cylinder fits within tolerance is reported as free-form area and NO
recipe is claimed for it: on a topology-optimised field the shipped number is the coverage fraction,
not a recipe.

Determinism: no random numbers are drawn anywhere in this module and every ordering is taken from a
measured quantity with a stable tie-break, so --selftest writes byte-identical stdout on repeated
runs (checked by the pytest wrapper, which runs the script twice and compares).

Run with --selftest: gate (a) rebuilds the synthetic bracket and the holed plate from their own
fields and compares volume and face mix, gate (b) reports the primitive coverage of a
topology-optimised field, and the script exits non-zero if gate (a) fails on either part.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "recipe"), os.path.join(HERE, "..", "..", "examples", "parts")):
    _a = os.path.abspath(_p)
    if _a not in sys.path:
        sys.path.insert(0, _a)

import faltkarna_v1_mesh_to_sdf as M2S  # noqa: E402

# faltkarna_v1_svep_v2_loft is imported inside _bund_identitet: importing it starts the field
# backend, whose banner would otherwise land on stdout before the selftest can route it away.

NORMAL_TOL_DEG = 8.0        # region growing: angle between a triangle normal and the region normal
PLAN_TOL_PITCH = 0.75       # a triangle belongs to a plane if its offset is inside this * pitch
CYL_TOL_PITCH = 0.75        # the same band for a cylinder's radius
PLAN_MIN_AREA_FRAC = 0.01   # a planar region must carry this fraction of the surface to be a plane
CYL_MIN_AREA_FRAC = 1.0e-3  # a leftover component must carry this fraction to be offered a cylinder
CYL_INLIER_MIN_FRAC = 0.5   # and this fraction of the component must end up inside the band
TACKNING_NORMAL_TOL_DEG = 25.0  # normal agreement required before a triangle counts as covered
RUND_MM = 6                 # rounding of the reported fit, in mm decimals


# ------------------------------------------------------------------ 1. sparse field -> dense window
def dense_fran_blocksparse(sf: dict, shape_l, block: int) -> np.ndarray:
    """Scatters a block-sparse SDF back into its dense window.

    Inactive blocks carry no samples: kind < 0 is entirely inside and kind > 0 entirely outside, and
    both are filled with the classifier's own Lipschitz margin (sf["margin_mm"]) with that sign, so
    the array's zero level set is exactly the level set the sparse field carries.
    """
    nx, ny, nz = shape_l
    n_bx, n_by, n_bz = sf["n_bx"], sf["n_by"], sf["n_bz"]
    px, py, pz = n_bx * block, n_by * block, n_bz * block
    kind = sf["kind"].reshape(n_bx, n_by, n_bz)
    margin = float(sf["margin_mm"])
    padded = np.empty((px, py, pz), dtype=np.float32)
    vy = padded.reshape(n_bx, block, n_by, block, n_bz, block).transpose(0, 2, 4, 1, 3, 5)
    vy[:] = np.where(kind < 0, -margin, margin).astype(np.float32)[:, :, :, None, None, None]
    if len(sf["active_ids"]) > 0:
        bi, bj, bk = np.unravel_index(sf["active_ids"], (n_bx, n_by, n_bz))
        vy[bi, bj, bk] = sf["tiles"]
    return padded[:nx, :ny, :nz]


def blocksparse_av_mesh(V, T, pitch: float, marginal_mm: float = 0.0, block: int = M2S.BLOCK):
    """Mesh -> block-sparse SDF through the existing rasteriser. Returns (sf, shape_l, gmin, pitch)."""
    wp, _have_cuda = M2S._import_warp()
    device = "cpu"   # the reverse pass runs on the CPU backend; the rasteriser supports both
    lo = np.asarray(V, dtype=np.float64).min(axis=0) - 4.0 * pitch
    r = M2S.mesh_to_sdf_del(wp, V, T, pitch, marginal_mm, lo, device, block=block,
                            returnera_falt=True)
    return r["sf"], r["shape_l"], np.asarray(r["gmin"], dtype=np.float64), float(r["pitch_effektiv"])


# ------------------------------------------------------------------ 2. zero level set -> mesh
def nollnivamesh(dense: np.ndarray, pitch: float, gmin):
    """Marching cubes on the dense window at level 0. Voxel i is the point gmin + i*pitch
    (faltkarna_v1_mesh_to_sdf.VOXEL_PROVPUNKT = 0.0), so no half-voxel shift is applied."""
    from skimage import measure
    verts, faces, _n, _v = measure.marching_cubes(np.asarray(dense, dtype=np.float32), level=0.0,
                                                 spacing=(pitch,) * 3)
    return np.asarray(verts, dtype=np.float64) + np.asarray(gmin, dtype=np.float64), \
        np.asarray(faces, dtype=np.int64)


# ------------------------------------------------------------------ 3. mesh -> primitives
def _tri_geometri(V, T):
    """Per-triangle normal (unit, outward as the level set orients it), area and centroid."""
    P = V[T]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    area = 0.5 * a2
    n = n / np.maximum(a2, 1e-300)[:, None]
    return n, area, P.mean(axis=1)


def _grannar(T):
    """Triangle adjacency over shared edges, as a list of index arrays (deterministic order)."""
    m = len(T)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    tri = np.tile(np.arange(m, dtype=np.int64), 3)
    order = np.lexsort((tri, e[:, 1], e[:, 0]))
    e, tri = e[order], tri[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    a, b = tri[:-1][same], tri[1:][same]
    par = np.concatenate([np.stack([a, b], axis=1), np.stack([b, a], axis=1)], axis=0)
    par = par[np.lexsort((par[:, 1], par[:, 0]))]
    start = np.searchsorted(par[:, 0], np.arange(m + 1))
    return par[:, 1], start


def _regioner(V, T, normal, area, cent, tol_mm: float):
    """Region growing over the triangle adjacency: a triangle joins a region when its normal is
    within NORMAL_TOL_DEG of the region's running mean normal and its centroid within tol_mm of the
    region's running plane. Seeds are taken in order of decreasing area, so the segmentation is a
    function of the mesh alone and repeats bit-identically."""
    nb, start = _grannar(T)
    cos_tol = math.cos(math.radians(NORMAL_TOL_DEG))
    label = np.full(len(T), -1, dtype=np.int64)
    ordning = np.argsort(-area, kind="stable")
    regioner = []
    for seed in ordning:
        if label[seed] >= 0:
            continue
        rid = len(regioner)
        label[seed] = rid
        medlem = [int(seed)]
        n_sum = normal[seed] * area[seed]
        w_sum = float(area[seed])
        d_sum = float(np.dot(normal[seed], cent[seed]) * area[seed])
        stack = [int(seed)]
        while stack:
            t = stack.pop()
            for k in range(start[t], start[t + 1]):
                g = int(nb[k])
                if label[g] >= 0:
                    continue
                nm = n_sum / max(np.linalg.norm(n_sum), 1e-300)
                if float(np.dot(normal[g], nm)) < cos_tol:
                    continue
                if abs(float(np.dot(nm, cent[g])) - d_sum / w_sum) > tol_mm:
                    continue
                label[g] = rid
                medlem.append(g)
                n_sum = n_sum + normal[g] * area[g]
                w_sum += float(area[g])
                d_sum += float(np.dot(nm, cent[g]) * area[g])
                stack.append(g)
        regioner.append(np.asarray(medlem, dtype=np.int64))
    return label, regioner


def _plan_passning(normal, area, cent, idx):
    """Area-weighted plane fit over a triangle set. Returns (normal, point, max residual)."""
    w = area[idx]
    n = (normal[idx] * w[:, None]).sum(axis=0)
    n = n / max(np.linalg.norm(n), 1e-300)
    p = (cent[idx] * w[:, None]).sum(axis=0) / w.sum()
    res = np.abs((cent[idx] - p) @ n)
    return n, p, float(res.max())


def _cyl_passning(normal, area, cent, idx):
    """Cylinder fit over a triangle set: the axis is the least-variance direction of the normals
    (a cylinder's normals are all perpendicular to its axis), the centre and radius come from an
    algebraic circle fit of the centroids projected across that axis. Returns
    (axis, point, radius, max radial residual, +1 outward / -1 inward)."""
    w = area[idx]
    N = normal[idx]
    C = (N * w[:, None]).T @ N
    ev, evec = np.linalg.eigh(C)
    ax = evec[:, 0]
    if ax[np.argmax(np.abs(ax))] < 0:
        ax = -ax
    e1 = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = e1 - ax * np.dot(e1, ax)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    P = cent[idx]
    u, v = P @ e1, P @ e2
    A = np.stack([u, v, np.ones_like(u)], axis=1) * w[:, None]
    b = (u ** 2 + v ** 2) * w
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    uc, vc = 0.5 * sol[0], 0.5 * sol[1]
    r2 = sol[2] + uc ** 2 + vc ** 2
    if r2 <= 0.0:
        return ax, P.mean(axis=0), 0.0, float("inf"), 0
    r = math.sqrt(r2)
    d = np.sqrt((u - uc) ** 2 + (v - vc) ** 2)
    res = float(np.abs(d - r).max())
    centrum = e1 * uc + e2 * vc
    centrum = centrum + ax * float(np.dot(P.mean(axis=0), ax))
    radial = P - centrum
    radial = radial - np.outer(radial @ ax, ax)
    radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-300)[:, None]
    sida = 1 if float((np.sum(radial * N, axis=1) * w).sum()) > 0.0 else -1
    return ax, centrum, r, res, sida


def _komponenter(T, label, rest_ids: set):
    """Connected components (over shared edges) of the triangles whose region was not accepted as a
    plane. A staircase-faceted cylinder wall comes out of the planar growing as many small facets;
    they are adjacent, so the component is the whole wall and the cylinder is fitted to it."""
    nb, start = _grannar(T)
    ingar = np.array([lab in rest_ids for lab in label])
    sedd = np.zeros(len(T), dtype=bool)
    comps = []
    for t0 in range(len(T)):
        if sedd[t0] or not ingar[t0]:
            continue
        sedd[t0] = True
        stack, med = [t0], [t0]
        while stack:
            t = stack.pop()
            for k in range(start[t], start[t + 1]):
                g = int(nb[k])
                if ingar[g] and not sedd[g]:
                    sedd[g] = True
                    med.append(g)
                    stack.append(g)
        comps.append(np.asarray(sorted(med), dtype=np.int64))
    return comps


def _cyl_passning_trimmad(normal, area, cent, idx, tol_mm, n_varv=3):
    """Cylinder fit with the out-of-tolerance triangles trimmed away, up to n_varv times.

    A bore's wall in a rasterised field carries a rounded rim where it meets the end faces: those
    triangles are not on the cylinder and, left in, they push the fit's residual over the band. They
    are dropped from the fit and, being outside the band, they are NOT counted as covered either.
    Returns (axis, point, radius, residual over the inliers, side, inlier triangle indices).
    """
    inl = idx
    ax = c = None
    r = res = 0.0
    sida = 0
    for _ in range(n_varv):
        if len(inl) < 8:
            return None
        ax, c, r, res, sida = _cyl_passning(normal, area, cent, inl)
        if not np.isfinite(res) or r <= 0.0:
            return None
        d = _cyl_avstand(cent[inl], ax, c, r)
        ny = inl[d <= tol_mm]
        if len(ny) == len(inl):
            break
        inl = ny
    if ax is None or len(inl) < 8:
        return None
    ax, c, r, res, sida = _cyl_passning(normal, area, cent, inl)
    return ax, c, r, res, sida, inl


def _cyl_avstand(P, ax, c, r):
    """|distance from each point to the cylinder's surface|."""
    d = P - c
    d = d - np.outer(d @ ax, ax)
    return np.abs(np.linalg.norm(d, axis=1) - r)


def _slaihop_plan(planer, tol_mm):
    """Merges plane fits that are the same plane (parallel within NORMAL_TOL_DEG and within tol_mm
    of each other), so a face split by the segmentation is one half-space, not two."""
    cos_tol = math.cos(math.radians(NORMAL_TOL_DEG))
    ut = []
    for pl in planer:
        n, p, a = np.asarray(pl["normal"]), np.asarray(pl["punkt"]), pl["area_mm2"]
        for q in ut:
            qn, qp = np.asarray(q["normal"]), np.asarray(q["punkt"])
            if float(n @ qn) >= cos_tol and abs(float(qn @ (p - qp))) <= tol_mm:
                w = q["area_mm2"] + a
                q["normal"] = list((qn * q["area_mm2"] + n * a) / w /
                                   np.linalg.norm((qn * q["area_mm2"] + n * a) / w))
                q["punkt"] = list((qp * q["area_mm2"] + p * a) / w)
                q["area_mm2"] = w
                q["residual_mm"] = max(q["residual_mm"], pl["residual_mm"])
                q["n_trianglar"] += pl["n_trianglar"]
                break
        else:
            ut.append(dict(pl))
    for q in ut:
        q["normal"] = [float(x) for x in q["normal"]]
        q["punkt"] = [float(x) for x in q["punkt"]]
    return ut


def _tackning(normal, area, cent, planer, cylindrar, tol_mm):
    """Area fraction within tol_mm of a fitted primitive, counting a triangle only when its own
    normal agrees with the primitive's normal there (otherwise a plane would "cover" a parallel face
    on the other side of the body). Returns (covered area, per-triangle covered flag)."""
    cos_tol = math.cos(math.radians(TACKNING_NORMAL_TOL_DEG))
    tackt = np.zeros(len(area), dtype=bool)
    for pl in planer:
        n, p = np.asarray(pl["normal"]), np.asarray(pl["punkt"])
        nara = np.abs((cent - p) @ n) <= tol_mm
        riktad = (normal @ n) >= cos_tol
        tackt |= nara & riktad
    for cy in cylindrar:
        ax, c, r = np.asarray(cy["axel"]), np.asarray(cy["punkt"]), cy["radie_mm"]
        nara = _cyl_avstand(cent, ax, c, r) <= tol_mm
        rad = cent - c
        rad = rad - np.outer(rad @ ax, ax)
        rad /= np.maximum(np.linalg.norm(rad, axis=1), 1e-300)[:, None]
        riktad = (np.sum(normal * rad, axis=1) * cy["sida"]) >= cos_tol
        tackt |= nara & riktad
    return float(area[tackt].sum()), tackt


def passa_primitiver(V, T, tol_mm: float) -> dict:
    """Fits the schema's primitive faces (planes, cylinders) to the mesh and measures the coverage.

    Planar regions that carry at least PLAN_MIN_AREA_FRAC of the surface become planes; everything
    else is grouped into connected components and offered a trimmed cylinder fit. The reported
    tackning_area_frac is the fraction of the total mesh area that lies within tol_mm of one of the
    accepted primitives AND whose own normal agrees with that primitive's there. The rest is
    free-form area and is NOT given a recipe.
    """
    normal, area, cent = _tri_geometri(V, T)
    tot = float(area.sum())
    label, regioner = _regioner(V, T, normal, area, cent, tol_mm)
    planer, rest = [], set()
    for rid, idx in enumerate(regioner):
        a = float(area[idx].sum())
        if a >= PLAN_MIN_AREA_FRAC * tot and len(idx) >= 4:
            n, p, res = _plan_passning(normal, area, cent, idx)
            if res <= tol_mm:
                planer.append({"normal": [float(x) for x in n], "punkt": [float(x) for x in p],
                               "area_mm2": a, "residual_mm": res, "n_trianglar": int(len(idx))})
                continue
        rest.add(rid)
    planer = _slaihop_plan(planer, tol_mm)
    cylindrar = []
    for idx in _komponenter(T, label, rest):
        a_komp = float(area[idx].sum())
        if a_komp < CYL_MIN_AREA_FRAC * tot:
            continue
        pas = _cyl_passning_trimmad(normal, area, cent, idx, tol_mm)
        if pas is None:
            continue
        ax, c, r, res, sida, inl = pas
        a_inl = float(area[inl].sum())
        if r <= tol_mm or res > tol_mm or a_inl < CYL_INLIER_MIN_FRAC * a_komp:
            continue
        cylindrar.append({"axel": [float(x) for x in ax], "punkt": [float(x) for x in c],
                          "radie_mm": float(r), "area_mm2": a_inl, "residual_mm": float(res),
                          "sida": int(sida), "n_trianglar": int(len(inl))})
    planer.sort(key=lambda d: (-round(d["area_mm2"], 9), [round(x, 9) for x in d["normal"]]))
    cylindrar.sort(key=lambda d: (-round(d["area_mm2"], 9), [round(x, 9) for x in d["punkt"]]))
    tackt, _flag = _tackning(normal, area, cent, planer, cylindrar, tol_mm)
    return {"planer": planer, "cylindrar": cylindrar, "area_total_mm2": tot,
            "area_primitiv_mm2": tackt, "fri_form_area_mm2": float(tot - tackt),
            "tackning_area_frac": float(tackt / max(tot, 1e-300)),
            "n_regioner": len(regioner), "tol_mm": float(tol_mm)}


# ------------------------------------------------------------------ 4. primitives -> B-rep
def solid_fran_passning(fit: dict, bbox_lo, bbox_hi):
    """Assembles the fitted primitives into a solid: the planes are taken as outward half-spaces and
    intersected inside a block covering the part, then every INWARD cylinder (its surface normals
    point at its own axis, so the material is outside it) is subtracted as a through bore.

    Only the face types the schema can rebuild are used. A fit whose planes do not bound a finite
    body raises ValueError rather than returning a half-open solid.
    """
    import build123d as bd

    lo = np.asarray(bbox_lo, dtype=float)
    hi = np.asarray(bbox_hi, dtype=float)
    mitt = 0.5 * (lo + hi)
    L = 3.0 * float(np.linalg.norm(hi - lo)) + 10.0
    sol = bd.Box(L, L, L).moved(bd.Location(bd.Vector(*mitt)))
    for pl in fit["planer"]:
        n = bd.Vector(*pl["normal"])
        p = bd.Vector(*pl["punkt"])
        kniv = bd.Plane(origin=p, z_dir=n) * bd.Box(
            L, L, L, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        sol = sol - kniv
    if not sol.solids():
        raise ValueError("the fitted planes do not bound a finite body")
    sol = bd.Part() + sol.solids()[0]
    for cy in fit["cylindrar"]:
        if cy["sida"] >= 0:
            continue  # an outward cylinder is part of the skin, not a bore
        ax = bd.Vector(*cy["axel"])
        c = np.asarray(cy["punkt"], dtype=float)
        a = np.asarray(cy["axel"], dtype=float)
        c = c + a * float(np.dot(mitt - c, a))
        borr = bd.Plane(origin=bd.Vector(*c), z_dir=ax) * bd.Cylinder(cy["radie_mm"], L)
        sol = bd.Part(sol.solids()) - borr
    return sol


# ------------------------------------------------------------------ 5. B-rep -> recipe (existing)
def _face_mix(shape) -> dict:
    from collections import Counter
    from brep_aag_v1 import build_aag
    return dict(Counter(f["type"] for f in build_aag(shape).faces))


def rasteringsbund(pitch: float, area_mm2: float, volym_mm3: float) -> float:
    """The rasterisation bound of this pitch for a body of this surface area and volume:
    0.5 * pitch * A / V, the bound faltkarna_v1_svep_v2_loft.rasterisation_tolerance states (a voxel
    occupancy resolves every face to at worst half a voxel). The identity with that function is
    asserted in the selftest on the shell it is written for, so the two cannot drift apart."""
    return 0.5 * pitch * area_mm2 / max(volym_mm3, 1e-300)


def falt_till_recept(V, T, pitch: float, namn: str, original=None, tol_mm=None) -> dict:
    """Full reverse pass for one part: mesh -> block-sparse SDF -> zero level set -> primitive fit
    -> B-rep -> the existing recogniser/synthesiser -> executed solid, with the comparison numbers.
    """
    import build123d as bd
    import brep_recept_synth_v1 as SYN
    import cad_op_exec_v1 as EX
    from cad_op_schema_v1 import validate_recipe

    sf, shape_l, gmin, pitch_eff = blocksparse_av_mesh(V, T, pitch)
    dense = dense_fran_blocksparse(sf, shape_l, M2S.BLOCK)
    MV, MT = nollnivamesh(dense, pitch_eff, gmin)
    tol = float(tol_mm if tol_mm is not None else PLAN_TOL_PITCH * pitch_eff)
    fit = passa_primitiver(MV, MT, tol)
    sol = solid_fran_passning(fit, MV.min(axis=0), MV.max(axis=0))
    recept = SYN.synth(sol)
    validate_recipe({"ops": recept["ops"]})
    res = EX.exec_ops(recept["ops"], params=recept["params"])
    byggd = res["solids"][recept["result_ref"]]

    ut = {"del": namn, "pitch_mm": float(pitch_eff),
          "n_blocks": int(sf["n_blocks"]), "n_active_blocks": int(sf["n_active"]),
          "n_trianglar_nivamesh": int(len(MT)),
          "passning": {"n_plan": len(fit["planer"]), "n_cylinder": len(fit["cylindrar"]),
                       "tackning_area_frac": round(fit["tackning_area_frac"], RUND_MM),
                       "radier_mm": sorted(round(c["radie_mm"], 3) for c in fit["cylindrar"]),
                       "max_residual_mm": round(max([0.0] + [d["residual_mm"] for d in
                                                             fit["planer"] + fit["cylindrar"]]), RUND_MM),
                       "tol_mm": round(tol, RUND_MM)},
          "ops": [o["op"] for o in recept["ops"]],
          "rebuilt": {"volume_mm3": float(byggd.volume), "face_mix": _face_mix(byggd)}}
    if original is not None:
        A = float(sum(f.area for f in original.faces()))
        Vol = float(original.volume)
        bund = rasteringsbund(pitch_eff, A, Vol)
        dev = abs(ut["rebuilt"]["volume_mm3"] - Vol) / Vol
        ut["orig"] = {"volume_mm3": Vol, "area_mm2": A, "face_mix": _face_mix(original)}
        ut["vol_dev_frac"] = dev
        ut["vol_dev_pct"] = round(100.0 * dev, RUND_MM)
        ut["rasterisation_bound_frac"] = bund
        ut["face_mix_match"] = (ut["orig"]["face_mix"] == ut["rebuilt"]["face_mix"])
        ut["PASS"] = bool(dev < bund and ut["face_mix_match"])
    _ = bd
    return ut


# ------------------------------------------------------------------ fixtures and gates
def _mesh_av_solid(part, tol=0.02, vinkel=0.05):
    import tempfile
    import trimesh
    from build123d import export_stl
    p = os.path.join(tempfile.gettempdir(), "field_to_recipe_v1_fixture.stl")
    export_stl(part, p, tolerance=tol, angular_tolerance=vinkel)
    m = trimesh.load(p, process=False)
    m.merge_vertices()
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def _bracket():
    """The synthetic bracket the synthesiser's own selftest uses: a rectangular plate with four
    through holes, whose recipe is therefore known independently of this module."""
    import brep_recept_synth_v1 as SYN
    return SYN._synthetic_bracket()


def _holed_plate():
    """The holed plate fixture: one plate with five through holes of radii 1..16 mm."""
    import holed_plate_v1 as HP
    return HP.build()


def _topopt_falt():
    """The density field of the topology optimiser's own selftest case, as a signed field in mm.

    lastfalt_v1_topopt.run() writes rho on its voxel grid; rho - 0.5 is the field whose zero level
    set is the optimised boundary, and the grid pitch comes from the run's own dx/dy/dz.
    """
    import lastfalt_v1_topopt as TO
    out = TO.run(n_iter=TOPOPT_ITER, tag="field_to_recipe_v1")
    rho = np.load(os.path.join(TO.OUT_DIR, "rho_field_field_to_recipe_v1.npy"))
    dx, dy, dz = out["dx_dy_dz_mm"]
    bb = TO.synthetic_plate_and_loads()[0]["bbox_mm"]
    return rho, (dx, dy, dz), (bb["xmin"], bb["ymin"], bb["zmin"]), out


TOPOPT_ITER = 12


def _topopt_tackning() -> dict:
    """Gate (b): how much of a topology-optimised boundary the schema's primitives actually cover.

    The field is resampled to an isotropic pitch (the optimiser's grid is anisotropic), marching
    cubes takes the zero level set of rho - 0.5, and the SAME fit as on the two manufacturable parts
    runs on it. The number shipped is the covered area fraction; the rest is free-form and no recipe
    is claimed for it.
    """
    from scipy import ndimage
    from skimage import measure
    rho, (dx, dy, dz), lo, out = _topopt_falt()
    pitch = float(min(dx, dy, dz))
    zoom = (dx / pitch, dy / pitch, dz / pitch)
    f = ndimage.zoom(np.asarray(rho, dtype=np.float64) - 0.5, zoom, order=1, mode="nearest")
    f = np.pad(f, 1, mode="constant", constant_values=-0.5)
    verts, faces, _n, _v = measure.marching_cubes(f.astype(np.float32), level=0.0,
                                                 spacing=(pitch,) * 3)
    verts = np.asarray(verts, dtype=np.float64) + np.asarray(lo, dtype=np.float64) - pitch
    fit = passa_primitiver(verts, np.asarray(faces, dtype=np.int64), PLAN_TOL_PITCH * pitch)
    return {"fall": "lastfalt_v1_topopt selftest case", "n_iter": TOPOPT_ITER,
            "grid": out["grid"], "final_volfrac": round(out["final_volfrac"], 6),
            "compliance_final_Nmm": round(out["compliance_final_Nmm"], 6),
            "pitch_mm": pitch, "tol_mm": round(PLAN_TOL_PITCH * pitch, RUND_MM),
            "n_trianglar": int(len(faces)),
            "area_total_mm2": round(fit["area_total_mm2"], 3),
            "area_primitiv_mm2": round(fit["area_primitiv_mm2"], 3),
            "fri_form_area_mm2": round(fit["fri_form_area_mm2"], 3),
            "n_plan": len(fit["planer"]), "n_cylinder": len(fit["cylindrar"]),
            "tackning_area_frac": round(fit["tackning_area_frac"], RUND_MM),
            "recept_for_fri_form": False}


def _bund_identitet() -> dict:
    """rasteringsbund must be the same bound faltkarna_v1_svep_v2_loft.rasterisation_tolerance
    states: checked on the shell that function is written for, so the two cannot drift apart."""
    import faltkarna_v1_svep_v2_loft as SVEP
    pitch, r_in, r_out, L = 0.5, 13.0, 15.0, 300.0
    ref = SVEP.rasterisation_tolerance(pitch, r_in, r_out, L)
    egen = rasteringsbund(pitch, ref["A_mantelyta_mm2"], ref["V_mm3"])
    return {"tol_frac_svep": ref["tol_frac"], "tol_frac_egen": egen,
            "IDENTISK": bool(abs(ref["tol_frac"] - egen) <= 1e-12 * max(ref["tol_frac"], 1e-12))}


@contextlib.contextmanager
def _stdout_till_stderr():
    """Sends everything the stages print, at Python level and at file-descriptor level (the
    rasteriser's backend writes its banner on fd 1 directly), to stderr."""
    sys.stdout.flush()
    dup = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(dup, 1)
        os.close(dup)


BRACKET_PITCH_MM = 1.0
PLATE_PITCH_MM = 0.5


def _selftest() -> dict:
    ident = _bund_identitet()
    delar = {}
    for namn, part, pitch in (("synthetic_bracket", _bracket(), BRACKET_PITCH_MM),
                              ("holed_plate", _holed_plate(), PLATE_PITCH_MM)):
        sol = part.solids()[0] if len(part.solids()) > 1 else part
        V, T = _mesh_av_solid(sol)
        delar[namn] = falt_till_recept(V, T, pitch, namn, original=sol)
    top = _topopt_tackning()
    ok_a = all(d["PASS"] for d in delar.values())
    return {"bound_identity": ident, "gate_a_parts": delar, "gate_b_topopt": top,
            "GATE_A_PASS": bool(ok_a and ident["IDENTISK"]),
            "PASS": bool(ok_a and ident["IDENTISK"])}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        # Everything the stages print on the way (the rasteriser's device banner, the optimiser's own
        # report with its wall clock) goes to stderr, so stdout carries the measured JSON alone and
        # two runs can be compared byte for byte.
        with _stdout_till_stderr():
            r = _selftest()
        print(json.dumps(r, indent=1, sort_keys=True))
        sys.exit(0 if r["PASS"] else 1)
    print(__doc__)
