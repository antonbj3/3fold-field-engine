#!/usr/bin/env python3
"""Diagnostic cell: locate the enclosed voids in a duct built by the Lipschitz-spine primitive.

It rebuilds the EXACT duct tree the verification cell measures (same synthetic room, same
WALL/SAFETY, same ambitious radius), meshes it at the SAME pitch the verification cell's convergence
check already used, then uses genus_v1.mesh_topologi's own connected-component labelling to isolate
the boundary components that are NOT the main duct shell -- i.e. the "n_hallrum" islands -- and
reports their centroid, bbox, volume and vertex count. For each such island it also samples the duct
SDF and its inner/outer sub-trees along lines through the centroid, to identify which term closes it.

Its gate: the independent labelling here must return the same number of boundary components as
genus_v1.mesh_topologi does on the same mesh. When there are no voids the cavity list is empty and
that is the reported result, not a failure.

Run: python lipschitz_spine_v1_locate_cavities.py
Output: artifacts/ikarus_lipschitz_v1_cavities.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import expr as E                      # noqa: E402
import eval_warp as W                 # noqa: E402
import genus_v1 as G                  # noqa: E402
import lipschitz_spine_v1             # noqa: E402
import synthetic_room_v1              # noqa: E402
from lipschitz_spine_v1_verify import (ART, AST, R_AMBITIOUS, SAFETY, WALL,  # noqa: E402
                                       mesh_via_ikarus)

DEVICE = os.environ.get("FIELD_ENGINE_DEVICE") or W.DEVICE
PITCH_MC = 0.8    # the coarser of the verification cell's two convergence resolutions
T0 = time.time()


def log(m):
    print(f"[locate] {time.time()-T0:7.1f}s {m}", flush=True)


def label_components(P, T):
    """The same labelling scheme as genus_v1.mesh_topologi (welded vertices, triangle adjacency via
    shared edges), but it also returns the per-triangle label array, so non-main components can be
    pulled out."""
    from genus_v1 import SVETS_DEC
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=np.int64)
    nyckel = np.round(P, SVETS_DEC)
    uniq, inv = np.unique(nyckel, axis=0, return_inverse=True)
    Tw = inv.reshape(-1)[T]
    ok = (Tw[:, 0] != Tw[:, 1]) & (Tw[:, 1] != Tw[:, 2]) & (Tw[:, 0] != Tw[:, 2])
    Tw = Tw[ok]
    anv = np.unique(Tw)
    omap = np.full(uniq.shape[0], -1, dtype=np.int64)
    omap[anv] = np.arange(anv.size)
    Pw = uniq[anv]
    Tw = omap[Tw]
    nF = Tw.shape[0]
    Edges = np.sort(np.concatenate([Tw[:, [0, 1]], Tw[:, [1, 2]], Tw[:, [2, 0]]], axis=0), axis=1)
    _, ei = np.unique(Edges, axis=0, return_inverse=True)
    tri_id = np.tile(np.arange(nF), 3)
    M = sp.coo_matrix((np.ones(ei.size), (ei.reshape(-1), tri_id)),
                      shape=(int(ei.max()) + 1, nF)).tocsr()
    ncomp, lab = csg.connected_components((M.T @ M) > 0, directed=False)
    return Pw, Tw, lab, ncomp


def main():
    if not os.path.exists(os.path.join(AST, "spine_v1.npz")):
        synthetic_room_v1.write(AST)
    A = np.load(os.path.join(AST, "spine_v1.npz"), allow_pickle=False)
    C, R_ref, T_tan = A["C"], A["R"], A["T"]
    Fld = np.load(os.path.join(AST, "falt_rum.npz"), allow_pickle=False)
    d_free, origin, pitch_grid = Fld["d_free"], Fld["origin"], float(Fld["pitch"])
    n = len(C)

    r_ambitious = np.full(n, R_AMBITIOUS)
    duct_tree, outer_tree, inner_tree, audit = lipschitz_spine_v1.hollow_duct_from_spine(
        C, r_ambitious, d_free, origin, pitch_grid, wall_mm=WALL, safety_mm=SAFETY,
        tangents_mm=T_tan, r_min_mm=1.0, mouth_margin_mm=5.0)
    r_used = np.array(audit["r_used_mm"])
    log(f"rebuilt duct_tree, node_count={E.node_count(duct_tree)}")

    device = DEVICE
    _ms = audit["mouth_stub"]
    stub_reach = _ms["spacing_mm"] * max(_ms["n_ext_mouth0"], _ms["n_ext_mouth1"]) + 3.0
    bbox_lo = C.min(0) - (r_used.max() + WALL + stub_reach)
    bbox_hi = C.max(0) + (r_used.max() + WALL + stub_reach)

    verts, faces = mesh_via_ikarus(duct_tree, bbox_lo, bbox_hi, PITCH_MC, device)
    log(f"meshed pitch={PITCH_MC}: {len(verts)}v {len(faces)}f")

    topo = G.mesh_topologi(verts, faces)
    topo.pop("_svetsad", None)
    log(f"handles_summa={topo['handles_summa']} n_hallrum={topo['n_hallrum']} "
        f"n_randkomponenter={topo['n_randkomponenter']}")

    Pw, Tw, lab, ncomp = label_components(verts, faces)
    labelling_agrees = bool(ncomp == topo["n_randkomponenter"])

    comps = []
    for c in range(ncomp):
        idx_tris = np.flatnonzero(lab == c)
        vids = np.unique(Tw[idx_tris])
        pts = Pw[vids]
        centroid = pts.mean(axis=0)
        bbox_lo_c = pts.min(axis=0)
        bbox_hi_c = pts.max(axis=0)
        # signed volume of this component's own closed surface (each component is itself closed,
        # both for the main duct and for an isolated cavity island)
        Tc = Tw[idx_tris]
        a, b, c3 = Pw[Tc[:, 0]], Pw[Tc[:, 1]], Pw[Tc[:, 2]]
        vol = float(abs(np.einsum("ij,ij->i", a, np.cross(b, c3)).sum()) / 6.0)
        comps.append({"label": int(c), "n_verts": int(vids.size), "n_tris": int(idx_tris.size),
                      "centroid_mm": [round(float(v), 3) for v in centroid],
                      "bbox_lo_mm": [round(float(v), 3) for v in bbox_lo_c],
                      "bbox_hi_mm": [round(float(v), 3) for v in bbox_hi_c],
                      "bbox_size_mm": [round(float(v), 4) for v in (bbox_hi_c - bbox_lo_c)],
                      "volume_mm3": round(vol, 6)})
    comps.sort(key=lambda k: -k["n_verts"])
    main_comp = comps[0]
    cavities = comps[1:]
    log(f"main component: {main_comp['n_verts']}v vol={main_comp['volume_mm3']:.1f}mm3")
    for cav in cavities:
        log(f"CAVITY label={cav['label']} n_verts={cav['n_verts']} vol={cav['volume_mm3']:.6f}mm3 "
            f"centroid={cav['centroid_mm']} bbox_size={cav['bbox_size_mm']}")

    # ---------------------------------------------------------------- SDF sampling along lines
    # through each cavity centroid, to identify WHICH term closes it: the duct tree (outer minus
    # inner), the outer tree alone, and the EXTENDED inner used inside hollow_duct_from_spine
    # (rebuilt here, since hollow_duct_from_spine does not return it).
    inf_field = np.full(d_free.shape, 1e6, dtype=np.float32)

    def _mouth_stub(c_end, t_end, r_end, sign, spacing):
        ext_total = (r_end + WALL) + 5.0
        n_ext = max(int(np.ceil(ext_total / spacing)), 1)
        dists = spacing * np.arange(1, n_ext + 1)
        pts = c_end[None, :] + sign * dists[:, None] * t_end[None, :]
        taper_from = int(np.ceil(n_ext * 2 / 3))
        radii = np.full(n_ext, r_end)
        if taper_from < n_ext:
            tail = np.arange(taper_from, n_ext)
            frac = (tail - taper_from + 1) / max(n_ext - taper_from, 1)
            radii[taper_from:] = np.maximum(r_end * (1.0 - 0.85 * frac), 1.0)
        return (pts[::-1] if sign < 0 else pts), (radii[::-1] if sign < 0 else radii)

    spacing = _ms["spacing_mm"]
    stub0, r0 = _mouth_stub(C[0], T_tan[0], r_used[0], -1.0, spacing)
    stub1, r1 = _mouth_stub(C[-1], T_tan[-1], r_used[-1], +1.0, spacing)
    C_ext = np.vstack([stub0, C, stub1])
    r_ext = np.concatenate([r0, r_used, r1])
    diag = {"cavities": []}
    if cavities:
        inner_ext_tree, _ = lipschitz_spine_v1.lipschitz_spine_tube(
            C_ext, r_ext, inf_field, origin, pitch_grid, wall_mm=0.0, safety_mm=0.0, r_min_mm=0.0)

        def sample_along(tree, p0, axis, half_len, n=41):
            ts = np.linspace(-half_len, half_len, n)
            pts = p0[None, :] + ts[:, None] * axis[None, :]
            vals = W.eval_batch(tree, pts, device=device)
            return ts, vals

        for cav in cavities:
            p0 = np.array(cav["centroid_mm"])
            # nearest spine station or stub point, to orient the sample axis along the local tangent
            d_to_all = np.linalg.norm(C_ext - p0[None, :], axis=1)
            nearest_i = int(np.argmin(d_to_all))
            if nearest_i < len(stub0):
                local_axis = T_tan[0]
                nearest_label = f"stub0[{nearest_i}]"
            elif nearest_i >= len(stub0) + len(C):
                local_axis = T_tan[-1]
                nearest_label = f"stub1[{nearest_i - len(stub0) - len(C)}]"
            else:
                si = nearest_i - len(stub0)
                local_axis = T_tan[min(si, len(T_tan) - 1)]
                nearest_label = f"spine[{si}]"
            local_axis = local_axis / (np.linalg.norm(local_axis) + 1e-12)
            perp = np.cross(local_axis, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(local_axis, np.array([0.0, 1.0, 0.0]))
            perp = perp / (np.linalg.norm(perp) + 1e-12)

            entry = {"centroid_mm": cav["centroid_mm"], "volume_mm3": cav["volume_mm3"],
                     "nearest_reference": nearest_label,
                     "nearest_dist_mm": round(float(d_to_all[nearest_i]), 3)}
            for axname, ax in (("tangent", local_axis), ("perp", perp)):
                ts, v_duct = sample_along(duct_tree, p0, ax, 6.0)
                _, v_outer = sample_along(outer_tree, p0, ax, 6.0)
                _, v_inner_ext = sample_along(inner_ext_tree, p0, ax, 6.0)
                entry[axname] = {
                    "t_mm": [round(float(t), 3) for t in ts],
                    "duct_sdf": [round(float(v), 4) for v in v_duct],
                    "outer_sdf": [round(float(v), 4) for v in v_outer],
                    "inner_ext_sdf": [round(float(v), 4) for v in v_inner_ext],
                }
            diag["cavities"].append(entry)

    out = {"cell": "ikarus_lipschitz_v1_locate_cavities", "pitch_mm": PITCH_MC,
           "n_randkomponenter": topo["n_randkomponenter"], "n_hallrum": topo["n_hallrum"],
           "handles_summa": topo["handles_summa"],
           "labelling_agrees_with_genus_v1": labelling_agrees,
           "n_cavities": len(cavities),
           "main_component": main_comp, "cavities": cavities, "sdf_diagnostic": diag}
    os.makedirs(ART, exist_ok=True)
    outp = os.path.join(ART, "ikarus_lipschitz_v1_cavities.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"WROTE {outp} labelling_agrees={labelling_agrees} n_cavities={len(cavities)}")
    return 0 if labelling_agrees else 1


if __name__ == "__main__":
    sys.exit(main())
