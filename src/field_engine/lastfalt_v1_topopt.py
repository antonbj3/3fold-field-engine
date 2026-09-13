#!/usr/bin/env python3
"""SIMP + OC 3D topology optimisation on a voxel grid of a bolted plate under a synthetic load case.

The geometry (a square plate with four bolt holes and a loaded footprint) and the load case (a vertical
force and a tilting moment) are generated in code by synthetic_plate_and_loads(). The element core is
cross-verified by lastfalt_v1_verify.py. Material flows where the load wants it: the nodes at the four
bolt holes and the loaded footprint are fixed, everything else in the plate volume is free.

Two falsification runs:
  - mirror the moment (+X against -X): the density field must change measurably;
  - zero the load: the degenerate-load gate must fail before the optimiser can return a tidy answer.

Run `python lastfalt_v1_topopt.py [--volfrac 0.35] [--mirror] [--zero-load] [--tag NAME]`; writes
artifacts/topopt_TAG.json and artifacts/rho_field_TAG.npy.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lastfalt_v1_fem import hex8_ke, build_grid, assemble_K_cpu, solve, compliance, node_id  # noqa: E402

ROOT = HERE
OUT_DIR = os.path.join(HERE, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

E_AL_MPA = 71000.0     # N/mm^2, aluminium alloy
NU_AL = 0.33            # standard value for aluminium (declared, not measured)
RHO_AL_KG_MM3 = 2700.0e-9

NX, NY, NZ = 32, 32, 5
RHO_MIN = 1e-3
SIMP_P = 3.0
RMIN_CELLS = 1.6
N_ITER = 40
MOVE = 0.2


def synthetic_plate_and_loads():
    """Generates the synthetic input: a square bolted plate and its load case.

    Returns (geom, Fv_N, Mk_Nm), where geom holds the bbox in mm, the four bolt-hole centres and
    radius, the loaded footprint centre and its radius. All values are declared, not measured.
    """
    half, thick = 283.3, 24.0
    geom = {
        "bbox_mm": {"xmin": -half, "ymin": -half, "zmin": 0.0,
                    "xmax": half, "ymax": half, "zmax": thick},
        "bolt_holes": {"xy_centers_mm": [[-220.0, -220.0], [220.0, -220.0],
                                         [220.0, 220.0], [-220.0, 220.0]],
                       "r_mm": [11.0]},
        "load_footprint": {"xy_mm": [0.0, 0.0], "r_mm": 102.0},
    }
    return geom, 12000.0, 4500.0


def get_Fv_Mk():
    """Returns the synthetic load case (vertical force in N, tilting moment in Nm)."""
    _geom, fv, mk = synthetic_plate_and_loads()
    return fv, mk


def run(volfrac=0.35, mirror=False, zero_load=False, n_iter=N_ITER, tag="baseline",
        design_mask_xy=None):
    """Runs the SIMP + OC loop.

    Takes the target volume fraction, the mirror and zero-load falsification switches, the iteration
    count, an output tag, and an optional (NX,NY) boolean design mask whose False cells are not part of
    the plate's material and are frozen to RHO_MIN and excluded from the search. Returns the result
    dict and writes it plus the density field next to this file.
    """
    geom, _fv, _mk = synthetic_plate_and_loads()
    bbox = geom["bbox_mm"]
    xmin, ymin, zmin = bbox["xmin"], bbox["ymin"], bbox["zmin"]
    xmax, ymax, zmax = bbox["xmax"], bbox["ymax"], bbox["zmax"]
    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin

    bolts = geom["bolt_holes"]["xy_centers_mm"]
    bolt_r = geom["bolt_holes"]["r_mm"][0]
    foot_xy = geom["load_footprint"]["xy_mm"]
    R_load = geom["load_footprint"]["r_mm"]      # loaded footprint radius

    dx, dy, dz = Lx / NX, Ly / NY, Lz / NZ
    fix_r = max(bolt_r, 0.85 * max(dx, dy))  # the grid is coarser than the hole; catch at least one node

    Ke0 = hex8_ke(dx, dy, dz, E=1.0, nu=NU_AL)
    elem_nodes, (ei, ej, ek) = build_grid(NX, NY, NZ)
    n_elem = elem_nodes.shape[0]
    n_nodes = (NX + 1) * (NY + 1) * (NZ + 1)
    n_dof = 3 * n_nodes

    # node coordinates
    xs = xmin + dx * np.arange(NX + 1)
    ys = ymin + dy * np.arange(NY + 1)
    zs = zmin + dz * np.arange(NZ + 1)
    NI, NJ, NK = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1), np.arange(NZ + 1), indexing="ij")
    node_x = xs[NI].ravel(); node_y = ys[NJ].ravel(); node_z = zs[NK].ravel()
    ids_flat = node_id(NI, NJ, NK, NY, NZ).ravel()
    order = np.argsort(ids_flat)
    node_x, node_y, node_z = node_x[order], node_y[order], node_z[order]

    # --- fixed dofs: every node within fix_r of a bolt hole, all z layers ---
    fixed_mask = np.zeros(n_nodes, dtype=bool)
    for (bx, by) in bolts:
        d2 = (node_x - bx) ** 2 + (node_y - by) ** 2
        fixed_mask |= (d2 <= fix_r ** 2)
    fixed_dofs = np.concatenate([3 * np.where(fixed_mask)[0] + c for c in range(3)])

    # --- load nodes: top layer (z=zmax) within R_load of the footprint centre ---
    fx, fy = foot_xy
    top_mask = np.isclose(node_z, zmax)
    d2f = (node_x - fx) ** 2 + (node_y - fy) ** 2
    load_mask = top_mask & (d2f <= R_load ** 2)
    load_ids = np.where(load_mask)[0]
    if len(load_ids) == 0:
        raise RuntimeError("no load nodes: the grid is too coarse for R_load")

    Fv, Mk = get_Fv_Mk()
    if zero_load:
        Fv, Mk = 0.0, 0.0
    sign = -1.0 if mirror else 1.0

    f = np.zeros(n_dof)
    # uniform pressure distribution (-Z) for Fv
    for nid in load_ids:
        f[3 * nid + 2] += -Fv / len(load_ids)
    # moment Mk about the Y axis, distributed as f_i = Mk * dx_i / sum(dx_i^2) in Z (plane sections)
    dxs = sign * (node_x[load_ids] - fx)
    denom = np.sum(dxs ** 2)
    if denom > 1e-9 and Mk != 0.0:
        for idx, nid in enumerate(load_ids):
            f[3 * nid + 2] += Mk * 1000.0 * dxs[idx] / denom   # Mk in Nm -> Nmm (dxs in mm)

    total_load_mag = float(np.sum(np.abs(f)))

    # --- non-design (always rho=1): elements near a bolt hole and under the load patch ---
    ecx = xmin + dx * (ei + 0.5); ecy = ymin + dy * (ej + 0.5); ecz = zmin + dz * (ek + 0.5)
    nondesign = np.zeros(n_elem, dtype=bool)
    for (bx, by) in bolts:
        nondesign |= ((ecx - bx) ** 2 + (ecy - by) ** 2) <= fix_r ** 2
    top_elem = (ek == NZ - 1)
    nondesign |= (top_elem & (((ecx - fx) ** 2 + (ecy - fy) ** 2) <= R_load ** 2))

    void_mask = np.zeros(n_elem, dtype=bool)
    if design_mask_xy is not None:
        assert design_mask_xy.shape == (NX, NY), design_mask_xy.shape
        void_mask = ~design_mask_xy[ei, ej]
        void_mask &= ~nondesign          # bult/last-non-design vinner alltid over void-masken
        nondesign = nondesign | void_mask

    n_design = int(np.sum(~nondesign))

    rho = np.full(n_elem, volfrac)
    rho[nondesign] = 1.0
    rho[void_mask] = RHO_MIN             # void is not material: frozen at the floor value

    # cone-filter neighbour offsets (radius RMIN_CELLS in cell units, isotropic on (i,j,k))
    R = int(np.ceil(RMIN_CELLS))
    offs = [(a, b, c) for a in range(-R, R + 1) for b in range(-R, R + 1) for c in range(-R, R + 1)
            if (a * a + b * b + c * c) <= RMIN_CELLS ** 2]
    eidx = np.arange(n_elem).reshape(NX, NY, NZ)

    def filter_field(x3d):
        acc = np.zeros_like(x3d)
        wsum = np.zeros_like(x3d)
        for (a, b, c) in offs:
            w = max(0.0, RMIN_CELLS - np.sqrt(a * a + b * b + c * c))
            shifted = np.roll(np.roll(np.roll(x3d, -a, axis=0), -b, axis=1), -c, axis=2)
            # zero-pad outside the grid (no wrap)
            mask = np.ones_like(x3d, dtype=bool)
            if a > 0: mask[-a:, :, :] = False
            if a < 0: mask[:-a, :, :] = False
            if b > 0: mask[:, -b:, :] = False
            if b < 0: mask[:, :-b, :] = False
            if c > 0: mask[:, :, -c:] = False
            if c < 0: mask[:, :, :-c] = False
            acc += np.where(mask, w * shifted, 0.0)
            wsum += np.where(mask, w, 0.0)
        return acc / np.maximum(wsum, 1e-12)

    compliance_hist = []
    t0 = time.time()
    for it in range(n_iter):
        K = assemble_K_cpu(elem_nodes, rho, Ke0, p=SIMP_P, E0=E_AL_MPA, n_nodes=n_nodes)
        u = solve(K, f, fixed_dofs)
        c = compliance(f, u)
        compliance_hist.append(c)

        # per-element sensitivity dc/drho = -p rho^(p-1) E0 * u_e^T Ke0 u_e
        dof_e = (elem_nodes[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(n_elem, 24)
        ue = u[dof_e]
        uKu = np.einsum('ei,ij,ej->e', ue, Ke0, ue)
        dc = -SIMP_P * np.power(rho, SIMP_P - 1) * E_AL_MPA * uKu

        dc3d = dc.reshape(NX, NY, NZ)
        rho3d = rho.reshape(NX, NY, NZ)
        dc_f = filter_field(dc3d * rho3d) / np.maximum(rho3d, 1e-3)
        dc_f = dc_f.ravel()
        dc_f[nondesign] = 0.0

        # OC update with a volume bisection, over the design elements only
        design_idx = np.where(~nondesign)[0]
        l1, l2 = 1e-12, 1e12
        rho_new = rho.copy()
        target_vol = volfrac * n_elem  # includes non-design (already 1), so the bisection is on total volume
        for _ in range(90):  # 1e-12..1e12 needs about 80 halvings for full precision; 40 is not enough
            lmid = 0.5 * (l1 + l2)
            Bcoef = np.sqrt(np.maximum(-dc_f[design_idx] / lmid, 0.0))
            cand = rho[design_idx] * Bcoef
            lo = np.maximum(rho[design_idx] - MOVE, RHO_MIN)
            hi = np.minimum(rho[design_idx] + MOVE, 1.0)
            cand = np.clip(cand, lo, hi)
            rho_new[design_idx] = cand
            vol = np.sum(rho_new)
            if vol > target_vol:
                l1 = lmid
            else:
                l2 = lmid
        rho = rho_new

    wall_s = time.time() - t0
    final_volfrac = float(np.mean(rho))

    tip_disp_mm = float(np.max(np.abs(u[3 * np.where(load_mask)[0]])))  # displacement under the load patch
    max_uz = float(np.max(np.abs(u[2::3])))

    out = {
        "tag": tag, "volfrac_target": volfrac, "mirror": mirror, "zero_load": zero_load,
        "grid": [NX, NY, NZ], "n_elem": int(n_elem), "n_design_elem": n_design,
        "design_mask_anvand": design_mask_xy is not None, "n_void_elem": int(void_mask.sum()),
        "dx_dy_dz_mm": [dx, dy, dz],
        "fix_r_mm": fix_r, "R_load_mm": R_load,
        "Fv_N": Fv, "Mk_Nm": Mk, "total_load_mag_abs_sum_N": total_load_mag,
        "n_iter": n_iter, "wall_s": wall_s,
        "compliance_hist_Nmm": compliance_hist,
        "compliance_final_Nmm": compliance_hist[-1],
        "final_volfrac": final_volfrac,
        "max_abs_uz_mm": max_uz,
    }

    rho_path = os.path.join(OUT_DIR, f"rho_field_{tag}.npy")
    np.save(rho_path, rho.reshape(NX, NY, NZ))
    out["rho_field_npy"] = os.path.relpath(rho_path, ROOT)

    meta_path = os.path.join(OUT_DIR, f"topopt_{tag}.json")
    with open(meta_path, "w") as fjs:
        json.dump(out, fjs, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "compliance_hist_Nmm"}, indent=1))
    print("-> wrote", meta_path, "and", rho_path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--volfrac", type=float, default=0.35)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--zero-load", action="store_true")
    ap.add_argument("--n-iter", type=int, default=N_ITER)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--design-mask", default=None,
                     help="npy file with an (NX,NY) boolean design mask")
    args = ap.parse_args()
    mask = np.load(args.design_mask) if args.design_mask else None
    run(volfrac=args.volfrac, mirror=args.mirror, zero_load=args.zero_load, n_iter=args.n_iter, tag=args.tag,
        design_mask_xy=mask)
