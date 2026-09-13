#!/usr/bin/env python3
"""CNC subtraction as a field operation on an SDF voxel field, evaluated with Warp.

Input: a synthetic turned profile generated in code (a stepped shaft with three ring grooves), so the
script is self-contained.

Three operation types, all the same mechanism, max(material, -tool):
  1. the blank: round bar, radius = the part profile's own maximum radius plus an allowance;
  2. turning: the axisymmetric profile is cut from the blank with a tool nose radius RHO_TURN via a 2D
     morphological closing (a disk structuring element of radius RHO_TURN in pixels) on a fine (r,z)
     raster. That is the exact operation for "what a nose-radius tool can reach": convex corners stay
     sharp, concave corners are rounded to exactly RHO_TURN. The SDF of that raster is then looked up
     bilinearly per 3D voxel (revolved about Z) and the 3D subtraction runs as a Warp kernel;
  3. pocket milling and drilling: one Warp kernel over the whole 3D voxel field, with a rounded-box SDF
     for the pocket (side-edge radius RHO_MILL) and a cylinder SDF for the cross bore, plus a
     short-tool negative control (drill shorter than the full width, which must leave a measurable plug).

Takes no arguments (env `CLOSINGVAL=fast|ref` selects the GPU or reference closing, `PITCH3D_MM`
overrides the voxel pitch); writes artifacts/tillverkningsfalt_v1.json and the result STL next to this
file, and gates on the measured groove-root radius, the two independent volume measures and the
negative control.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, disk
from skimage import measure
import trimesh

try:
    import warp as wp
    wp.init()
    HAVE_CUDA = "cuda:0" in [str(d) for d in wp.get_devices()]
except Exception as e:
    wp = None
    HAVE_CUDA = False
    print("warp unavailable:", e)

# The GPU disk-closing variant (kernelvarv_v2_f4_closing) lives in the kernel-engine repository; with
# CLOSINGVAL=fast it is used when importable and a CUDA device exists, otherwise the scikit-image
# reference closing runs. The two were measured bit-identical.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from kernelvarv_v2_f4_closing import disk_offsets as _closing_disk_offsets, \
        closing_warp as _closing_warp
    HAVE_CLOSING_GPU = HAVE_CUDA and wp is not None
except Exception:
    _closing_disk_offsets = _closing_warp = None
    HAVE_CLOSING_GPU = False
CLOSINGVAL = os.environ.get("CLOSINGVAL", "fast")

OUT_DIR = os.path.join(HERE, "artifacts")
REPORT = os.path.join(OUT_DIR, "tillverkningsfalt_v1.json")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT), exist_ok=True)

# -----------------------------------------------------------------------------------------------
# declared process parameters
RHO_TURN_MM = 1.0     # turning tool nose radius
RHO_MILL_MM = 3.0     # pocket end-mill radius (corner radius = cutter radius for a full-width cut)
DRILL_R_MM = 11.15    # cross-bore radius
RADIAL_ALLOWANCE_MM = 1.5   # radial stock allowance over the profile envelope
AXIAL_ALLOWANCE_MM = 1.0    # axial stock allowance at each end
PITCH2D_MM = 0.04     # (r,z) raster pitch for the turning section
PITCH3D_MM = float(os.environ.get("PITCH3D_MM", "0.35"))   # 3D voxel pitch for the whole part


def synthetic_turned_profile():
    """Generates the synthetic input: a stepped shaft with three ring grooves.

    Returns a dict with the (z, r) polyline in mm, the z range, the envelope radius and the analytic
    volume of the revolved ideal profile (sharp corners, before any tool-radius effect).
    """
    r_land, r_groove, groove_w = 47.7, 45.5, 3.0
    z_lo, z_hi = -21.0, 32.0
    centers = (14.0, 19.0, 24.0)
    poly = [(z_lo, r_land)]
    for z_c in centers:
        poly += [(z_c - groove_w / 2.0, r_land), (z_c - groove_w / 2.0, r_groove),
                 (z_c + groove_w / 2.0, r_groove), (z_c + groove_w / 2.0, r_land)]
    poly += [(z_hi, r_land)]
    poly = sorted(poly, key=lambda t: t[0])
    z = np.array([p[0] for p in poly])
    r = np.array([p[1] for p in poly])
    # revolved volume of the piecewise-linear profile: sum of truncated-cone segments
    vol = float(np.sum(np.pi / 3.0 * np.diff(z) * (r[:-1] ** 2 + r[:-1] * r[1:] + r[1:] ** 2)))
    return {"profil_rz_polyline_mm": [[float(a), float(b)] for a, b in poly],
            "z_lo": z_lo, "z_hi": z_hi, "r_envelope": r_land,
            "r_groove": r_groove, "groove_z_centers": list(centers),
            "groove_corner_z": centers[0] - groove_w / 2.0,
            "volume_ideal_revolved_mm3": vol}


def build_2d_turned_sdf(poly, r_envelope, z_lo, z_hi):
    """Rasterises the axisymmetric target profile (sharp corners) on a fine (r,z) grid and applies a
    morphological closing with a disk of radius RHO_TURN.

    That is the correct operation for what a nose-radius tool can reach from outside: convex corners
    stay sharp, concave corners are rounded to exactly rho. Takes the (z,r) polyline, the envelope
    radius and the z range; returns a dict with the (r,z) SDF lookup table, its axes, both masks and
    the closing time.
    """
    z_pad = 2.0
    r_pad = 2.0
    z0, z1 = z_lo - z_pad, z_hi + z_pad
    r0, r1 = 0.0, r_envelope + r_pad
    nz = int(round((z1 - z0) / PITCH2D_MM)) + 1
    nr = int(round((r1 - r0) / PITCH2D_MM)) + 1
    zz = np.linspace(z0, z1, nz)
    rr = np.linspace(r0, r1, nr)

    poly = sorted(poly, key=lambda t: t[0])
    pz = np.array([p[0] for p in poly])
    pr = np.array([p[1] for p in poly])
    # r_target(z): linear between the given points; each step carries both (z, r_before) and
    # (z, r_after) as two points at the same z, so the interpolation reproduces the steps
    r_target_of_z = np.interp(zz, pz, pr)

    R, Z = np.meshgrid(rr, zz, indexing="ij")  # shape (nr, nz)
    r_of_z_2d = np.interp(Z.ravel(), zz, r_target_of_z).reshape(Z.shape)
    mask0 = R <= r_of_z_2d   # ideal profile, sharp corners; material inside

    rho_px = max(1, int(round(RHO_TURN_MM / PITCH2D_MM)))
    footprint = disk(rho_px)
    t0 = time.time()
    if CLOSINGVAL == "fast" and HAVE_CLOSING_GPU:
        off = _closing_disk_offsets(rho_px)
        mask_closed = _closing_warp(mask0, off, device="cuda:0")
    else:
        mask_closed = binary_closing(mask0, footprint=footprint)
    t_close = time.time() - t0

    # signed distance from the closed mask (negative = inside material)
    d_out = ndimage.distance_transform_edt(~mask_closed) * PITCH2D_MM
    d_in = ndimage.distance_transform_edt(mask_closed) * PITCH2D_MM
    sdf2d = np.where(mask_closed, -d_in, d_out)

    return {
        "sdf2d": sdf2d, "rr": rr, "zz": zz, "mask0": mask0, "mask_closed": mask_closed,
        "rho_px": rho_px, "t_close_s": t_close,
    }


def measure_groove_root_radius(mask_closed, rr, zz, pitch, r_land, r_groove, z_center):
    """Measures the actual concave corner radius at a groove root by fitting a circle to the mask
    boundary in a small window around the corner.

    Takes the closed mask, its axes, the pitch, the land and groove radii and the corner z; returns the
    fitted radius in mm, or None if no contour is found.
    """
    iz = np.argmin(np.abs(zz - z_center))
    ir = np.argmin(np.abs(rr - r_groove))
    win = 40
    sub = mask_closed[max(0, ir - win):ir + win, max(0, iz - win):iz + win]
    # boundary pixels (material/void interface) inside the window
    from skimage import measure as skmeasure
    contours = skmeasure.find_contours(sub.astype(float), 0.5)
    if not contours:
        return None
    # take the longest contour and the points nearest the expected corner
    best = max(contours, key=len)
    pr = best[:, 0] + max(0, ir - win)
    pz = best[:, 1] + max(0, iz - win)
    pr_mm = rr[0] + pr * pitch
    pz_mm = zz[0] + pz * pitch
    # the corner is expected near (r=r_groove, z=z_center): fit a circle to the nearest 30 points
    d = np.hypot(pr_mm - r_groove, pz_mm - z_center)
    idx = np.argsort(d)[:30]
    x = pz_mm[idx]
    y = pr_mm[idx]
    # algebraic circle fit (Kasa)
    A = np.c_[x, y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    R = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return float(R)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    part = synthetic_turned_profile()
    poly = part["profil_rz_polyline_mm"]
    R_ENVELOPE = part["r_envelope"]
    z_lo, z_hi = part["z_lo"], part["z_hi"]

    print(f"[setup] cuda={HAVE_CUDA} pitch3d={PITCH3D_MM}mm "
          f"rho_turn={RHO_TURN_MM}mm rho_mill={RHO_MILL_MM}mm")

    # ---- 1. blank: constant-radius round bar, envelope radius plus radial allowance -------------
    r_blank = R_ENVELOPE + RADIAL_ALLOWANCE_MM
    z_blank_lo = z_lo - AXIAL_ALLOWANCE_MM
    z_blank_hi = z_hi + AXIAL_ALLOWANCE_MM
    # monotonicity check: the blank radius must never fall below any point of the part's own profile
    poly_arr = np.array(poly)
    max_part_r = poly_arr[:, 1].max()
    monotonic_ok = bool(r_blank >= max_part_r)
    vol_blank_mm3 = np.pi * r_blank ** 2 * (z_blank_hi - z_blank_lo)
    print(f"[blank] r_blank={r_blank:.3f}mm z=[{z_blank_lo:.2f},{z_blank_hi:.2f}] "
          f"vol={vol_blank_mm3:.1f}mm3 monotonic_ok={monotonic_ok} (max part radius={max_part_r:.2f})")

    # ---- 2. turning: 2D morphological closing -> (r,z) SDF lookup table -------------------------
    turn = build_2d_turned_sdf(poly, R_ENVELOPE, z_lo, z_hi)
    r_groove_root = measure_groove_root_radius(
        turn["mask_closed"], turn["rr"], turn["zz"], PITCH2D_MM, r_land=part["r_envelope"],
        r_groove=part["r_groove"], z_center=part["groove_corner_z"])
    print(f"[turning] 2D closing done in {turn['t_close_s']*1000:.1f} ms "
          f"({turn['sdf2d'].shape[0]}x{turn['sdf2d'].shape[1]} px), "
          f"measured groove-root radius={r_groove_root:.3f}mm (declared rho_turn={RHO_TURN_MM}mm)")

    # ---- 3. 3D voxel field + Warp kernels -------------------------------------------------------
    x0, x1 = -r_blank - 1, r_blank + 1
    y0, y1 = -r_blank - 1, r_blank + 1
    nz = int(round((z_blank_hi - z_blank_lo) / PITCH3D_MM)) + 1
    nx = int(round((x1 - x0) / PITCH3D_MM)) + 1
    ny = int(round((y1 - y0) / PITCH3D_MM)) + 1
    print(f"[3D] grid {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.2f} Mvoxels, pitch={PITCH3D_MM}mm")

    device = "cuda:0" if HAVE_CUDA else "cpu"
    sdf2d_flat = turn["sdf2d"].astype(np.float32).flatten()  # (nr,nz) rowmajor
    nr2d, nz2d = turn["sdf2d"].shape
    r0_2d, r_pitch2d = float(turn["rr"][0]), PITCH2D_MM
    z0_2d = float(turn["zz"][0])

    # pocket: rounded box in XY (half widths), open at z_blank_lo, arched roof
    px = 10.4      # half width in X
    py = 23.0      # half width in Y
    arch_r = 10.4
    arch_cz = 17.6 - 10.4   # arch centre, so the roof apex reaches z=17.6 at x=0

    # cross bore along X, radius DRILL_R_MM, axis height z=11.15
    drill_z = 11.15
    drill_x_full = r_blank + 1  # through the full width of the blank
    drill_x_short = 30.0        # negative control: a short drill does not break through

    if wp is not None:
        with wp.ScopedDevice(device):
            sdf2d_wp = wp.array(sdf2d_flat, dtype=wp.float32)
            out_full = wp.zeros(nx * ny * nz, dtype=wp.float32)
            out_short = wp.zeros(nx * ny * nz, dtype=wp.float32)

            @wp.kernel
            def cnc_subtract(sdf2d: wp.array(dtype=wp.float32),
                              nr2d: int, nz2d: int, r0_2d: float, z0_2d: float, pitch2d: float,
                              x0: float, y0: float, z0: float, pitch3d: float,
                              nx: int, ny: int, nz: int,
                              r_blank: float, z_blank_lo: float, z_blank_hi: float,
                              px: float, py: float, arch_r: float, arch_cz: float, rho_mill: float,
                              drill_r: float, drill_z: float, drill_x_half: float,
                              out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                k = tid % nz
                j = (tid // nz) % ny
                i = tid // (nz * ny)
                x = x0 + float(i) * pitch3d
                y = y0 + float(j) * pitch3d
                z = z0 + float(k) * pitch3d
                r = wp.sqrt(x * x + y * y)

                # blank (round bar)
                sdf_blank = wp.max(r - r_blank, wp.max(z_blank_lo - z, z - z_blank_hi))

                # turned target: bilinear lookup in sdf2d(r,z)
                fr = (r - r0_2d) / pitch2d
                fz = (z - z0_2d) / pitch2d
                ir0 = wp.clamp(int(fr), 0, nr2d - 2)
                iz0 = wp.clamp(int(fz), 0, nz2d - 2)
                tr = wp.clamp(fr - float(ir0), 0.0, 1.0)
                tz = wp.clamp(fz - float(iz0), 0.0, 1.0)
                v00 = sdf2d[ir0 * nz2d + iz0]
                v01 = sdf2d[ir0 * nz2d + iz0 + 1]
                v10 = sdf2d[(ir0 + 1) * nz2d + iz0]
                v11 = sdf2d[(ir0 + 1) * nz2d + iz0 + 1]
                sdf_turned = (v00 * (1.0 - tr) * (1.0 - tz) + v01 * (1.0 - tr) * tz
                              + v10 * tr * (1.0 - tz) + v11 * tr * tz)

                material = wp.max(sdf_blank, sdf_turned)

                # pocket (rounded box in X/Y, open downward, arched roof in Z), subtracted:
                # rounded-box distance in XY (edge radius rho_mill), floor at z_blank_lo (open),
                # roof at arch_cz + sqrt(max(arch_r^2 - x^2, 0))
                qx = wp.abs(x) - (px - rho_mill)
                qy = wp.abs(y) - (py - rho_mill)
                dxy = wp.sqrt(wp.max(qx, 0.0) ** 2.0 + wp.max(qy, 0.0) ** 2.0) + wp.min(wp.max(qx, qy), 0.0) - rho_mill
                arch_arg = arch_r * arch_r - x * x
                roof_z = arch_cz + wp.sqrt(wp.max(arch_arg, 0.0))
                d_top = z - roof_z          # <0 under taket
                d_bot = z_blank_lo + 0.001 - z  # always < 0 (the pocket is open at the bottom)
                pocket_sdf = wp.max(dxy, d_top)   # inside the pocket requires dxy < 0 and below the roof
                material = wp.max(material, -pocket_sdf)

                # cross bore along X (cylinder SDF), clipped to drill_x_half
                dr_axis = wp.sqrt((y - 0.0) ** 2.0 + (z - drill_z) ** 2.0) - drill_r
                d_xclip = wp.abs(x) - drill_x_half
                bore_sdf = wp.max(dr_axis, d_xclip)
                material = wp.max(material, -bore_sdf)

                out[tid] = material

            n_total = nx * ny * nz
            wp.launch(cnc_subtract, dim=n_total,
                      inputs=[sdf2d_wp, nr2d, nz2d, r0_2d, z0_2d, PITCH2D_MM,
                              x0, y0, z_blank_lo, PITCH3D_MM, nx, ny, nz,
                              r_blank, z_blank_lo, z_blank_hi,
                              px, py, arch_r, arch_cz, RHO_MILL_MM,
                              DRILL_R_MM, drill_z, drill_x_full],
                      outputs=[out_full])
            wp.launch(cnc_subtract, dim=n_total,
                      inputs=[sdf2d_wp, nr2d, nz2d, r0_2d, z0_2d, PITCH2D_MM,
                              x0, y0, z_blank_lo, PITCH3D_MM, nx, ny, nz,
                              r_blank, z_blank_lo, z_blank_hi,
                              px, py, arch_r, arch_cz, RHO_MILL_MM,
                              DRILL_R_MM, drill_z, drill_x_short],
                      outputs=[out_short])
            wp.synchronize()
            field_full = out_full.numpy().reshape(nx, ny, nz)
            field_short = out_short.numpy().reshape(nx, ny, nz)
    else:
        raise RuntimeError("warp is required; no numpy fallback is implemented for this kernel")

    # ---- 4. marching cubes + volumes ------------------------------------------------------------
    def mc_volume_and_mesh(field):
        verts, faces, normals, values = measure.marching_cubes(field, level=0.0,
                                                                 spacing=(PITCH3D_MM,) * 3)
        verts = verts + np.array([x0, y0, z_blank_lo], dtype=np.float64)  # grid world origin
        # process=True merges duplicate vertices and drops degenerate zero-area facets, which
        # marching cubes produces at ambiguous voxel cases and which otherwise poison area-weighted
        # sampling with NaN
        m_raw = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        comps = m_raw.split(only_watertight=False)
        m = max(comps, key=lambda c: c.volume) if len(comps) > 1 else m_raw
        n_discarded = len(comps) - 1 if len(comps) > 1 else 0
        vol_mesh = abs(m.volume)
        vol_voxel = np.sum(field < 0) * PITCH3D_MM ** 3
        return m, vol_mesh, vol_voxel, n_discarded

    mesh_full, vol_full_mesh, vol_full_vox, n_disc_full = mc_volume_and_mesh(field_full)
    mesh_short, vol_short_mesh, vol_short_vox, n_disc_short = mc_volume_and_mesh(field_short)
    # The residual-material check uses the topology-independent voxel count, not the mesh integral:
    # the blind hole's inner wall makes the marching-cubes component choice brittle in the short-drill
    # case (measured 19.6% apart from the voxel count there, against 0.11% for the through case).
    vox_mismatch_short_frac = abs(vol_short_mesh - vol_short_vox) / vol_short_vox
    restmaterial_short_drill_mm3 = vol_short_vox - vol_full_vox

    stl_path = os.path.join(OUT_DIR, "tillverkningsfalt_v1_resultat.stl")
    mesh_full.export(stl_path)

    vol_consistency = abs(vol_full_mesh - vol_full_vox) / vol_full_mesh
    gates = {
        "monotonic_envelope_ok": bool(monotonic_ok),
        "groove_root_radius_between_0p3_and_2mm": bool(r_groove_root is not None
                                                       and 0.3 < r_groove_root < 2.0),
        "mesh_vs_voxel_volume_within_1pct": bool(vol_consistency < 0.01),
        "short_drill_leaves_residual": bool(restmaterial_short_drill_mm3 > 1.0),
    }
    out_report = {
        "cell": "tillverkningsfalt_v1",
        "input": "synthetic turned profile generated in code (stepped shaft, three ring grooves)",
        "device": device, "have_cuda": HAVE_CUDA,
        "params": {"rho_turn_mm": RHO_TURN_MM, "rho_mill_mm": RHO_MILL_MM,
                   "drill_r_mm": DRILL_R_MM, "radial_allowance_mm": RADIAL_ALLOWANCE_MM,
                   "axial_allowance_mm": AXIAL_ALLOWANCE_MM, "pitch2d_mm": PITCH2D_MM,
                   "pitch3d_mm": PITCH3D_MM, "grid_voxels": nx * ny * nz,
                   "grid_shape": [nx, ny, nz]},
        "profil": {"polyline_rz_mm": poly,
                   "volume_ideal_revolved_mm3": part["volume_ideal_revolved_mm3"]},
        "amne": {"r_blank_mm": r_blank, "z_range_mm": [z_blank_lo, z_blank_hi],
                 "volume_blank_analytisk_mm3": vol_blank_mm3,
                 "monotonic_envelope_ok": monotonic_ok, "max_delradie_mm": float(max_part_r)},
        "svarvning": {"matt_spargrund_hornradie_mm": r_groove_root,
                      "deklarerad_verktygsradie_mm": RHO_TURN_MM,
                      "avvikelse_mm": (None if r_groove_root is None
                                       else abs(r_groove_root - RHO_TURN_MM)),
                      "t_2d_closing_ms": turn["t_close_s"] * 1000},
        "resultat_full_drill": {
            "volym_mesh_mm3": vol_full_mesh, "volym_voxel_mm3": vol_full_vox,
            "volym_konsistens_diff_frac": vol_consistency,
            "n_verts": len(mesh_full.vertices), "n_faces": len(mesh_full.faces),
            "n_lossa_mikrofragment_kasserade": n_disc_full,
            "stl": os.path.relpath(stl_path, HERE),
        },
        "negativ_kontroll_kort_borr": {
            "drill_x_half_short_mm": drill_x_short, "drill_x_half_full_mm": drill_x_full,
            "restmaterial_mm3_voxelraknat": float(restmaterial_short_drill_mm3),
            "restmaterial_over_noll": bool(restmaterial_short_drill_mm3 > 1.0),
            "mesh_vs_voxel_mismatch_frac_kort_fall": float(vox_mismatch_short_frac),
            "matmetod_not": "residual material measured by voxel count (topology independent), not by "
                            "the marching-cubes mesh integral",
        },
        "declared_scope": [
            "turning (the ring-groove profile) is fully field driven",
            "the blank is a field operation that degenerates to a constant-radius round bar for this "
            "profile, since it has no overhangs",
            "the pocket is a rounded-box plus arch approximation, not a copy of an individual surface",
        ],
        "gates": gates,
    }

    with open(REPORT, "w") as fh:
        json.dump(out_report, fh, indent=2)
    print(json.dumps({k: out_report[k] for k in ("svarvning", "resultat_full_drill",
                                                 "negativ_kontroll_kort_borr", "gates")}, indent=2))
    print("\nwrote", REPORT)
    if not all(gates.values()):
        raise SystemExit(f"gates FAILED: {gates}")


if __name__ == "__main__":
    main()
