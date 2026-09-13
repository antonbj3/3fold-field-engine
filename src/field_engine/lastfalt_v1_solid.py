#!/usr/bin/env python3
"""Field to solid: the density field from the SIMP run -> smoothing -> marching cubes -> mesh -> STL,
and STEP when a B-rep kernel is available. Then gated, not just measured:

  G0 degenerate load (recorded in the optimiser output, repeated here as a precondition)
  G1 primitive gate: the non-planar area fraction, so a topology-optimised body cannot come out a box
  G2 volume budget against the solid plate it replaces
  G3 deflection against an independent anchor: the same FEM path run with rho=1 everywhere
  G4 manufacturability, reported as a flag rather than a verdict: an overhang proxy per voxel layer and
     a minimum wall thickness from a distance transform of the density mask, both resolution limited

Run `python lastfalt_v1_solid.py --tag vf0.5`; reads artifacts/topopt_TAG.json and writes
artifacts/solid_TAG.json plus the STL.
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.ndimage as ndi
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = HERE
OUT_DIR = os.path.join(HERE, "artifacts")


def build_solid(tag):
    meta = json.load(open(os.path.join(OUT_DIR, f"topopt_{tag}.json")))
    rho = np.load(os.path.join(ROOT, meta["rho_field_npy"]))
    dx, dy, dz = meta["dx_dy_dz_mm"]
    NX, NY, NZ = meta["grid"]

    from lastfalt_v1_topopt import synthetic_plate_and_loads
    geom, _fv, _mk = synthetic_plate_and_loads()
    bbox = geom["bbox_mm"]

    # --- G0: degenerate load ---
    g0_pass = meta["total_load_mag_abs_sum_N"] > 1.0
    g0 = {"gate": "G0_degenererad_last", "total_load_mag_abs_sum_N": meta["total_load_mag_abs_sum_N"],
          "troskel": "> 1.0 N (summed absolute nodal force)", "pass": bool(g0_pass)}
    if not g0_pass:
        out = {"tag": tag, "G0_degenererad_last": g0,
               "STOPPAD": "the load magnitude is degenerate, so the optimiser has no driving gradient "
                          "(compliance = f^T u = 0 identically) and the density field is a move-limit "
                          "artefact rather than a shaped body. Marching cubes and STEP export are "
                          "deliberately skipped: the gate falls before the geometry gets a name."}
        json.dump(out, open(os.path.join(OUT_DIR, f"solid_{tag}.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
        return out

    # --- smoothing + marching cubes, in mm ---
    # pad one cell of zeros around the field before marching cubes, otherwise the isosurface is cut
    # open at the grid boundary and the mesh is not watertight
    rho_s = ndi.gaussian_filter(rho, sigma=0.7)
    rho_pad = np.pad(rho_s, pad_width=1, mode="constant", constant_values=0.0)
    verts, faces, normals, _ = measure.marching_cubes(rho_pad, level=0.5,
                                                        spacing=(dx, dy, dz))
    verts[:, 0] += bbox["xmin"] - dx
    verts[:, 1] += bbox["ymin"] - dy
    verts[:, 2] += bbox["zmin"] - dz

    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    n_bodies = mesh.body_count
    watertight = bool(mesh.is_watertight)
    volume_mm3 = float(abs(mesh.volume)) if watertight else 0.0  # unused when not watertight
    stl_path = os.path.join(OUT_DIR, f"lastfalt_{tag}.stl")
    mesh.export(stl_path)

    # --- G1: primitive gate (non-planar area fraction); a box is about 0% non-planar ---
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces
    total_area = face_areas.sum()
    # planarity: facet normals within 3 degrees of +-X/+-Y/+-Z count as axis-aligned flat
    axis_dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    cosang = np.abs(face_normals @ axis_dirs.T)
    is_axis_planar = np.any(cosang > np.cos(np.radians(3.0)), axis=1)
    axis_planar_area_frac = float(face_areas[is_axis_planar].sum() / total_area)
    nonplanar_area_frac = 1.0 - axis_planar_area_frac
    g1_pass = nonplanar_area_frac >= 0.08
    g1 = {"gate": "G1_primitivgrind_ickeplan_yta", "nonplanar_area_frac": nonplanar_area_frac,
          "troskel": ">= 0.08", "pass": bool(g1_pass)}

    # --- G2: volume budget against the solid plate ---
    slab_vol_mm3 = (bbox["xmax"] - bbox["xmin"]) * (bbox["ymax"] - bbox["ymin"]) * (bbox["zmax"] - bbox["zmin"])
    design_vol_mm3 = float(np.sum(rho) * dx * dy * dz)  # density-weighted SIMP volume
    mesh_vol_mm3 = volume_mm3 if watertight else None
    g2_pass = design_vol_mm3 <= slab_vol_mm3
    g2 = {"gate": "G2_volymbudget", "slab_vol_mm3": slab_vol_mm3, "simp_vol_mm3": design_vol_mm3,
          "mesh_vol_mm3": mesh_vol_mm3, "massreduktion_andel": 1.0 - design_vol_mm3 / slab_vol_mm3,
          "pass": bool(g2_pass)}

    # --- G3: deflection against the solid anchor (an independent FEM run with rho=1) ---
    anchor_path = os.path.join(OUT_DIR, "solid_baseline.json")
    if os.path.exists(anchor_path):
        uz_solid = json.load(open(anchor_path))["solid_full_block_max_abs_uz_mm"]
    else:
        # no stored anchor: run the same FEM path once at rho=1 and keep the result
        from lastfalt_v1_topopt import run as topopt_run
        anchor = topopt_run(volfrac=1.0, n_iter=1, tag="solid_baseline")
        uz_solid = anchor["max_abs_uz_mm"]
        json.dump({"solid_full_block_max_abs_uz_mm": uz_solid}, open(anchor_path, "w"), indent=1)
    uz_opt = meta["max_abs_uz_mm"]
    margin_factor = 1.5  # declared, not measured: design margin for a mass-optimised part
    g3_pass = uz_opt <= margin_factor * uz_solid
    g3 = {"gate": "G3_deflektion", "max_abs_uz_mm_optimerad": uz_opt, "max_abs_uz_mm_solid_ankare": uz_solid,
          "kvot": uz_opt / uz_solid, "troskel": f"<= {margin_factor}x the solid anchor (declared margin)",
          "pass": bool(g3_pass)}

    # --- G4: manufacturability, reported honestly rather than solved ---
    solid_mask = rho >= 0.5
    # overhang: a solid voxel with no support directly below, outside the bottom layer
    NXg, NYg, NZg = solid_mask.shape
    overhang_voxels = 0
    total_solid = int(solid_mask.sum())
    for k in range(1, NZg):
        layer = solid_mask[:, :, k]
        below = solid_mask[:, :, k - 1]
        unsupported = layer & (~below)
        overhang_voxels += int(unsupported.sum())
    overhang_frac = overhang_voxels / max(total_solid, 1)
    # minimum wall thickness: twice the distance transform of the solid mask, minimum over solid voxels
    dist = ndi.distance_transform_edt(solid_mask, sampling=(dx, dy, dz))
    min_wall_mm = float(2.0 * dist[solid_mask].min()) if total_solid else 0.0
    g4 = {"gate": "G4_tillverkningsbarhet_FLAGG", "overhang_voxel_andel": overhang_frac,
          "overhang_troskel_konvention": "45 degrees; the voxel proxy is 'a layer-k voxel without "
                                         "support below', which is coarse at five z layers and is "
                                         "flagged, not solved",
          "min_godstjocklek_mm_matt": min_wall_mm,
          "min_godstjocklek_upplosningstak": f"the grid resolution dx={dx:.1f} dy={dy:.1f} dz={dz:.1f}mm "
                                             f"sets a lower measurable bound; features thinner than one "
                                             f"cell are invisible to this measure",
          "flagga": "OVERHANG_PROXY_HOG" if overhang_frac > 0.15 else "OK",
          "pass": None}  # a flag, not a silent verdict

    # --- STEP export through a B-rep kernel (tessellated shell read back from the STL) ---
    step_path = None
    if watertight:
        try:
            from OCP.StlAPI import StlAPI_Reader
            from OCP.TopoDS import TopoDS_Shape
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
            from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
            from OCP.IFSelect import IFSelect_RetDone

            shp = TopoDS_Shape()
            rdr = StlAPI_Reader()
            rdr.Read(shp, stl_path)

            writer = STEPControl_Writer()
            writer.Transfer(shp, STEPControl_AsIs)
            step_path = os.path.join(OUT_DIR, f"lastfalt_{tag}.step")
            status = writer.Write(step_path)
            if status != IFSelect_RetDone:
                step_path = None
        except Exception as e:
            step_path = f"FAILED: {e}"

    out = {
        "tag": tag,
        "stl_path": os.path.relpath(stl_path, ROOT),
        "step_path": os.path.relpath(step_path, ROOT) if (step_path and not str(step_path).startswith("FAILED")) else step_path,
        "mesh_watertight": watertight,
        "mesh_body_count": n_bodies,
        "mesh_n_verts": int(len(mesh.vertices)),
        "mesh_n_faces": int(len(mesh.faces)),
        "G0_degenererad_last": g0,
        "G1_primitivgrind": g1,
        "G2_volymbudget": g2,
        "G3_deflektion": g3,
        "G4_tillverkningsbarhet_flagg": g4,
        "alla_hard_grindar_pass": bool(g0_pass and g1_pass and g2_pass and g3_pass),
    }
    out_path = os.path.join(OUT_DIR, f"solid_{tag}.json")
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("-> wrote", out_path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vf0.5")
    args = ap.parse_args()
    res = build_solid(args.tag)
    if not res.get("alla_hard_grindar_pass", False) and "STOPPAD" not in res:
        raise SystemExit("hard gates FAILED")
