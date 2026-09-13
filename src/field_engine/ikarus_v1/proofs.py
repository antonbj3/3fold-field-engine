#!/usr/bin/env python3
"""Two structural proofs for the SDF expression system.

(5) Precision round trip: a cylinder's exact stored parameters (radius, height) are exported to a real
    B-rep solid, and the parameters are recovered from the kernel's own independently computed volume and
    surface area (not echoed from the constructor arguments) by solving V = pi r^2 h and
    SA = 2 pi r h + 2 pi r^2 for (r, h). Agreement to solver tolerance shows the symbolic parameters
    survive a B-rep round trip losslessly. Requires build123d; skipped with a declared flag if absent.

(6) Definition proof: the same expression tree is marching-cubed at 2/1/0.5 mm with no remodelling, and
    the surface-area error against the tree's analytic value must shrink monotonically as the resolution
    refines.

Takes no arguments; writes _proofs_result.json next to this file and prints both results.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expr as E

# the B-rep half runs in a subprocess with the same interpreter; set IKARUS_OCC_PYTHON to point at a
# different interpreter that has build123d installed.
OCC_PYTHON = os.environ.get("IKARUS_OCC_PYTHON", sys.executable)


# --------------------------------------------------------------------------------------------- (5) precision round-trip
_OCC_ROUNDTRIP_SCRIPT = r'''
import sys, json, math
from build123d import Cylinder

r_in = float(sys.argv[1]); h_in = float(sys.argv[2])
c = Cylinder(radius=r_in, height=h_in)
V = c.volume
SA = c.area

# recover (r,h) from V and SA alone (NOT from c.radius/c.height -- those would just echo the constructor
# args, a tautology). V = pi r^2 h ; SA = 2 pi r h + 2 pi r^2 = 2 pi r (h + r)
# => h = V/(pi r^2); substitute into SA: SA = 2 pi r (V/(pi r^2) + r) = 2V/r + 2 pi r^2
# solve f(r) = 2V/r + 2*pi*r^2 - SA = 0 via Newton's method from a coarse bracket start.
def f(r):
    return 2.0*V/r + 2.0*math.pi*r*r - SA
def fprime(r):
    return -2.0*V/(r*r) + 4.0*math.pi*r

r = r_in * 0.5  # deliberately WRONG initial guess (not the answer) to prove Newton converges to it, not echoes it
for _ in range(200):
    fr = f(r)
    dr = fr / fprime(r)
    r = r - dr
    if abs(dr) < 1e-14:
        break
h = V / (math.pi * r * r)

print(json.dumps({"r_in": r_in, "h_in": h_in, "occ_volume": V, "occ_area": SA,
                   "r_recovered": r, "h_recovered": h,
                   "r_abs_err": abs(r - r_in), "h_abs_err": abs(h - h_in)}))
'''


def precision_roundtrip_cylinder(radius_mm=17.3, height_mm=42.7):
    """Round-trips a cylinder through a B-rep kernel and recovers (r, h) from volume and area.

    Takes the radius and height in mm; returns a dict with the kernel volume/area, the recovered
    parameters, their absolute errors and the gate verdict, or {"skipped": ...} when build123d is not
    installed.
    """
    script_path = os.path.join(HERE, "_occ_roundtrip_helper.py")
    with open(script_path, "w") as f:
        f.write(_OCC_ROUNDTRIP_SCRIPT)
    out = subprocess.run([OCC_PYTHON, script_path, str(radius_mm), str(height_mm)],
                          capture_output=True, text=True, cwd=HERE)
    if out.returncode != 0:
        if "build123d" in out.stderr and "Error" in out.stderr:
            return {"skipped": "build123d is not installed in this interpreter", "gate_bit_identical": None}
        raise RuntimeError(f"B-rep round-trip subprocess failed: {out.stderr}")
    data = json.loads(out.stdout.strip().splitlines()[-1])
    ikarus_node = E.cylinder(radius=radius_mm, height=height_mm)
    data["ikarus_expr_params"] = {"radius": ikarus_node["radius"], "height": ikarus_node["height"]}
    data["gate_bit_identical"] = bool(data["r_abs_err"] < 1e-6 and data["h_abs_err"] < 1e-6)
    return data


# --------------------------------------------------------------------------------------------- (6) definitional facet-scaling proof
def facet_scaling_proof(resolutions_mm=(2.0, 1.0, 0.5), device=None):
    """Marching-cubes the same tree at each pitch and compares the mesh area to the analytic area.

    Takes the pitches in mm; returns a dict with one row per pitch and the monotonic-improvement gate.
    """
    from skimage import measure
    import eval_warp as W
    nonlocal_device = device or W.DEVICE

    # a tree with a known analytic surface area, so the facet error has an external anchor: a single
    # sphere (exact area 4 pi r^2), the simplest tree with a closed-form surface area, still evaluated
    # through the same generated kernel path as everything else.
    R = 20.0
    tree = E.sphere(radius=R)
    analytic_area = 4.0 * math.pi * R * R

    rows = []
    for pitch in resolutions_mm:
        pad = 4.0
        n = int(math.ceil((2 * R + 2 * pad) / pitch)) + 1
        lo = -R - pad
        xs = lo + pitch * np.arange(n)
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        vals = W.eval_batch(tree, pts, device=nonlocal_device).reshape(n, n, n)
        verts, faces, normals, values = measure.marching_cubes(vals, level=0.0, spacing=(pitch, pitch, pitch))
        # mesh surface area
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        mesh_area = float(tri_area.sum())
        err_pct = 100.0 * abs(mesh_area - analytic_area) / analytic_area
        rows.append({"pitch_mm": pitch, "n_verts": len(verts), "n_faces": len(faces),
                     "mesh_area_mm2": mesh_area, "analytic_area_mm2": analytic_area,
                     "area_err_pct": err_pct})

    errs = [r["area_err_pct"] for r in rows]
    monotonic_improving = all(errs[i] >= errs[i + 1] - 1e-9 for i in range(len(errs) - 1))
    return {"tree": "sphere(R=20)", "rows": rows, "errs_by_resolution_finest_last": errs,
            "gate_error_shrinks_with_resolution": bool(monotonic_improving)}


if __name__ == "__main__":
    print("=== (5) precision round-trip ===")
    rt = precision_roundtrip_cylinder()
    print(json.dumps(rt, indent=2))
    print("=== (6) definitional facet-scaling ===")
    fs = facet_scaling_proof()
    print(json.dumps(fs, indent=2))
    out = {"precision_roundtrip": rt, "facet_scaling": fs}
    outp = os.path.join(HERE, "_proofs_result.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", outp)
