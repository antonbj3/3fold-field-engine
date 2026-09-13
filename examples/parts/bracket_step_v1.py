#!/usr/bin/env python3
"""Synthetic bracket as a STEP file plus a parts_manifest_v1, input for the drawing generator.

The bracket is the repository's existing recipe part (examples/recipes/bracket_v1_recipe.json)
built at one declared theta: a filleted plate with four through holes and a rectangular slot cut
through it. The solid is exported to STEP and listed in a parts_manifest_v1 with its measured
volume and the mass that volume gives at a declared density, so the drawing's title block and the
parity gate's K3 channel have a source that is not the drawing.

Run: python bracket_step_v1.py [out_dir]
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bracket_part_v1 as BP  # noqa: E402

THETA = {"plate_w": 150.0, "plate_h": 100.0, "thk": 10.0, "hole_d": 10.0}
DENSITY_KG_MM3 = 2.70e-6      # declared: aluminium
MATERIAL = "EN AW-6082"


def build_shape():
    """Builds the bracket solid at THETA."""
    return BP.build(dict(THETA))


def write_step_and_manifest(out_dir: str) -> str:
    """Writes bracket_v1.step and parts_manifest_v1.json into out_dir; returns the manifest path."""
    import build123d as bd

    os.makedirs(out_dir, exist_ok=True)
    shape = build_shape()
    step_path = os.path.join(out_dir, "bracket_v1.step")
    bd.export_step(shape, step_path)
    vol = float(shape.volume)
    bb = shape.bounding_box()
    manifest = {
        "schema": "parts_manifest_v1",
        "part_source": "examples/recipes/bracket_v1_recipe.json",
        "theta": THETA,
        "density_kg_mm3": DENSITY_KG_MM3,
        "parts": [{
            "namn": "bracket_v1",
            "part": "bracket_v1",
            "step": step_path,
            "transform_mm": None,
            "volume_mm3": round(vol, 4),
            "mass_kg": round(vol * DENSITY_KG_MM3, 3),
            "material": MATERIAL,
            "bbox": [round(bb.min.X, 4), round(bb.min.Y, 4), round(bb.min.Z, 4),
                     round(bb.max.X, 4), round(bb.max.Y, 4), round(bb.max.Z, 4)],
        }],
    }
    man_path = os.path.join(out_dir, "parts_manifest_v1.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print("STEP %s  volume %.4f mm3  mass %.3f kg  bbox %s"
          % (step_path, vol, vol * DENSITY_KG_MM3, manifest["parts"][0]["bbox"]))
    return man_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "artifacts")
    write_step_and_manifest(out)
