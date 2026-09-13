#!/usr/bin/env python3
"""Generator for the small-feature test part: one plate with five through holes.

The plate is a rectangular block of fixed thickness with through holes of radii 1, 2, 4, 8 and 16 mm
drilled along X, spaced so that no two holes and no hole and no outer wall come within 10 mm of each
other. The point of the part is that one single rasterisation pitch has to resolve a 1 mm and a 16 mm
cylinder at the same time, which is exactly the case in which a block-sparse SDF turns a through hole
into a chain of blobs.

build(...) returns the build123d solid, mesh(...) returns (V, T) as faltkarna_v1_mesh_to_sdf expects
(float64 vertices, int64 triangles, closed and outward oriented), and running the file as a script
writes the STL next to it. The STL is generated, not stored.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STL_PATH = os.path.join(HERE, "holed_plate_v1.stl")

HOLE_RADII_MM = (1.0, 2.0, 4.0, 8.0, 16.0)
THICKNESS_MM = 6.0          # fixed plate thickness for every hole
GAP_MM = 10.0               # material between two holes and between a hole and the outer wall
# linear/angular tessellation tolerance: 0.02 mm is 2 % of the smallest hole radius, so the mesh is
# not the limiting resolution in the measurement below -- the voxel pitch is.
TESS_LINEAR_MM = 0.02
TESS_ANGULAR_RAD = 0.05


def hole_centres_mm():
    """X positions of the hole centres, one per radius, with GAP_MM of material between the walls."""
    xs = []
    x = 0.0
    for k, r in enumerate(HOLE_RADII_MM):
        if k > 0:
            x += HOLE_RADII_MM[k - 1] + GAP_MM + r
        xs.append(x)
    return xs


def plate_size_mm():
    """(length, width) of the plate, from the hole layout plus GAP_MM of material all round."""
    xs = hole_centres_mm()
    rmax = max(HOLE_RADII_MM)
    length = (xs[-1] + HOLE_RADII_MM[-1] + GAP_MM) - (xs[0] - HOLE_RADII_MM[0] - GAP_MM)
    width = 2.0 * (rmax + GAP_MM)
    return length, width


def build():
    """Builds the plate with its five through holes and returns the build123d solid."""
    from build123d import Axis, Box, Cylinder, Location, Mode, Part, Vector

    length, width = plate_size_mm()
    xs = hole_centres_mm()
    x_mid = 0.5 * (xs[0] - HOLE_RADII_MM[0] - GAP_MM + xs[-1] + HOLE_RADII_MM[-1] + GAP_MM)
    part = Box(length, width, THICKNESS_MM)
    part = Part() + part.moved(Location(Vector(x_mid, 0.0, 0.0)))
    for x, r in zip(xs, HOLE_RADII_MM):
        bore = Cylinder(r, THICKNESS_MM * 3.0).moved(Location(Vector(x, 0.0, 0.0)))
        part = Part(part.solids()) - bore
    _ = Axis, Mode
    return part


def write_stl(path=STL_PATH):
    """Tessellates the solid and writes it as a binary STL; returns the path."""
    from build123d import export_stl

    part = build()
    export_stl(part, path, tolerance=TESS_LINEAR_MM, angular_tolerance=TESS_ANGULAR_RAD)
    return path


def mesh(path=STL_PATH, regenerate=False):
    """Returns (V, T) for the part, regenerating the STL when it is missing or asked for."""
    import trimesh

    if regenerate or not os.path.exists(path):
        write_stl(path)
    m = trimesh.load(path, process=False)
    m.merge_vertices()
    V = np.asarray(m.vertices, dtype=np.float64)
    T = np.asarray(m.faces, dtype=np.int64)
    return V, T


def brep_hole_volume_mm3(r):
    """Closed-form volume of one through hole of radius r in this plate."""
    return float(np.pi * r * r * THICKNESS_MM)


if __name__ == "__main__":
    import trimesh

    p = write_stl()
    m = trimesh.load(p, process=False)
    m.merge_vertices()
    length, width = plate_size_mm()
    solid_mm3 = length * width * THICKNESS_MM - sum(brep_hole_volume_mm3(r) for r in HOLE_RADII_MM)
    print(f"plate {length:.1f} x {width:.1f} x {THICKNESS_MM:.1f} mm, holes "
          f"{', '.join('R%.0f' % r for r in HOLE_RADII_MM)}")
    print(f"stl -> {os.path.basename(p)}  triangles={len(m.faces)}  watertight={m.is_watertight}")
    print(f"volume mesh={m.volume:.3f} mm3  closed form={solid_mm3:.3f} mm3  "
          f"rel={abs(m.volume - solid_mm3) / solid_mm3:.2e}")
