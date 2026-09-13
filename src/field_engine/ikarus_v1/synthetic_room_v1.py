#!/usr/bin/env python3
"""Synthetic room and floor route: the input the Lipschitz-spine verification cells run on.

Everything here is declared in code and generated with numpy; no external asset is read.

THE ROOM (all dimensions declared below, millimetres): a rectangular room whose interior is
ROOM_MM = 200 x 150 x 100, enclosed by walls WALL_T = 8 thick. Inside it stands ONE partition made
of TWO internal wall segments at x = PART_X, leaving a DOORWAY of DOOR_W = 20 between them, centred
on the room's y axis. The route the duct follows runs along the floor at height SPINE_Z, straight
through that doorway, so the doorway -- and nothing else -- is what forces the Lipschitz shrink.

THE FIELDS written to <out_dir>:
  falt_rum.npz   d_free  (nx,ny,nz) float32   mm-true clearance to the nearest occupied cell,
                                              a Euclidean distance transform of the occupancy
                 origin  (3,) float           the grid's lower corner in mm
                 pitch   scalar float         the grid pitch in mm
  spine_v1.npz   C  (N,3)  spine stations in mm
                 R  (N,)   a REFERENCE radius profile: the same Lipschitz cap, but smoothed with a
                           7-station box filter and clipped to R_MAX, i.e. produced by a different
                           path than the primitive's own per-station clamp, so the verification
                           cell can compare the two without the comparison being a tautology
                 T  (N,3)  unit tangents

Run: python synthetic_room_v1.py [out_dir]
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import ndimage as ndi

# ---------------------------------------------------------------- declared geometry, mm
ROOM_MM = (200.0, 150.0, 100.0)   # interior extent (x, y, z)
WALL_T = 8.0                      # thickness of the enclosing walls and of the partition
PART_X = 100.0                    # partition plane, x coordinate of its near face
DOOR_W = 20.0                     # doorway width in y, centred on the room's y axis
GRID_PITCH = 1.0                  # clearance-field grid pitch
SPINE_Z = 30.0                    # route height above the interior floor
SPINE_X0, SPINE_X1 = 20.0, 180.0  # route start and end, x
SPINE_SPACING = 3.0               # distance between spine stations
R_MAX = 12.0                      # design cap the reference profile is clipped to
REF_MARGIN = 3.5                  # wall + safety the reference profile subtracts (2.5 + 1.0)


def build_occupancy():
    """Returns (occ, origin, pitch): a boolean occupancy grid of the room's solid material."""
    lo = np.array([-WALL_T, -WALL_T, -WALL_T])
    hi = np.array([ROOM_MM[0] + WALL_T, ROOM_MM[1] + WALL_T, ROOM_MM[2] + WALL_T])
    ns = np.ceil((hi - lo) / GRID_PITCH).astype(int) + 1
    xs = lo[0] + GRID_PITCH * np.arange(ns[0])
    ys = lo[1] + GRID_PITCH * np.arange(ns[1])
    zs = lo[2] + GRID_PITCH * np.arange(ns[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    inside = ((X >= 0.0) & (X <= ROOM_MM[0])
              & (Y >= 0.0) & (Y <= ROOM_MM[1])
              & (Z >= 0.0) & (Z <= ROOM_MM[2]))
    occ = ~inside                                   # the enclosing walls are everything outside
    # the two internal wall segments: the partition slab minus the doorway
    slab = (X >= PART_X) & (X <= PART_X + WALL_T)
    door = (Y >= (ROOM_MM[1] - DOOR_W) / 2.0) & (Y <= (ROOM_MM[1] + DOOR_W) / 2.0)
    occ |= slab & inside & ~door
    return occ, lo, GRID_PITCH


def build_d_free():
    """Returns (d_free, origin, pitch): mm-true clearance to the nearest occupied cell."""
    occ, origin, pitch = build_occupancy()
    d = ndi.distance_transform_edt(~occ, sampling=(pitch, pitch, pitch))
    return d.astype(np.float32), origin, pitch


def build_spine(d_free, origin, pitch):
    """Returns (C, R, T): the floor route through the doorway, a reference radius profile and
    unit tangents."""
    n = int(np.floor((SPINE_X1 - SPINE_X0) / SPINE_SPACING)) + 1
    xs = SPINE_X0 + SPINE_SPACING * np.arange(n)
    C = np.stack([xs, np.full(n, ROOM_MM[1] / 2.0), np.full(n, SPINE_Z)], axis=1)
    T = np.tile(np.array([[1.0, 0.0, 0.0]]), (n, 1))
    idx = np.round((C - np.asarray(origin)) / pitch).astype(int)
    d_at = d_free[idx[:, 0], idx[:, 1], idx[:, 2]].astype(float)
    R = np.clip(d_at - REF_MARGIN, 0.0, R_MAX)
    R = ndi.uniform_filter1d(R, size=7, mode="nearest")
    R = np.minimum(R, np.clip(d_at - REF_MARGIN, 0.0, R_MAX))
    return C, R, T


def write(out_dir):
    """Writes falt_rum.npz and spine_v1.npz into out_dir and returns the directory."""
    os.makedirs(out_dir, exist_ok=True)
    d_free, origin, pitch = build_d_free()
    C, R, T = build_spine(d_free, origin, pitch)
    np.savez(os.path.join(out_dir, "falt_rum.npz"), d_free=d_free,
             origin=np.asarray(origin, dtype=float), pitch=float(pitch))
    np.savez(os.path.join(out_dir, "spine_v1.npz"), C=C, R=R, T=T)
    print("ROOM interior %s mm, wall %.1f mm, partition at x=%.1f with a %.1f mm doorway; "
          "grid %s at pitch %.2f mm" % (list(ROOM_MM), WALL_T, PART_X, DOOR_W,
                                        list(d_free.shape), pitch))
    print("SPINE n=%d stations, z=%.1f mm, d_free=[%.2f,%.2f] mm, reference R=[%.2f,%.2f] mm"
          % (len(C), SPINE_Z, float(d_free.min()), float(d_free.max()),
             float(R.min()), float(R.max())))
    print("WROTE %s" % out_dir)
    return out_dir


if __name__ == "__main__":
    write(sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "synthetic_room_v1"))
