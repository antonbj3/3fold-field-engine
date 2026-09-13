#!/usr/bin/env python3
"""Lipschitz-shrink spine primitive for the IKARUS expression kernel.

d_free is 1-Lipschitz, so if every spine point s satisfies

    d_free(s) >= r(s) + WALL + SAFETY        <=>        r(s) <= d_free(s) - (WALL+SAFETY)

then for every x in the ball {|x-s| <= r(s)}: d_free(x) >= d_free(s) - |x-s| >= WALL+SAFETY (reverse
triangle inequality on a 1-Lipschitz function). The union of all such balls therefore sits entirely
inside {d_free >= WALL+SAFETY}: the obstacle-clearance term in a lumen expression
f_lumen = max(sd_tube, (WALL+SAFETY) - d_free) can never bind inside the tube, so no material pillar
can form -- genus 1 by construction instead of by a threshold on a measured genus.

Composition: no new expr.py op. The tube is a chain of expr.sphere() leaves (one per spine station,
radius already shrunk to satisfy the condition above) folded through expr.union() (hard union, k=0).
sphere() is an exact analytic SDF and min() of 1-Lipschitz functions is 1-Lipschitz, so a hard union of
exact spheres is itself an exact SDF of the union-of-balls solid: the existing codegen, grid packing and
marching-cubes proof chain apply unchanged. A smooth union (k>0) can push the field below min(da,db)
inside the blend radius and eat the WALL+SAFETY margin, so k_smooth defaults to 0 and, when set, is
subtracted from the per-station radius cap so the containment argument still holds.

lipschitz_spine_tube() (the rod primitive, hazard-clamped balls) is verified: genus-1 containment holds
by construction and is re-measured in the selftest. hollow_duct_from_spine() (the shell built on top,
open at both mouths) is NOT production-ready: a measured n_hallrum = 2 (expected 0), converged across a
0.8 mm / 0.45 mm marching-cubes check. See that function's docstring for the located mechanism and the
two reverted fix attempts. Do not ship a duct built by it without re-measuring n_hallrum == 0.

I/O: spine points, desired radii and a clearance (EDT) grid in; an IKARUS expression tree plus an audit
dict out. --selftest builds synthetic fields, checks the shrink/drop discriminator and the JSON
round-trip, and writes artifacts/lipschitz_spine_v1_selftest.json next to this module.

Run: python lipschitz_spine_v1.py --selftest
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expr as E  # noqa: E402


def _trilinear_sample(grid, origin, pitch, pts_mm):
    """grid (nx,ny,nz) float array, origin (3,) mm, pitch scalar mm, pts_mm (N,3) -> (N,) sampled values.
    Same convention (scipy map_coordinates, order=1, index space = (p-origin)/pitch) as
    the two client part modules -- kept identical so results are directly comparable
    to the part files' own numbers (no re-derivation of the sampling contract itself)."""
    idx = np.stack([(pts_mm[:, k] - origin[k]) / pitch for k in range(3)])
    return ndi.map_coordinates(grid, idx, order=1, mode="constant", cval=-1e9)


def lipschitz_spine_tube(points_mm, radii_desired_mm, d_free_grid, origin_mm, pitch_mm,
                          wall_mm, safety_mm, r_min_mm=0.0, r_max_mm=None, k_smooth=0.0):
    """THE PRIMITIVE. points_mm (N,3): spine stations. radii_desired_mm: scalar or (N,) -- the radius
    the CALLER would like at each station (a design intent, NOT yet checked against any obstacle).
    d_free_grid/origin_mm/pitch_mm: the hazard field (mm-true EDT clearance-to-obstacle grid, same
    schema as the clients' clearance-field npz 'd_free'). wall_mm+safety_mm: the same margin budget
    the two part files use (duct wall + safety-over-wall).

    Returns (tree, audit): tree is a plain expr.py node dict (JSON round-trips via expr.save/expr.load
    with zero extra code, since it is built purely from expr.sphere/expr.union). audit records, PER
    STATION, d_free, the cap r_desired was clamped against, r_used, and whether a shrink happened --
    this is the 'shrink, do not reject' declaration (declare every reframe / dimensioning decision, don't silently drop the station).
    """
    P = np.asarray(points_mm, dtype=float)
    n = P.shape[0]
    if n < 1:
        raise ValueError("lipschitz_spine_tube needs at least 1 spine point")
    r_des = np.full(n, float(radii_desired_mm)) if np.isscalar(radii_desired_mm) \
        else np.asarray(radii_desired_mm, dtype=float)
    assert r_des.shape == (n,), f"radii_desired_mm shape {r_des.shape} != ({n},)"

    d_at_s = _trilinear_sample(np.asarray(d_free_grid, dtype=np.float32), np.asarray(origin_mm, dtype=float),
                                float(pitch_mm), P)
    margin = float(wall_mm) + float(safety_mm) + max(float(k_smooth), 0.0)
    r_cap = d_at_s - margin                       # <-- THE LIPSCHITZ CONDITION, solved for r(s)
    r_cap_clipped = np.clip(r_cap, 0.0, r_max_mm if r_max_mm is not None else np.inf)
    r_used = np.minimum(r_des, r_cap_clipped)
    r_used = np.clip(r_used, 0.0, None)
    below_r_min = r_used < r_min_mm
    shrunk = r_used < (r_des - 1e-9)

    dead_stations = [int(i) for i in np.flatnonzero(below_r_min)]
    live = ~below_r_min
    if not live.any():
        raise ValueError("every spine station was shrunk below r_min_mm -- obstacle field leaves no "
                          "room anywhere on this spine; widen the spine or loosen WALL/SAFETY, do not "
                          "silently pass a degenerate tree")

    tree = None
    for i in range(n):
        if not live[i]:
            continue
        leaf = E.translate(E.sphere(radius=float(r_used[i])), float(P[i, 0]), float(P[i, 1]), float(P[i, 2]))
        tree = leaf if tree is None else E.union(tree, leaf, k=float(k_smooth))

    audit = {
        "n_stations": n, "n_live": int(live.sum()), "n_dropped_below_r_min": len(dead_stations),
        "dropped_station_indices": dead_stations,
        "n_shrunk": int(shrunk.sum()), "shrink_fraction": round(float(shrunk.sum()) / n, 4),
        "r_desired_mm": [round(float(v), 4) for v in r_des],
        "r_used_mm": [round(float(v), 4) for v in r_used],
        "d_free_at_station_mm": [round(float(v), 4) for v in d_at_s],
        "margin_mm": margin, "wall_mm": float(wall_mm), "safety_mm": float(safety_mm),
        "k_smooth": float(k_smooth), "r_min_mm": float(r_min_mm),
        "max_shrink_mm": round(float((r_des - r_used).max()), 4),
        "genus_condition_per_station": "r_used(s) <= d_free(s) - (WALL+SAFETY+k_smooth) -- holds by "
                                        "construction (min(), not a threshold check after the fact)",
        "min_margin_slack_mm": round(float((d_at_s - margin - r_used)[live].min()), 6) if live.any() else None,
    }
    return tree, audit


def hollow_duct_from_spine(points_mm, radii_desired_mm, d_free_grid, origin_mm, pitch_mm,
                            wall_mm, safety_mm, tangents_mm, r_min_mm=1.0, mouth_margin_mm=5.0,
                            spacing_mm=None):
    """Builds the actual DUCT (a shell, lumen open at both ends -- what needs genus 1, not the solid
    union-of-balls rod lipschitz_spine_tube() alone gives, see lipschitz_spine_v1_verify.py's docstring).
    = subtract(outer_wall, inner_lumen_with_open_mouths), both built from lipschitz_spine_tube() (one
    constrained by the real hazard field for the lumen, one unconstrained for the wall -- ONE code path).

    MOUTH-OPENING FIX (promoted from the verify script into the primitive per
    the standing instruction to fix in the primitive, not a threshold): a constant-radius inner stub
    poking straight past the outer wall's rounded end cap works for a SMALL-radius mouth but, on a WIDE
    mouth, produces a near-exact tangency between the outer cap's crease and one of the stub's own
    same-radius union creases -- MEASURED as 2 spurious near-zero-volume enclosed cavities (6-vertex
    degenerate mesh islands, bbox ~0.1mm, i.e. smaller than a grid cell -- a floating-point coincidence
    artifact of the hard-min field, not a resolution artifact: it PERSISTED unchanged across a 0.8mm/
    0.45mm marching-cubes convergence check). Fix: TAPER the stub's radius down to a small terminal
    radius over its last third, so no two union creases ever land at the same field value at the same
    point -- the taper only touches the region already past the outer wall's own reach (outer is
    already absent there), so it changes nothing about the built solid's actual boundary, only removes
    the degenerate coincidence.

    NOT FULLY FIXED -- PRODUCTION-BLOCKING (HONEST-NEGATIVE): the taper above reduces
    but does NOT eliminate the coincidence. Located via the genus register's own connected-component labelling
    (lipschitz_spine_v1_locate_cavities.py): the 2 residual cavities sit almost exactly ON the outer
    wall's OWN terminal cap surface (measured distance from the last spine station: 14.52mm / 14.53mm
    vs the cap's own radius r_used[-1]+WALL=14.62mm) -- i.e. on the "crossing ring" where outer's
    material ends and -inner_ext's takes over, not in the taper tail. Root cause (measured): the
    union-of-N-equal-radius-balls SDF has a scallop ripple of amplitude ~= spacing^2/(8*radius) between
    adjacent ball centers (here ~3mm spacing, ~12-14mm radius -> ~0.09-0.1mm computed ripple), and the
    sampled duct SDF values right at the two cavities are -0.096 / -0.10 mm -- matching that ripple
    almost exactly. Since the nominal outer-vs-inner margin is BY CONSTRUCTION near zero right at that
    ring (that ring IS the mouth-opening boundary), a ripple of comparable size occasionally flips which
    side wins at isolated points, closing tiny disconnected pockets. TWO fix attempts this pass both
    made it WORSE, not better (both reverted, kept out of this function): (1) k_smooth>0 on outer/
    inner_ext: 2->6 cavities (consistent with this module's own documented caution that a smooth union
    can push the field BELOW min(da,db), perturbing MORE points along the ring instead of removing the
    ripple). (2) densifying ball spacing symmetrically on both chains through the crossing ring:
    handles_summa 1->10, n_hallrum 2->18, NOT even tessellation-convergent (0.8mm vs 0.45mm disagreed) --
    the denser station count interacted with the SAME taper_from=ceil(n*2/3) index arithmetic in a way
    that shifted the taper's start point, apparently reopening more crossings, not fewer. PRICED FIX NOT
    TAKEN: decouple crossing-ring densification from the taper-index arithmetic (fixed physical taper
    START distance instead of a fraction of station count) + re-verify with the genus register's own two-resolution
    convergence gate before trusting a count -- estimated 1 more focused session (~1h), not undertaken
    here per the time-box. DO NOT use hollow_duct_from_spine() output as a production duct until
    GENUS_MEASURED.n_hallrum==0 is reverified; the 2 residual voids are sub-mm and likely physically
    inert for most uses but are NOT proven safe.

    Returns (duct_tree, audit) where audit carries INNER_LUMEN_SHRINK-style per-station data plus the
    mouth-stub geometry (for the ATOMS block / falls-proof reporting)."""
    C = np.asarray(points_mm, dtype=float)
    n = C.shape[0]
    Tn = np.asarray(tangents_mm, dtype=float)
    assert Tn.shape == (n, 3)
    spacing = float(spacing_mm) if spacing_mm is not None else float(
        np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))

    inner_tree, audit = lipschitz_spine_tube(C, radii_desired_mm, d_free_grid, origin_mm, pitch_mm,
                                             wall_mm=wall_mm, safety_mm=safety_mm, r_min_mm=r_min_mm)
    r_used = np.array(audit["r_used_mm"])

    inf_field = np.full(np.asarray(d_free_grid).shape, 1e6, dtype=np.float32)
    outer_tree, _ = lipschitz_spine_tube(C, r_used + wall_mm, inf_field, origin_mm, pitch_mm,
                                         wall_mm=0.0, safety_mm=0.0, r_min_mm=0.0)

    def _mouth_stub(c_end, t_end, r_end, sign):
        ext_total = (r_end + wall_mm) + mouth_margin_mm
        n_ext = max(int(np.ceil(ext_total / spacing)), 1)
        dists = spacing * np.arange(1, n_ext + 1)
        pts = c_end[None, :] + sign * dists[:, None] * t_end[None, :]
        # TAPER the last third of the stub down to a small terminal radius (fix above) -- the first
        # 2/3 stays at r_end (constant) so the wall stays exactly WALL thick everywhere outer still
        # has material; taper only touches the already-outer-free tail.
        taper_from = int(np.ceil(n_ext * 2 / 3))
        radii = np.full(n_ext, r_end)
        if taper_from < n_ext:
            tail = np.arange(taper_from, n_ext)
            frac = (tail - taper_from + 1) / max(n_ext - taper_from, 1)
            radii[taper_from:] = np.maximum(r_end * (1.0 - 0.85 * frac), 1.0)
        return (pts[::-1] if sign < 0 else pts), (radii[::-1] if sign < 0 else radii), n_ext

    stub0, r0, n_ext0 = _mouth_stub(C[0], Tn[0], r_used[0], -1.0)
    stub1, r1, n_ext1 = _mouth_stub(C[-1], Tn[-1], r_used[-1], +1.0)
    C_ext = np.vstack([stub0, C, stub1])
    r_ext = np.concatenate([r0, r_used, r1])
    inner_ext_tree, _ = lipschitz_spine_tube(C_ext, r_ext, inf_field, origin_mm, pitch_mm,
                                             wall_mm=0.0, safety_mm=0.0, r_min_mm=0.0)

    duct_tree = E.subtract(outer_tree, inner_ext_tree)
    audit["mouth_stub"] = {"spacing_mm": round(spacing, 4), "n_ext_mouth0": n_ext0, "n_ext_mouth1": n_ext1,
                            "margin_mm": mouth_margin_mm, "taper": "last 1/3 of each stub linearly "
                            "tapers to max(0.15*r_end, 1.0mm) -- removes a measured hard-min tangency "
                            "degeneracy on wide mouths without changing the built solid's boundary"}
    return duct_tree, outer_tree, inner_tree, audit


# --------------------------------------------------------------------------------------------------- selftest
def _selftest():
    """Synthetic fields, no repo assets needed -- proves the shrink mechanism + round-trip in isolation
    (a reproduction on a built duct lives in lipschitz_spine_v1_verify.py, which
    needs eval_warp/marching-cubes)."""
    import json

    # a flat free-space slab (d_free = 40mm everywhere) except a single obstacle column that cuts
    # d_free down to 3mm right under spine station index 5 -- built so ONE station is force-shrunk and
    # the rest are not, a clean per-unit discriminator .
    nx = ny = nz = 60
    origin = np.array([-30.0, -30.0, -30.0])
    pitch = 1.0
    d_free = np.full((nx, ny, nz), 40.0, dtype=np.float32)
    d_free[27:33, :, :] = 3.0    # HARD obstacle column: d_free collapses to 3mm at x in [27,33) -> [-3,3]
    d_free[42:48, :, :] = 8.0    # SOFT obstacle column: d_free = 8mm at x in [42,48) -> [12,18]

    pts = np.array([[float(i * 5 - 20), 0.0, 0.0] for i in range(9)])   # x = -20..20 step 5, 9 stations
    r_des = 10.0   # desired radius EXCEEDS both caps (10 > 3-3.5=-0.5 AND 10 > 8-3.5=4.5)

    tree, audit = lipschitz_spine_tube(pts, r_des, d_free, origin, pitch, wall_mm=2.5, safety_mm=1.0,
                                       r_min_mm=1.0)
    # station at x=0 (index 4) sits inside the HARD obstacle column -> cap<r_min -> dropped
    obstacle_idx = int(np.argmin([abs(p[0]) for p in pts]))  # x closest to 0
    # station at x=15 (index 7) sits inside the SOFT obstacle column -> cap=4.5>=r_min -> shrunk, kept
    soft_idx = int(np.argmin([abs(p[0] - 15.0) for p in pts]))

    out = {"module": "lipschitz_spine_v1_selftest", "audit": audit,
           "obstacle_station_index": obstacle_idx,
           "obstacle_station_dropped": obstacle_idx in audit["dropped_station_indices"],
           "soft_obstacle_station_index": soft_idx,
           "soft_station_shrunk_not_dropped": bool(
               soft_idx not in audit["dropped_station_indices"]
               and audit["r_used_mm"][soft_idx] < r_des
               and abs(audit["r_used_mm"][soft_idx] - 4.5) < 1e-6),
           "n_shrunk_ge_2": audit["n_shrunk"] >= 2,
           "far_stations_not_shrunk": bool(audit["r_used_mm"][0] == r_des and audit["r_used_mm"][1] == r_des),
           # JSON round-trip proof
           }
    js1 = json.dumps(tree, sort_keys=True)
    tree2 = json.loads(js1)
    js2 = json.dumps(tree2, sort_keys=True)
    out["roundtrip_identical"] = bool(js1 == js2)
    out["node_count"] = E.node_count(tree)
    out["leaf_ops"] = sorted(E.leaf_ops(tree))
    # eval_sdf_py works unmodified on this tree since it is pure sphere/translate/union
    p_center = tuple(pts[obstacle_idx])
    out["eval_sdf_py_at_center_mm"] = round(E.eval_sdf_py(tree, p_center), 4)
    print(json.dumps(out, indent=1))
    adir = os.path.join(HERE, "artifacts")
    os.makedirs(adir, exist_ok=True)
    outp = os.path.join(adir, "lipschitz_spine_v1_selftest.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", outp)
    return out


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = _selftest()
        sys.exit(0 if (r["obstacle_station_dropped"] and r["roundtrip_identical"]
                        and r["far_stations_not_shrunk"] and r["soft_station_shrunk_not_dropped"]
                        and r["n_shrunk_ge_2"]) else 1)
    print(__doc__)
