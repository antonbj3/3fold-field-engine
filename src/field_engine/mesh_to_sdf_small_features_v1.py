#!/usr/bin/env python3
"""Small-feature resolution of the mesh-to-SDF path, measured on a plate with five through holes.

The question: at which pitch does a through hole of radius R survive the block-sparse SDF, and what
does it look like when it does not. The part (examples/parts/holed_plate_v1.py) carries holes of
radius 1, 2, 4, 8 and 16 mm at one fixed thickness, so one pitch has to serve all five at once.

Three numbers per hole and pitch, all measured on the field reconstructed from the sparse blocks
(background blocks filled with their Lipschitz bound, active blocks with their exact tiles):
  * volym_fel -- the reconstructed hole volume against the closed-form pi*R^2*t. The hole volume is
    counted as the void voxels inside a cylinder of radius R + 5 mm (the material gap of the part)
    on the z layers that lie fully inside the plate, times the layer area and the plate thickness, so
    the measure does not depend on how many z layers the pitch happens to place inside the plate;
  * radiell_avvikelse -- the zero level set's radial error around the hole: the field is sampled at
    r = R at 64 angles at mid-thickness, where the true signed distance is 0, and the deviation is
    max |sd| / R. The same rays are also walked outwards to find where the field actually crosses
    zero, which measures the same thing without assuming |grad sd| = 1
    (radiell_avvikelse_nollgenomgang); the two agree once the sub-voxel surface correction is on;
  * en_ring -- whether the wall is still one closed ring: the sign of the field is read at the same
    64 angles at r = R - h and r = R + h. One ring means no sign change along either circle, with
    void inside and material outside. A wall broken into blobs flips sign several times. The test has
    no room when R - h <= R/4, which is reported as not measurable rather than as a pass.

Run `python mesh_to_sdf_small_features_v1.py --small-features` for the pitch sweep and the gate.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "parts"))

import faltkarna_v1_mesh_to_sdf as M2S  # noqa: E402
import holed_plate_v1 as PART           # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
UT_JSON = os.path.join(ARTIFACTS, "mesh_to_sdf_small_features.json")

PITCHAR_MM = (2.0, 1.0, 0.5, 0.25)
N_VINKLAR = 64
GATE_RADIE_MM = 2.0        # the gate applies to holes of at least this radius
GATE_AVVIKELSE = 0.05      # 5 % radial deviation of the zero level set


def tat_sd_fran_gles(sf, shape_l, block):
    """Rebuilds a dense signed field from the block kinds and tiles.

    Inactive blocks carry no values, only a sign, so they are filled with the classifier's own
    Lipschitz bound (+/- margin_mm): that is the strongest statement the sparse field makes about
    them. Active blocks are written from their exact tiles. Returns a float32 array of shape_l.
    """
    n_bx, n_by, n_bz = sf["n_bx"], sf["n_by"], sf["n_bz"]
    px, py, pz = n_bx * block, n_by * block, n_bz * block
    kind_grid = sf["kind"].reshape(n_bx, n_by, n_bz)
    bakgrund = np.where(kind_grid < 0, -sf["margin_mm"], sf["margin_mm"]).astype(np.float32)
    padded = np.empty((px, py, pz), dtype=np.float32)
    vy = padded.reshape(n_bx, block, n_by, block, n_bz, block).transpose(0, 2, 4, 1, 3, 5)
    vy[:] = bakgrund[:, :, :, None, None, None]
    if len(sf["active_ids"]) > 0:
        bi, bj, bk = np.unravel_index(sf["active_ids"], (n_bx, n_by, n_bz))
        vy[bi, bj, bk] = sf["tiles"]
    nx, ny, nz = shape_l
    return padded[:nx, :ny, :nz]


def _prov(sd, origin_l, pitch, punkter):
    """Trilinear sample of the dense field at world points; returns an array of mm values."""
    idx = (np.asarray(punkter, dtype=np.float64) - np.asarray(origin_l)) / pitch
    return ndimage.map_coordinates(sd, idx.T, order=1, mode="nearest")


def _teckenbyten(varden):
    """Sign changes around a closed circle of samples."""
    t = np.sign(varden)
    t[t == 0] = 1.0
    return int(np.count_nonzero(t != np.roll(t, 1)))


def mat_hal(sd, solid, origin_l, pitch, x_c, R, tjocklek):
    """The three measures for one hole; returns a dict."""
    vinklar = np.linspace(0.0, 2.0 * math.pi, N_VINKLAR, endpoint=False)
    cos, sin = np.cos(vinklar), np.sin(vinklar)

    def cirkel(r):
        return np.stack([x_c + r * cos, r * sin, np.zeros(N_VINKLAR)], axis=1)

    pa_vaggen = _prov(sd, origin_l, pitch, cirkel(R))
    avvikelse = float(np.max(np.abs(pa_vaggen)) / R)

    # where the field really crosses zero along each radial ray
    rr = np.linspace(max(R - 2.0 * pitch, 0.1 * R), R + 2.0 * pitch, 81)
    dr, n_traff = [], 0
    for a in vinklar:
        v = _prov(sd, origin_l, pitch, np.stack([x_c + rr * np.cos(a), rr * np.sin(a),
                                                 np.zeros_like(rr)], axis=1))
        k = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
        if len(k):
            n_traff += 1
            i = k[0]
            w = v[i] / (v[i] - v[i + 1])
            dr.append(abs(rr[i] + w * (rr[i + 1] - rr[i]) - R))
    avvikelse_noll = float(max(dr) / R) if dr else None

    r_inre = R - pitch
    ring_matbar = r_inre > 0.25 * R
    if ring_matbar:
        inre = _prov(sd, origin_l, pitch, cirkel(r_inre))
        yttre = _prov(sd, origin_l, pitch, cirkel(R + pitch))
        byten = _teckenbyten(inre) + _teckenbyten(yttre)
        en_ring = bool(byten == 0 and np.all(inre > 0.0) and np.all(yttre < 0.0))
    else:
        byten, en_ring = None, None

    # hole volume from the reconstructed occupancy
    nx, ny, nz = solid.shape
    gx = origin_l[0] + np.arange(nx) * pitch
    gy = origin_l[1] + np.arange(ny) * pitch
    gz = origin_l[2] + np.arange(nz) * pitch
    lager = np.abs(gz) <= (0.5 * tjocklek - pitch)
    ringm = (gx[:, None] - x_c) ** 2 + gy[None, :] ** 2 <= (R + 5.0) ** 2
    n_lager = int(np.count_nonzero(lager))
    if n_lager == 0:
        vol, fel = None, None
    else:
        tomt = (~solid[:, :, lager]) & ringm[:, :, None]
        area = float(np.count_nonzero(tomt)) * pitch ** 2 / n_lager
        vol = area * tjocklek
        facit = math.pi * R * R * tjocklek
        fel = (vol - facit) / facit
    return dict(R_mm=R, pitch_mm=pitch, R_per_h=R / pitch, radiell_avvikelse=avvikelse,
                radiell_avvikelse_nollgenomgang=avvikelse_noll, n_radier_med_nollgenomgang=n_traff,
                n_teckenbyten=byten, en_ring=en_ring, ring_matbar=bool(ring_matbar),
                vol_rekonstruerad_mm3=vol, vol_facit_mm3=math.pi * R * R * tjocklek,
                volym_fel=fel, n_lager=n_lager)


def kor_ett_fall(wp, device, V, T, pitch, feature_radius_min=None, ytkorrektion=None):
    """One rasterisation of the part; returns (per-hole rows, run info)."""
    lo = np.asarray(V, dtype=np.float64).min(axis=0) - 3.0 * pitch
    r = M2S.mesh_to_sdf_del(wp, V, T, pitch, 0.0, lo, device,
                            feature_radius_min=feature_radius_min, ytkorrektion=ytkorrektion,
                            returnera_falt=True)
    h = float(r["pitch_effektiv"])
    origin_l = lo + np.asarray(r["gmin"], dtype=np.float64) * h
    sd = tat_sd_fran_gles(r["sf"], r["shape_l"], r["block"])
    solid = r["solid_final"]
    rader = [mat_hal(sd, solid, origin_l, h, x, R, PART.THICKNESS_MM)
             for x, R in zip(PART.hole_centres_mm(), PART.HOLE_RADII_MM)]
    info = dict(pitch_begard=float(r["pitch_begard"]), pitch_effektiv=h,
                feature_radius_min=feature_radius_min,
                ytkorrektion=bool(r["signering"].get("ytkorrektion")), feature=r["feature"],
                shape_l=[int(x) for x in r["shape_l"]], n_blocks=int(r["n_blocks"]),
                n_active=int(r["n_active"]), vattentathet=r["vattentathet"]["status"],
                t_flood_s=r["t_flood_s"], t_gpu_classify_s=r["t_gpu_classify_s"])
    return rader, info


def _tabell(namn, fall):
    """Prints one hole-radius x pitch table; returns the printed text."""
    rader = [f"{namn}: radial deviation of the zero level set (|sd|/R), one ring, hole volume error"]
    pitchar = [f["info"]["pitch_effektiv"] for f in fall]
    rader.append("  R [mm] | " + " | ".join(f"h={h:g} mm".center(22) for h in pitchar))
    for k, R in enumerate(PART.HOLE_RADII_MM):
        celler = []
        for f in fall:
            d = f["rader"][k]
            ring = "-" if d["en_ring"] is None else ("1 ring" if d["en_ring"] else f"{d['n_teckenbyten']} flips")
            vf = "n/a" if d["volym_fel"] is None else f"{100 * d['volym_fel']:+.1f}%"
            celler.append(f"{100 * d['radiell_avvikelse']:6.1f}% {ring:>8s} {vf:>8s}")
        rader.append(f"  {R:6.0f} | " + " | ".join(celler))
    text = "\n".join(rader)
    print(text)
    return text


def kor(strikt=True):
    """The pitch sweep before and after the fix, the gate and the evidence JSON."""
    wp, have_cuda = M2S._import_warp()
    device = "cuda:0" if have_cuda else "cpu"
    V, T = PART.mesh()
    # BEFORE is the module as it stood: a fixed pitch and the raw distance transform, no sub-voxel
    # surface correction and no feature-driven pitch.
    fore = []
    for h in PITCHAR_MM:
        rader, info = kor_ett_fall(wp, device, V, T, h, ytkorrektion=False)
        fore.append(dict(rader=rader, info=info))
    efter = []
    rader, info = kor_ett_fall(wp, device, V, T, max(PITCHAR_MM), feature_radius_min="auto")
    efter.append(dict(rader=rader, info=info))

    t_fore = _tabell("BEFORE (fixed pitch, raw distance transform)", fore)
    print()
    t_efter = _tabell("AFTER (feature_radius_min='auto', sub-voxel surface correction)", efter)

    grind = []
    for d in efter[0]["rader"]:
        if d["R_mm"] >= GATE_RADIE_MM:
            grind.append(dict(R_mm=d["R_mm"],
                              pass_avvikelse=bool(d["radiell_avvikelse"] <= GATE_AVVIKELSE),
                              pass_ring=bool(d["en_ring"]),
                              radiell_avvikelse=d["radiell_avvikelse"]))
    ut = dict(part=dict(radier_mm=list(PART.HOLE_RADII_MM), tjocklek_mm=PART.THICKNESS_MM,
                        n_trianglar=int(len(T))),
              device=device, n_vinklar=N_VINKLAR,
              gate=dict(radie_min_mm=GATE_RADIE_MM, avvikelse_max=GATE_AVVIKELSE, rader=grind),
              fore=fore, efter=efter,
              R1_rad_efter=efter[0]["rader"][0],
              ALL_PASS=bool(all(g["pass_avvikelse"] and g["pass_ring"] for g in grind)))
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(UT_JSON, "w") as f:
        json.dump(ut, f, indent=1, default=float)
    print(f"\n-> wrote {os.path.relpath(UT_JSON, ROOT)}")
    d1 = efter[0]["rader"][0]
    print(f"R=1 mm row (reported, not gated): radial deviation {100 * d1['radiell_avvikelse']:.1f} %, "
          f"ring={d1['en_ring']}, volume error "
          f"{'n/a' if d1['volym_fel'] is None else '%+.1f %%' % (100 * d1['volym_fel'])}")
    print("ALL_PASS" if ut["ALL_PASS"] else "GATE FAILED")
    _ = t_fore, t_efter
    return ut


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-features", action="store_true", help="run the pitch sweep and the gate")
    a = ap.parse_args()
    res = kor()
    if not res["ALL_PASS"]:
        raise SystemExit("mesh_to_sdf small-feature gate FAILED")
