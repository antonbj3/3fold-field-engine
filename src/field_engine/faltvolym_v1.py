#!/usr/bin/env python3
"""Two meters for the volume of a signed field, and the pitch that decides which one can be right.

A block-sparse signed field has two obvious volume integrals and they are not the same quantity.

  VOXEL RIEMANN        count the cells whose sample is negative and multiply by pitch^3. A Riemann
                       sum with a first-order boundary error: every boundary cell is either wholly in
                       or wholly out, so the error is bounded by half a pitch times the surface area,
                       and it does not cancel -- a thin wall four voxels across carries a quarter of
                       its own volume in that boundary layer.
  MARCHING CUBES       take the zero isosurface and integrate the divergence over the closed mesh.
                       The isosurface interpolates linearly inside each cell, so the boundary is
                       resolved to a fraction of a pitch instead of to a whole one.

WHAT THE METER CHANGE IS WORTH, AND WHAT IT IS NOT. Measured over the 70-part bench catalogue, on
BYTE-IDENTICAL FIELDS -- the same `sdf_sha256` for both readings, so nothing about the engine changed:

    volume within 5 % of the exact solid, voxel Riemann        13 / 70
    volume within 5 % of the exact solid, marching cubes       17 / 70

Four parts gained net: five parts cross the band (thin_wall plus four parameter parts) and
ONE crosses back the other way. That one, `dpp_r1_51`, is the reason the first report's "0
regressions" was wrong: it passed on the voxel reading at 3.319 % and fails on the isosurface reading
at 8.531 %, and it stayed out of the regression watch because it was already failing a different gate
in the same family. A meter change that gains five and loses one is a meter change, not an engine
improvement, and the honest headline is 13 -> 17, not the +35 that a different pitch and an OR of both
meters produce together.

For the record, what that +35 was: taking OR(voxel, marching cubes) AND refining the pitch from the
CAD parameters reaches 48 / 70; marching cubes alone on the refined field reaches 46 / 70 and voxel
alone 47 / 70. So on the refined field the isosurface is the WORSE of the two meters, and essentially
all of the gain is the pitch.

THE PITCH: `feature_radius_min`. The sparse field's rasteriser can choose its own pitch from the
smallest resolvable feature in the mesh instead of taking the caller's. This module sets that choice
ON by default on the measurement path (`FEATURE_RADIUS_MIN_DEFAULT`), which is where the gain
actually lives, and it leaves the engine entry point's own default alone so that no existing caller's
numbers move. Measured on the 19 parts whose failure was a CSG wall thinner than half a pitch:

    volume within 5 %, requested pitch      marching cubes  3 / 19    voxel  1 / 19
    volume within 5 %, feature-chosen pitch marching cubes 15 / 19    voxel 15 / 19

with three things that must travel with it:

  * IT COSTS 88x. Field time over those 19 parts goes from 0.4489 s to 147.0041 s, a median of
    0.0242 s to 2.5925 s per part. A finer pitch is more voxels; there is no free resolution.
  * FOUR PARTS STILL FAIL, and two of them fail their own bound: the chosen pitch comes out larger
    than half the wall it was supposed to resolve.
  * IT CANNOT SEE A FLAT WALL. The feature measurement looks for smooth dihedral pairs and returns
    early when there are none, so on a 10 x 10 x 0.1 mm plate it changes nothing: same 1.0 mm pitch
    and the same field hash 69a4cb5953b713f8804b4be459449190f35b1b142aff5d5af4dedc7656a81860 with
    the choice on and off, while the exact volume is 10 mm^3 and the occupancy reads 0. On a
    polyhedron this default is inert, and a polyhedron is exactly where a thin wall is likeliest.

ORDER OF CONVERGENCE IS NOT ESTABLISHED. Halving the pitch makes the error WORSE on 14 of 70 parts
for the voxel meter and 13 of 70 for the isosurface, with a median fitted order of 1.51 and 1.88 and
a spread from -3.6 to +8.2. `thin_wall` goes from 4.76 % to 14.06 % under refinement. Until that is
explained the field construction itself is under suspicion, not just the meter, and no asymptotic
claim should be made from these numbers.
"""
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "faltvolym_v1")
KATALOG_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                            "data", "cadbank_v1")

# On the measurement path the pitch is chosen from the mesh, not taken from the caller. The engine
# entry point keeps its own default (off) so that existing callers' numbers do not move; see the
# module docstring for what this costs and where it is inert.
FEATURE_RADIUS_MIN_DEFAULT = "auto"

VOLYM_BAND = 0.05


def voxel_riemann_volym(falt, pitch):
    """Riemann sum over the occupancy: cells with a negative sample, times pitch^3."""
    return float(np.count_nonzero(np.asarray(falt) < 0.0) * float(pitch) ** 3)


def marching_cubes_volym(falt, pitch, origin=(0.0, 0.0, 0.0)):
    """Volume enclosed by the zero isosurface of the field.

    Returns 0.0 when the field has no zero crossing -- a window that is entirely solid or entirely
    empty has no isosurface, and reporting that as a volume would hide the failure. Degenerate
    zero-area facets from ambiguous cells are merged away before the integral, and when marching
    cubes produces several components the largest is taken; the number of discarded components is
    returned beside the volume so that a caller can refuse a part where the surface fell apart."""
    from skimage.measure import marching_cubes
    import trimesh

    f = np.asarray(falt, dtype=np.float64)
    if not (f.min() < 0.0 < f.max()):
        return 0.0, 0
    v, t, _, _ = marching_cubes(f, 0.0, spacing=(float(pitch),) * 3)
    m = trimesh.Trimesh(v + np.asarray(origin, dtype=np.float64), t, process=True)
    delar = m.split(only_watertight=False)
    if len(delar) > 1:
        m = max(delar, key=lambda c: abs(c.volume))
        return float(abs(m.volume)), len(delar) - 1
    return float(abs(m.volume)), 0


def las_matarbyte(sokvag=None):
    with open(os.path.join(sokvag or DATA, "matarbyte.json")) as fh:
        return json.load(fh)


def matarbytet(delar, band=VOLYM_BAND):
    """The meter change alone, on unchanged field bytes: how many parts each meter puts inside the
    band, and which parts cross in each direction."""
    def inom(v, b):
        return abs(v - b) / abs(b) <= band

    vox = {d["id"] for d in delar if inom(d["voxel_volume"], d["brep_volume"])}
    mc = {d["id"] for d in delar if inom(d["mc_volume"], d["brep_volume"])}
    return {"voxel": len(vox), "marching_cubes": len(mc), "n": len(delar),
            "vunna": sorted(mc - vox), "forlorade": sorted(vox - mc),
            "identiska_faltbytes": len({d["sdf_sha256"] for d in delar}) == len(delar)}


def finfunktionsutfall(data=None, band=VOLYM_BAND):
    """The feature-chosen pitch on the parts whose walls are thinner than half the requested pitch."""
    d = data or las_matarbyte()
    c2 = [r for r in d["delar"] if r["c2"]]

    def rakna(nyckel, matare):
        n = 0
        for r in c2:
            if nyckel == "original":
                v = r["voxel_volume"] if matare == "voxel" else r["mc_volume"]
            else:
                if not r["finfunktion_auto"]:
                    continue
                v = r["finfunktion_auto"][matare]
            n += int(abs(v - r["brep_volume"]) / abs(r["brep_volume"]) <= band)
        return n

    kost = d["kostnad_c2"]
    return {"n_c2": len(c2),
            "begard_pitch": {"voxel": rakna("original", "voxel"), "mc": rakna("original", "mc")},
            "vald_pitch": {"voxel": rakna("auto", "voxel"), "mc": rakna("auto", "mc")},
            "kostnadskvot_median": kost["median_ratio"],
            "sekunder_begard": kost["original"]["total_s"],
            "sekunder_vald": kost["patch_auto"]["total_s"],
            "kvar_rott": d["kvar_rott_c2"], "bundna_fel": d["bundna_fel"]}


def platt_vagg_ar_osynlig(data=None):
    """The counter-test: on a flat-walled plate the feature choice changes neither pitch nor field."""
    d = data or las_matarbyte()
    p = d["platt_lada_kontroll"]
    return {"dims_mm": p["dims"], "exakt_volym_mm3": p["occ_volume"],
            "pitch_av": p["off"]["pitch"], "pitch_auto": p["auto"]["pitch"],
            "faltet_oforandrat": p["off"]["sdf_sha256"] == p["auto"]["sdf_sha256"],
            "voxelvolym": p["off"]["voxel_volume"],
            "matt_feature_radie": p["feature_radius"]}


def ordning_ouppklarad(data=None):
    """How many parts get WORSE when the pitch is halved. Until this is zero, no order claim holds."""
    d = data or las_matarbyte()
    s = d["skalning"]
    return {"voxel_samre": s["voxel"]["worse"], "mc_samre": s["marching_cubes"]["worse"],
            "voxel_median_ordning": s["voxel"]["median_p"],
            "mc_median_ordning": s["marching_cubes"]["median_p"],
            "n": d["n"]}


def gitterfasprov(radie=5.0, pitch=1.0, fas=0.137, halvbredd=8.0):
    """Read both meters on an analytic sphere at two grid phases, and return the spread of each.

    This is the mechanism behind the 13 -> 17, isolated from any part: shifting the sample lattice by
    a fraction of a pitch changes NOTHING about the geometry, so any reading that moves is measuring
    the lattice. At radius 5 mm and a 1 mm pitch the Riemann sum reads 485.00 and 516.00 mm^3 at the
    two phases -- a 6 % swing across a shift that changed nothing -- while the isosurface reads
    510.99 and 511.14, a swing of 0.03 %. The exact volume is 523.60. Neither meter is accurate at
    this resolution; only one of them is STABLE, and that is the property the bench's volume family
    was failing on."""
    exakt = 4.0 / 3.0 * np.pi * radie ** 3
    vox, mc = [], []
    for off in (0.0, fas):
        ax = [np.arange(-halvbredd, halvbredd + pitch, pitch) + off for _ in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        falt = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - radie
        vox.append(voxel_riemann_volym(falt, pitch))
        mc.append(marching_cubes_volym(falt, pitch)[0])
    return {"exakt_mm3": exakt, "voxel": vox, "mc": mc,
            "voxel_spridning": abs(vox[0] - vox[1]) / exakt,
            "mc_spridning": abs(mc[0] - mc[1]) / exakt}


def _selftest():
    d = las_matarbyte()
    m = matarbytet(d["delar"])
    f = finfunktionsutfall(d)
    p = platt_vagg_ar_osynlig(d)
    o = ordning_ouppklarad(d)
    print(f"meter change on identical field bytes: voxel {m['voxel']}/{m['n']} -> "
          f"marching cubes {m['marching_cubes']}/{m['n']}")
    print(f"  gained {m['vunna']}")
    print(f"  lost   {m['forlorade']}")
    print(f"feature-chosen pitch on {f['n_c2']} thin-wall parts: "
          f"mc {f['begard_pitch']['mc']} -> {f['vald_pitch']['mc']}, "
          f"voxel {f['begard_pitch']['voxel']} -> {f['vald_pitch']['voxel']}, "
          f"cost x{f['kostnadskvot_median']:.1f}")
    print(f"flat plate: pitch {p['pitch_av']} -> {p['pitch_auto']}, field unchanged="
          f"{p['faltet_oforandrat']}, exact {p['exakt_volym_mm3']} mm3, occupancy {p['voxelvolym']}")
    print(f"halved pitch makes it worse on voxel {o['voxel_samre']}/{o['n']}, "
          f"marching cubes {o['mc_samre']}/{o['n']}")

    g = gitterfasprov()
    print(f"analytic sphere {g['exakt_mm3']:.2f} mm3 read at two grid phases:")
    print(f"  voxel Riemann  {g['voxel'][0]:.2f} / {g['voxel'][1]:.2f}  "
          f"spread {g['voxel_spridning']:.4%}")
    print(f"  marching cubes {g['mc'][0]:.2f} / {g['mc'][1]:.2f}  spread {g['mc_spridning']:.4%}")

    fel = []
    if not m["identiska_faltbytes"]:
        fel.append("the two meters must be read off the same field bytes")
    if g["mc_spridning"] >= g["voxel_spridning"] / 10.0:
        fel.append("the isosurface reading must be far less grid-phase dependent than the Riemann sum")
    if p["faltet_oforandrat"] is not True:
        fel.append("the flat-plate counter-test must show an unchanged field")
    if fel:
        raise SystemExit("FAIL: " + "; ".join(fel))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
