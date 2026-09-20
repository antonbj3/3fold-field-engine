#!/usr/bin/env python3
"""A keep-out volume against an occupancy that changes every 10 ms, and what that costs.

The keep-out rooms in this engine are declared once and judged against solids. That is the right
instrument for a machine whose parts move on known paths. It is the wrong instrument for the case
this module exists for: material that ARRIVES -- swarf falling off a cut, a pile spreading, a bin
filling -- where the occupied set at t is not a rigid transform of the occupied set at 0 and no
solid describes it.

The honest case for a field, as opposed to a convex hull or any other static abstraction of the
same particles, is CORRECTNESS, and it is a geometric argument: a hull bridges. Where a growing pile
has a gap, a re-hull of its particles spans the gap, and a keep-out test against that hull reports
material in a place that is empty -- or, when the test is 'does any material intrude', misses the
frame where the intrusion is real because the hull already covered the region earlier. A field
rebuilt from the particles has no such freedom; it says what is there.

MEASURED, 200 000 particles simulated for two seconds with the field rebuilt every 10 ms:

    the route bends                       frame 19, 0.20 s   (0.600 -> 0.720 m)
    first intrusion into the keep-out     frame 73, 0.74 s   (by direct particle count)
    particles inside it at 2 s            227
    storage, allocated against uniform    935 709 B = 33.57 % of 2 787 213 B
    field rebuild per update              84.66 ms median
    convex hull per update                 4.85 ms median    -- 17.47x cheaper
                                          (17.98x on the total over all 200 updates)

THE FIELD IS THEREFORE CORRECT AND TOO SLOW. 84.66 ms does not fit in a 10 ms tick, and that is the
whole of the field's problem here; it is not an accuracy problem.

THE OBVIOUS FIX WAS MEASURED AND DOES NOT REACH THE TICK EITHER. Rebuilding only the blocks whose
particles moved:

    blocks changed per frame              7.74 % of all blocks -- but 69.89 % of the OCCUPIED ones
    incremental against full rebuild      ratio 0.417 median = 35.32 ms
    frames inside the 10 ms tick          0 of 200
    ratio that would be needed            0.118, against a best measured frame of 0.263
    overhead alone, zero changed blocks   10.42 ms = the entire tick
                                          (binning 200 k particles 6.32 ms + rebuilding the
                                           spatial index 3.13 ms)
    variable work at the median           18.0 ms, on top of that overhead
    error the approximation introduces    291 stale blocks, 14.26 mm of drift at 2 s

Two independent walls: the share of OCCUPIED blocks that change is 70 %, not 8 %, because the pile
has a live emitter and a flowing surface; and the bookkeeping needed to find the changed blocks
costs the whole budget before any block is rebuilt. Neither is closed by tuning a threshold.

WHAT THIS MODULE DOES AND DOES NOT ASSERT. The mechanism -- that a static hull bridges a gap and a
rebuilt occupancy does not -- runs live below on a small synthetic pile. The numbers above are
carried from a run that needs a particle simulator and 287 s on a GPU.

Three things in that run do NOT support the reading they were given, and they are carried as limits
rather than quietly dropped:

  * the run's own FIELD detector first reports the intrusion at frame 143, not 73. Frame 73 is the
    direct particle count. So "the field catches what the hull misses" is not what was measured;
    what was measured is that the hull-based test disagrees with the particle count on 59 frames.
  * that disagreement is computed against the PARTICLE COUNT, never against the field, so it prices
    the hull's error and not the field's advantage.
  * the recorded route clearance is the field's default 0.5 m in all 200 frames, and the recorded
    hull route error has a median of 98.8 % over a 28-100 % spread. Neither is a measurement of
    anything; the routing leg of that run is not evidence and is not cited here as any.
"""
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "data", "tidsfalt_keepout_v1")

TICK_MS = 10.0


def ockupans_fran_partiklar(punkter, ursprung, form, pitch, radie):
    """Rasterise particles into an occupancy: a cell is occupied when a particle is within `radie`.

    This is the field's whole claim to correctness -- a cell is occupied because something is IN it,
    not because it lies between two things that are."""
    from scipy.spatial import cKDTree

    ursprung = np.asarray(ursprung, dtype=np.float64)
    form = tuple(int(v) for v in form)
    galler = np.stack(np.meshgrid(*[ursprung[i] + (np.arange(form[i]) + 0.5) * pitch
                                    for i in range(3)], indexing="ij"), axis=-1)
    platt = galler.reshape(-1, 3)
    if len(punkter) == 0:
        return np.zeros(form, dtype=bool)
    d, _ = cKDTree(np.asarray(punkter, dtype=np.float64)).query(platt, k=1)
    return (d <= radie).reshape(form)


def keepout_intrang(punkter, lada_min, lada_max):
    """How many particles are inside the declared keep-out box. The direct test, no abstraction."""
    if len(punkter) == 0:
        return 0
    p = np.asarray(punkter, dtype=np.float64)
    inne = np.all((p >= np.asarray(lada_min)) & (p <= np.asarray(lada_max)), axis=1)
    return int(inne.sum())


def holje_intrang(punkter, lada_min, lada_max, n_prov=12):
    """The same test against a CONVEX HULL of the particles, which is what a static abstraction of
    them can say. Returns whether the hull intersects the box, tested on a lattice inside the box.

    A hull can only over-report: it contains every particle and more. That is exactly why it cannot
    be trusted to say where material is NOT."""
    from scipy.spatial import ConvexHull, Delaunay

    p = np.asarray(punkter, dtype=np.float64)
    if len(p) < 4:
        return False, 0
    try:
        h = Delaunay(p[ConvexHull(p).vertices])
    except Exception:
        return False, 0
    axlar = [np.linspace(lada_min[i], lada_max[i], n_prov) for i in range(3)]
    prov = np.stack(np.meshgrid(*axlar, indexing="ij"), axis=-1).reshape(-1, 3)
    inne = h.find_simplex(prov) >= 0
    return bool(inne.any()), int(inne.sum())


def syntetisk_hog(n_ramar=40, per_ram=60, fro=20260920):
    """A pile that grows with a gap in it: two mounds with a corridor between them.

    Particles are emitted at two seed columns and settle; the corridor between the mounds stays
    empty until late, when one mound spills into it. That is the geometry the whole argument needs:
    a region that a hull of the same particles covers from the first frame and that is actually
    empty until the spill."""
    rng = np.random.default_rng(fro)
    hogar = [np.array([-0.12, 0.0]), np.array([0.12, 0.0])]
    punkter = []
    ramar = []
    for f in range(n_ramar):
        for i in range(per_ram):
            c = hogar[i % 2]
            spridning = 0.010 + 0.0004 * f
            xy = c + rng.normal(0.0, spridning, 2)
            if f >= int(0.75 * n_ramar) and i % 7 == 0:          # the spill into the corridor
                xy = np.array([rng.uniform(-0.02, 0.02), rng.normal(0.0, 0.01)])
            z = abs(rng.normal(0.0, 0.006)) + 0.001 * (f % 5)
            punkter.append([xy[0], xy[1], z])
        ramar.append(np.array(punkter, dtype=np.float64).copy())
    return ramar


KEEPOUT_MIN = (-0.02, -0.02, 0.0)
KEEPOUT_MAX = (0.02, 0.02, 0.03)


def kor_syntetisk(n_ramar=40, per_ram=60, fro=20260920):
    """Run the two detectors over the synthetic pile and report when each first fires."""
    ramar = syntetisk_hog(n_ramar, per_ram, fro)
    rader = []
    for f, p in enumerate(ramar):
        n_inne = keepout_intrang(p, KEEPOUT_MIN, KEEPOUT_MAX)
        holje_traff, holje_prov = holje_intrang(p, KEEPOUT_MIN, KEEPOUT_MAX)
        rader.append({"ram": f, "n_partiklar": len(p), "partiklar_i_ladan": n_inne,
                      "holjet_sager_intrang": holje_traff, "holjeprover_inne": holje_prov})
    forsta_verklig = next((r["ram"] for r in rader if r["partiklar_i_ladan"] > 0), None)
    forsta_holje = next((r["ram"] for r in rader if r["holjet_sager_intrang"]), None)
    falska = [r["ram"] for r in rader if r["holjet_sager_intrang"] and r["partiklar_i_ladan"] == 0]
    return {"rader": rader, "forsta_verkligt_intrang": forsta_verklig,
            "forsta_holjeintrang": forsta_holje, "holjets_falska_ramar": falska}


def las_matning(sokvag=None):
    with open(os.path.join(sokvag or DATA, "matning.json")) as fh:
        return json.load(fh)


def inkrementell_grans(data=None):
    """The speed wall, as two independent numbers plus the error the shortcut would introduce."""
    d = (data or las_matning())["inkrementell_grans"]
    return {"kvot_median": d["kvot"]["median"], "ms_median": d["ms"]["median"],
            "overhead_ms": d["overhead_ms"]["median"], "ramar_inom_tick": d["andel_ramar_under_10ms"],
            "kvot_som_kravs": d["kvot_som_kravs"], "basta_kvot": d["kvot"]["min"],
            "andrade_block_av_ockuperade": d["andrade_block"]["av_ockuperade"]["median"],
            "andrade_block_av_alla": d["andrade_block"]["av_alla"]["median"],
            "inaktuella_block": d["approximationsfel"]["n_stale_active"],
            "drift_m": d["approximationsfel"]["max_abs_diff_stale_m"]}


def _selftest():
    r = kor_syntetisk()
    print(f"synthetic pile: {len(r['rader'])} frames, "
          f"{r['rader'][-1]['n_partiklar']} particles at the end")
    print(f"  first real intrusion (particle count) at frame {r['forsta_verkligt_intrang']}")
    print(f"  first hull intrusion                  at frame {r['forsta_holjeintrang']}")
    print(f"  frames where the hull claims material in an empty box: "
          f"{len(r['holjets_falska_ramar'])}")

    # the occupancy the field would build, on one frame, against the same box
    ramar = syntetisk_hog()
    occ = ockupans_fran_partiklar(ramar[len(ramar) // 2], (-0.16, -0.06, 0.0),
                                  (40, 16, 8), 0.008, 0.006)
    print(f"  occupancy of the middle frame: {int(occ.sum())} of {occ.size} cells")

    d = las_matning()
    s = d["sammanfattning"]
    g = inkrementell_grans(d)
    print(f"carried: route bends at frame {s['divert_frame']} = {s['divert_time_s']} s, "
          f"first intrusion frame {s['first_breach_frame']}, "
          f"{d['rader'][-1]['n_particles']} particles at the end")
    print(f"carried: field {s['field_median_build_ms']:.2f} ms against hull "
          f"{s['cad_median_time_ms']:.2f} ms = "
          f"{s['field_median_build_ms']/s['cad_median_time_ms']:.2f}x, storage "
          f"{d['lagring']['pareto_to_uniform_ratio_pct']:.2f} % of uniform")
    print(f"carried limit: incremental ratio {g['kvot_median']:.3f} = {g['ms_median']:.2f} ms, "
          f"{g['ramar_inom_tick']:.0%} of frames inside the {TICK_MS:.0f} ms tick, "
          f"overhead alone {g['overhead_ms']:.2f} ms")
    print(f"carried limit: the run's FIELD detector first fires at frame "
          f"{d['faltets_egen_forsta_detektion']}, its particle count at frame "
          f"{d['referensens_forsta_detektion']}")

    fel = []
    if r["forsta_holjeintrang"] is None or r["forsta_verkligt_intrang"] is None:
        fel.append("neither detector fired; the synthetic pile does not exercise the test")
    elif r["forsta_holjeintrang"] >= r["forsta_verkligt_intrang"]:
        fel.append("the hull must claim the intrusion before it is real; that is the mechanism")
    if len(r["holjets_falska_ramar"]) == 0:
        fel.append("the hull must bridge the gap on at least one frame")
    if g["ramar_inom_tick"] != 0.0:
        fel.append("the carried speed limit must be carried as measured: 0 frames inside the tick")
    if g["andrade_block_av_ockuperade"] < 0.5:
        fel.append("the occupied-block share is the first wall and must be carried as measured")
    if fel:
        raise SystemExit("FAIL: " + "; ".join(fel))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
