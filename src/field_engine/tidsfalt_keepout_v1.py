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

THE INCREMENTAL PATH AND ITS REMOVAL PASS. Rebuilding the occupancy every tick is what makes the
path slow; the alternative is to keep the owner channel and re-judge only the cells whose owner can
change. That set is the radius-expanded AABB of each body's old and new particle positions, plus --
and this is the part a naive update misses -- the cells a SHRUNKEN owner used to hold. Measured on
five scenes (a growing pile, three bodies on sinusoids, a body that jumps 3 cells/tick, a body
removed at tick 4, and a pile that shrinks 60 -> 120 -> 90 -> 60 -> 30 -> 0 particles):

    divergent ticks against the full rebuild    0 of 40 / 200 / 8 / 8 / 6
    cells the removal pass re-judges            vanish 234 (tick 4); shrink 138, 113, 81, 42
    removal pass cost when nothing is removed    0 cells on the pile and the three bodies
    update per tick, CPU                        pile 0.982 / 0.980 ms, three bodies 0.593 / 0.589 ms
                                                against 0.924 / 0.562 ms before the pass (+5-6 %)

The removal pass is exact under one stated assumption: that an owner's surviving particles are a
PREFIX of its previous cloud (a shrinking pile keeps frames[f][:n], a vanished body loses its whole
tail). A removal in the middle of an owner's cloud would need a general set difference; the prefix
assumption is written into `borttagna_per_agare`. The pass is CPU only; the GPU variant of this path
was measured to differ from the rebuild on exact distance ties and is not carried here.

THE INCREMENTAL PATH'S TIE-BREAK. The owner of a cell is the LOWEST particle id among those at the
smallest squared distance, and that distance is compared as an EXACT int64 in fixed point (1 quantum
= 1 nm), never as float equality or float ordering. scipy's cKDTree does not return bit-identical
distances for symmetric ties -- its summation order differs by 1 ulp -- so a float `==` would leave
the tie to the tree's structure and the path would stop being a function of the particles alone.
Measured on the five scenes above, the rule changes exactly 91 cells against the previous float
KDTree (jump_3cells 66, disappear_at_4 25, the other three 0), and every one of those cells is an
exact tie where at least two particles share the minimal int64 squared distance. The tree-and-ballot
engine and an independent chunked int64 brute force break all 91 the same way.

THE MASS-GRID FEED. A simulator's mass grid can be handed to the occupancy with the threshold in the
feed rather than in this module:

    occ = mass > theta * rho * dx**3        theta = 0.5 by default

Measured on the saved MPM bed against the highest particle per column: theta = 0.5 gives a p95
within one cell in all 50 frames (worst frame 0.4865 mm, cell 2.0833 mm); theta = 0.25 in 46 of 50;
theta = 0.05 in 8 of 50. The threshold is the feed's, not the field's: `ockupans_fran_massgitter`
takes the grid, the density and the pitch.
"""
import hashlib
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "data", "tidsfalt_keepout_v1")

TICK_MS = 10.0

# The owner channel's tie-break: coordinates in int64 fixed point, 1 quantum = 1 nm.
KVANTUM = 1_000_000_000
TIE_MARGIN = 1e-8
BALL_EPS = 1e-6
_INT64_MAX = np.int64(9_223_372_036_854_775_807)


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


def ockupans_fran_massgitter(massa, rho, dx, theta=0.5):
    """Occupancy of a simulator mass grid: `mass > theta * rho * dx**3` cell by cell.

    The threshold lives in the feed, not in the field: this function only applies it. theta is the
    share of a full cell mass (rho*dx**3), so the default 0.5 is half a cell mass, which is the MPM's
    own surface definition. The grid is indexed [ix, iy, iz] with node iy at height iy*dx.
    """
    troskel = float(theta) * (float(rho) * float(dx) ** 3)
    return np.asarray(massa, dtype=np.float64) > troskel


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


# ─────────────── owner-indexed occupancy, incrementally updated ───────────────
# The occupancy in this section carries an owner id per cell instead of a flag: owner(cell) is the
# id of the particle with the smallest squared distance d2 among particles with d2 <= radie**2,
# ties by particle index, -1 when there is none. `agare_full` is the rebuild reference (one KDTree
# over all particles, every cell queried); the incremental path keeps the previous channel and
# re-judges the cells where the owner can change.
#
# Distances are compared in int64 fixed point (1 quantum = 1 nm) for the decision that matters --
# which particle owns a cell -- so an exact tie is an integer equality and the winner is the lowest
# particle id. Float64 is used only for conservative candidate sets and margin inequalities (never
# for `==`, never for ordering); cKDTree's distances smear symmetric ties by 1 ulp.

def cellcentra(ursprung, pitch, form):
    """(M, 3) float64 cell centres, C-order, index = i*ny*nz + j*nz + k."""
    ursprung = np.asarray(ursprung, dtype=np.float64)
    axlar = [ursprung[d] + (np.arange(int(form[d])) + 0.5) * float(pitch) for d in range(3)]
    g = np.stack(np.meshgrid(*axlar, indexing="ij"), axis=-1)
    return np.ascontiguousarray(g.reshape(-1, 3), dtype=np.float64)


def celler_i_lada(form, ursprung, pitch, lo, hi):
    """Linear indices of the cells whose CENTRE lies inside [lo, hi] (inclusive), ascending."""
    ursprung = np.asarray(ursprung, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    form = np.asarray(form, dtype=np.int64)
    i0 = np.ceil((lo - ursprung) / pitch - 0.5).astype(np.int64)
    i1 = np.floor((hi - ursprung) / pitch - 0.5).astype(np.int64)
    i0 = np.maximum(i0, 0)
    i1 = np.minimum(i1, form - 1)
    if np.any(i0 > i1):
        return np.zeros(0, dtype=np.int64)
    axlar = [np.arange(i0[d], i1[d] + 1) for d in range(3)]
    g = np.meshgrid(*axlar, indexing="ij")
    lin = (g[0] * (form[1] * form[2]) + g[1] * form[2] + g[2]).reshape(-1)
    return np.unique(lin)


def aabb_celler(form, ursprung, pitch, lo, hi, pad=0.0):
    """Conservative superset: every cell whose centre is within `pad` of the AABB [lo, hi]."""
    lo = np.asarray(lo, dtype=np.float64) - float(pad)
    hi = np.asarray(hi, dtype=np.float64) + float(pad)
    return celler_i_lada(form, ursprung, pitch, lo, hi)


def kvantisera(X):
    """(N, 3) int64 fixed-point coordinates: round(x * 1e9). Exact, no float remains."""
    return np.rint(np.asarray(X, dtype=np.float64) * float(KVANTUM)).astype(np.int64)


def r2_kvant(radie):
    """radie**2 in quanta**2 as a Python int (r <= 0.012 gives (r*1e9)**2 < 2**53)."""
    return int(np.rint((float(radie) * float(KVANTUM)) ** 2))


def _bygg_agartrad(punkter):
    """One cKDTree per tick, shared by the reference and the incremental subset calls."""
    from scipy.spatial import cKDTree

    if len(punkter) == 0:
        return None
    return cKDTree(np.ascontiguousarray(punkter, dtype=np.float64),
                   balanced_tree=False, compact_nodes=False)


def _ballot_agare(Pq, agare, Cq, r2q, bollar):
    """Vectorised int64 ballot over ball-candidate lists: winner = lowest particle id at the
    minimal int64 d2 <= r2q. Returns {local row: owner} for rows with an in-range candidate."""
    counts = np.fromiter((len(b) for b in bollar), dtype=np.int64, count=len(bollar))
    nz = np.nonzero(counts)[0]
    if len(nz) == 0:
        return {}
    delar = np.concatenate([np.asarray(bollar[j], dtype=np.int64) for j in nz])
    rader = np.repeat(nz, counts[nz])
    d = Pq[delar] - Cq[rader]
    d2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
    m = d2 <= r2q
    if not bool(np.any(m)):
        return {}
    rm, d2m, pm = rader[m], d2[m], delar[m]
    ordning = np.lexsort((pm, d2m, rm))  # primary cell, then d2, then particle id
    rm_s = rm[ordning]
    forst = np.concatenate((np.ones(1, dtype=bool), rm_s[1:] != rm_s[:-1]))
    vinn = ordning[forst]
    return dict(zip(rm[vinn].tolist(), agare[pm[vinn]].tolist()))


def agare_exakt(punkter, agare, cellcentra, radie, cellindex, trad=None):
    """The tie-break rule on a cell subset: cKDTree k=4 with `distance_upper_bound=radie+M`,
    then the exact int64 ballot only for tied or boundary cells. `trad` (from `_bygg_agartrad`)
    may be shared across calls over the same particles; the result is identical without it.

    M = TIE_MARGIN. Tree error (~1e-17 m) and the 0.5 nm quantization both lie far below M, so
    the float margin inequalities are exact proofs, not nearness tests:
    * out of range (d0 > radie, not within M): -1.
    * clear (d0 <= radie, |d0-radie| > M, k=2 gap > M): the tree's winner is uniquely nearest
      by value and nearest in int64 too.
    * agreed tie (k=2 gap <= M but every neighbour within M/2 of the minimum shares i0's owner,
      witnessed inside the top 4): the ballot winner has that owner, so no ballot is needed.
    * ballot (boundary, or a tie without owner agreement): exact int64 ballot over
      ball(radie + BALL_EPS), winner = lowest id at minimal d2. No float `==` anywhere.
    """
    J = 4
    S = len(cellindex)
    ut = np.full(S, -1, dtype=np.int32)
    if S == 0 or len(punkter) == 0:
        return ut
    agare = np.asarray(agare, dtype=np.int32)
    N = len(agare)
    if trad is None:
        trad = _bygg_agartrad(punkter)
    pts = np.ascontiguousarray(np.asarray(cellcentra)[cellindex], dtype=np.float64)
    M = TIE_MARGIN
    halv = 0.5 * M
    with np.errstate(invalid="ignore"):  # empty cells hold inf; handled below, never warn
        dd, ii = trad.query(pts, k=J, workers=1, distance_upper_bound=float(radie) + M)
        dd = np.asarray(dd, dtype=np.float64).reshape(S, J)
        ii = np.asarray(ii, dtype=np.int64).reshape(S, J)
        d0 = dd[:, 0]
        gap2 = np.where(np.isinf(dd[:, 1]), np.inf, dd[:, 1] - d0)
        inG = (dd - d0[:, None]) <= halv
        gJ = np.where(np.isinf(dd[:, J - 1]), np.inf, dd[:, J - 1] - d0)
    giltig = ii < N
    egen = agare[np.where(giltig, ii, 0)]
    samma0 = giltig & (egen == egen[:, :1])
    begransad = gJ > halv
    enasG = begransad & ((~inG | samma0).all(axis=1))
    nara_r = np.abs(d0 - float(radie)) <= M
    inom = d0 <= float(radie)
    oppen = d0 <= float(radie) + M
    unik = gap2 > M
    hoppa = inom & (~nara_r) & (unik | enasG)
    ih = np.nonzero(hoppa)[0]
    ut[ih] = agare[ii[ih, 0]]
    exakta = np.nonzero(oppen & (~hoppa))[0]
    if len(exakta):
        bollar = trad.query_ball_point(pts[exakta], float(radie) + BALL_EPS)
        Pq = kvantisera(punkter)
        Cq = kvantisera(np.ascontiguousarray(np.asarray(cellcentra)[cellindex[exakta]],
                                             dtype=np.float64))
        r2q = np.int64(r2_kvant(radie))
        vinst = _ballot_agare(Pq, agare, Cq, r2q, list(bollar))
        vr = np.fromiter(vinst.keys(), dtype=np.int64, count=len(vinst))
        va = np.fromiter(vinst.values(), dtype=np.int32, count=len(vinst))
        ut[exakta[vr]] = va
    return ut


def agare_brute_kvant(punkter, agare, cellcentra, radie, cellindex):
    """Independent reference for the same rule: chunked int64 brute force over ALL particles,
    no tree and no margin. Too slow for the update path; the test uses it to prove that two
    implementations break every tie identically."""
    cellindex = np.asarray(cellindex, dtype=np.int64)
    ut = np.full(len(cellindex), -1, dtype=np.int32)
    if len(cellindex) == 0 or len(punkter) == 0:
        return ut
    agare = np.asarray(agare)
    Pq = kvantisera(punkter)
    Cq = kvantisera(np.ascontiguousarray(np.asarray(cellcentra)[cellindex], dtype=np.float64))
    r2q = np.int64(r2_kvant(radie))
    steg = max(1, int(4_000_000 // max(len(Pq), 1)))
    for s in range(0, len(Cq), steg):
        cc = Cq[s:s + steg]
        dx = cc[:, None, 0] - Pq[None, :, 0]
        dy = cc[:, None, 1] - Pq[None, :, 1]
        dz = cc[:, None, 2] - Pq[None, :, 2]
        d2 = dx * dx + dy * dy + dz * dz
        m = d2 <= r2q
        d2m = np.where(m, d2, _INT64_MAX)
        dmin = d2m.min(axis=1)
        for k in range(len(cc)):
            if not bool(m[k].any()):
                continue
            lika = np.nonzero((d2[k] == dmin[k]) & m[k])[0]
            ut[s + k] = agare[int(lika.min())]
    return ut


def agare_full(punkter, agare, cellcentra, radie, trad=None):
    """Rebuild reference: exact owner per cell under the int64 tie-break rule."""
    return agare_exakt(punkter, agare, cellcentra, radie,
                       np.arange(len(cellcentra), dtype=np.int64), trad=trad)


def agare_delvis(punkter, agare, cellcentra, radie, cellindex, trad=None):
    """Nearest owner for a subset of cells; identical semantics to `agare_full`."""
    return agare_exakt(punkter, agare, cellcentra, radie,
                       np.asarray(cellindex, dtype=np.int64), trad=trad)


def borttagna_per_agare(punkter_fore, agare_fore, agare):
    """Owner -> the removed particles (the previous cloud's suffix) whose count fell.

    Assumes an owner's surviving particles are a PREFIX of its previous cloud: a shrinking pile
    keeps frames[f][:n], a vanished body loses its whole tail. An owner absent now has all its
    previous particles removed. A removal in the middle of a cloud would need a set difference.
    """
    ut = {}
    agare_fore = np.asarray(agare_fore)
    for o in np.unique(agare_fore):
        o = int(o)
        n_fore = int(np.count_nonzero(agare_fore == o))
        n_nu = int(np.count_nonzero(np.asarray(agare) == o))
        if n_nu < n_fore:
            pts = np.asarray(punkter_fore, dtype=np.float64)[agare_fore == o]
            svans = pts[n_nu:] if n_nu > 0 else pts
            if len(svans):
                ut[o] = svans
    return ut


def borttagningsceller(agare_fore_kanal, punkter, agare, punkter_fore, agare_fore,
                       form, ursprung, pitch, radie):
    """Cells that must be re-judged because material was removed.

    For every owner whose count fell, the radius-expanded AABB of the removed particles, restricted
    to the cells that owner held at the previous tick (the owner's own cell-set difference, never the
    whole field). A cell can only lose its nearest particle when a particle within `radie` of it
    disappeared, and every such cell was owned by that particle's owner, so the filter loses nothing.
    Ascending unique linear indices.
    """
    borttagna = borttagna_per_agare(punkter_fore, agare_fore, agare)
    if not borttagna:
        return np.zeros(0, dtype=np.int64)
    delar = []
    for o, pts in borttagna.items():
        lo = pts.min(axis=0) - float(radie)
        hi = pts.max(axis=0) + float(radie)
        cand = aabb_celler(form, ursprung, pitch, lo, hi)
        cand = cand[agare_fore_kanal[cand] == o]
        if len(cand):
            delar.append(cand)
    if not delar:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(delar))


def agare_inkrementell(agare_fore_kanal, punkter, agare, punkter_fore, agare_fore,
                       form, ursprung, pitch, radie, misstankta, agare_fn):
    """The incremental update: the removal pass first, then the movement/addition suspect set.

    `agare_fn(punkter, agare, cellindex)` is the per-cell owner back end (a KDTree subset here).
    The removal pass runs on the copy of the previous channel BEFORE the suspect cells are
    overwritten, so no stale owner survives a removed particle. Returns
    (channel, n_removal_cells, n_suspect_cells).
    """
    kanal = np.asarray(agare_fore_kanal, dtype=np.int32).copy()
    bort = borttagningsceller(kanal, punkter, agare, punkter_fore, agare_fore,
                              form, ursprung, pitch, radie)
    if len(bort):
        kanal[bort] = agare_fn(punkter, agare, bort)
    misstankta = np.asarray(misstankta, dtype=np.int64)
    if len(misstankta):
        kanal[misstankta] = agare_fn(punkter, agare, misstankta)
    return kanal, int(len(bort)), int(len(misstankta))


def keepout_dom(agare_kanal, box_index):
    """Canonical bytes of the keep-out judgement over `box_index` (ascending linear indices)."""
    occ_idx = box_index[agare_kanal[box_index] >= 0]
    owners = agare_kanal[occ_idx]
    nyttolast = np.concatenate([occ_idx.astype("<i8"), owners.astype("<i4")]).tobytes()
    return nyttolast, int(occ_idx.size)


def dom_sha(nyttolast):
    return hashlib.sha256(nyttolast).hexdigest()


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
