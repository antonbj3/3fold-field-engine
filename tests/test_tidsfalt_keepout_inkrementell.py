"""The incremental owner occupancy and the tick it would miss without its removal pass.

The field's time-varying path rebuilds the whole occupancy every tick. Keeping the owner channel and
re-judging only the cells that can change is correct for material that moves or arrives, and wrong
for material that leaves: a removed body and a shrinking pile leave cells whose owner no longer
covers them. This file runs the five audited scenes against the full rebuild and locks two readings:
with the removal pass there are 0 divergent ticks, and without it the two removal scenes diverge on
exactly the ticks the audit recorded. The mechanism runs live in a few seconds; no simulator, no GPU.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "field_engine"))

import tidsfalt_keepout_v1 as TK                                                    # noqa: E402


# ───────────────────────────── scenes ─────────────────────────────
class Scen:
    """A tick sequence on a grid; bodies are particle clouds with an owner id (>= 0)."""

    def __init__(self, namn, ursprung, pitch, form, radie, lada_lo, lada_hi, ticks):
        self.name = namn
        self.origin = tuple(float(v) for v in ursprung)
        self.pitch = float(pitch)
        self.form = tuple(int(v) for v in form)
        self.radius = float(radie)
        self.box_lo = tuple(float(v) for v in lada_lo)
        self.box_hi = tuple(float(v) for v in lada_hi)
        self.ticks = int(ticks)
        self.centers = TK.cellcentra(self.origin, self.pitch, self.form)
        self.box_idx = TK.celler_i_lada(self.form, self.origin, self.pitch,
                                        self.box_lo, self.box_hi)

    def particle_state(self, tick):
        raise NotImplementedError


class Hog(Scen):
    """The field's own growing pile: all particles owner 0, so the channel only grows."""

    def __init__(self, n_ramar=40):
        super().__init__("hog", (-0.16, -0.06, 0.0), 0.008, (40, 16, 8), 0.006,
                         TK.KEEPOUT_MIN, TK.KEEPOUT_MAX, n_ramar)
        self._frames = TK.syntetisk_hog(n_ramar, 60, 20260920)

    def particle_state(self, tick):
        P = np.ascontiguousarray(self._frames[tick], dtype=np.float64)
        return P, np.zeros(len(P), dtype=np.int32), np.array([[0, len(P)]], dtype=np.int64)


class TreKroppar(Scen):
    """Three bodies on smooth sinusoids, no body ever removed."""

    def __init__(self, ticks=200):
        super().__init__("tre_kroppar", (-0.15, -0.15, -0.02), 0.012, (25, 25, 10), 0.012,
                         (-0.036, -0.036, 0.0), (0.036, 0.036, 0.036), ticks)
        h = 0.030
        pts = np.arange(-h, h + 1e-12, self.pitch)
        g = np.stack(np.meshgrid(pts, pts, pts, indexing="ij"), axis=-1).reshape(-1, 3)
        self._body = g.astype(np.float64)
        self._base = np.array([[-0.055, 0.0, 0.018], [0.055, 0.0, 0.018],
                               [0.0, 0.055, 0.018]], dtype=np.float64)
        self._amp = np.array([[0.070, 0.0, 0.0], [0.0, 0.070, 0.0],
                              [0.0, 0.0, 0.030]], dtype=np.float64)
        self._period = np.array([180.0, 240.0, 150.0])
        self._phase = np.array([0.0, 1.1, 2.3])

    def centers_of(self, tick):
        s = np.sin(2.0 * np.pi * tick / self._period + self._phase)
        return self._base + s[:, None] * self._amp

    def particle_state(self, tick):
        c = self.centers_of(tick)
        P = (self._body[None, :, :] + c[:, None, :]).reshape(-1, 3)
        O = np.repeat(np.arange(3, dtype=np.int32), len(self._body))
        body = np.stack([np.arange(3) * len(self._body),
                         np.arange(1, 4) * len(self._body)], axis=1).astype(np.int64)
        return np.ascontiguousarray(P, dtype=np.float64), O, body


class EdgeScene(Scen):
    """Three cubes on a small grid; body 0 jumps `jump_cells` per tick, body 2 can vanish."""

    def __init__(self, namn, jump_cells=3, disappear_at=None, ticks=8):
        super().__init__(namn, (-0.15, -0.15, -0.02), 0.012, (25, 25, 10), 0.012,
                         (-0.06, -0.06, 0.0), (0.06, 0.06, 0.05), ticks)
        h = 0.030
        pts = np.arange(-h, h + 1e-12, self.pitch)
        g = np.stack(np.meshgrid(pts, pts, pts, indexing="ij"), axis=-1).reshape(-1, 3)
        self._body = g.astype(np.float64)
        self._jump = float(jump_cells) * self.pitch
        self.disappear_at = disappear_at

    def n_active(self, tick):
        return 2 if (self.disappear_at is not None and tick >= self.disappear_at) else 3

    def centers_of(self, tick):
        c = np.array([[-0.09 + tick * self._jump, 0.0, 0.018],
                      [0.06, 0.0, 0.018], [0.0, 0.06, 0.018]])
        return c[:self.n_active(tick)]

    def particle_state(self, tick):
        c = self.centers_of(tick)
        nb = len(c)
        P = (self._body[None, :, :] + c[:, None, :]).reshape(-1, 3)
        O = np.repeat(np.arange(nb, dtype=np.int32), len(self._body))
        body = np.stack([np.arange(nb) * len(self._body),
                         np.arange(1, nb + 1) * len(self._body)], axis=1).astype(np.int64)
        return np.ascontiguousarray(P, dtype=np.float64), O, body


class ShrinkScene(Scen):
    """A pile whose particle count falls: 60, 120, 90, 60, 30, 0 particles (owners all 0)."""

    def __init__(self, ticks=6):
        super().__init__("shrink", (-0.16, -0.06, 0.0), 0.008, (40, 16, 8), 0.006,
                         (-0.06, -0.06, 0.0), (0.06, 0.06, 0.04), ticks)
        rng = np.random.default_rng(7)
        counts = [60, 120, 90, 60, 30, 0][:ticks]
        allp = rng.normal(0.0, 0.02, size=(max(counts), 3))
        allp[:, 2] = np.abs(allp[:, 2])
        self._frames = [np.ascontiguousarray(allp[:n]) for n in counts]

    def particle_state(self, tick):
        P = self._frames[tick]
        return P, np.zeros(len(P), dtype=np.int32), np.array([[0, len(P)]], dtype=np.int64)


# the suspect set the movement/addition path uses; the removal pass is the field module's
def suspect_cells(scene, f, radie):
    old_c = scene.centers_of(f - 1)
    new_c = scene.centers_of(f)
    delar = []
    for b in range(len(new_c)):
        lo = np.minimum(scene._body.min(axis=0) + old_c[b],
                        scene._body.min(axis=0) + new_c[b]) - radie
        hi = np.maximum(scene._body.max(axis=0) + old_c[b],
                        scene._body.max(axis=0) + new_c[b]) + radie
        delar.append(TK.aabb_celler(scene.form, scene.origin, scene.pitch, lo, hi))
    return np.unique(np.concatenate(delar)) if delar else np.zeros(0, np.int64)


def suspect_cells_pile(scene, f, radie):
    n_prev = len(scene._frames[f - 1]) if f > 0 else 0
    P = scene._frames[f][n_prev:]
    if len(P) == 0:
        return np.zeros(0, np.int64)
    return TK.aabb_celler(scene.form, scene.origin, scene.pitch,
                          P.min(axis=0) - radie, P.max(axis=0) + radie)


def suspect_for(scene, f, radie):
    if hasattr(scene, "centers_of"):
        return suspect_cells(scene, f, radie)
    return suspect_cells_pile(scene, f, radie)


def _run(scene):
    radie = scene.radius
    centers = scene.centers
    agare_fn = lambda P, O, idx: TK.agare_delvis(P, O, centers, radie, idx)      # noqa: E731
    prev_del = prev_nodel = None
    P_prev = O_prev = None
    rows = []
    for f in range(scene.ticks):
        P, O, _ = scene.particle_state(f)
        a = TK.agare_full(P, O, centers, radie)
        if f == 0:
            b_del = b_nodel = a.copy()
            n_del = 0
        else:
            misstankta = suspect_for(scene, f, radie)
            b_del, n_del, _ = TK.agare_inkrementell(prev_del, P, O, P_prev, O_prev,
                                                    scene.form, scene.origin, scene.pitch,
                                                    radie, misstankta, agare_fn)
            b_nodel = prev_nodel.copy()
            if len(misstankta):
                b_nodel[misstankta] = agare_fn(P, O, misstankta)
        nytt_a, n_a = TK.keepout_dom(a, scene.box_idx)
        nytt_del, n_del_dom = TK.keepout_dom(b_del, scene.box_idx)
        nytt_nodel, n_nodel_dom = TK.keepout_dom(b_nodel, scene.box_idx)
        rows.append({
            "tick": f, "n_particles": int(len(P)), "n_deletion": int(n_del),
            "dom_a": n_a, "dom_del": n_del_dom, "dom_nodel": n_nodel_dom,
            "dom_sha_a": TK.dom_sha(nytt_a), "dom_sha_del": TK.dom_sha(nytt_del),
            "dom_sha_nodel": TK.dom_sha(nytt_nodel),
            "owner_del_equal": bool(np.array_equal(a, b_del)),
            "owner_nodel_equal": bool(np.array_equal(a, b_nodel)),
        })
        prev_del, prev_nodel = b_del, b_nodel
        P_prev, O_prev = P, O
    return rows


FALL = ("hog", "tre_kroppar", "jump_3cells", "disappear_at_4", "shrink")


@pytest.fixture(scope="module")
def fem():
    scener = [Hog(), TreKroppar(ticks=200), EdgeScene("jump_3cells", jump_cells=3),
              EdgeScene("disappear_at_4", jump_cells=3, disappear_at=4), ShrinkScene()]
    return {sc.name: _run(sc) for sc in scener}


# --- the removal pass, live -----------------------------------------------------------------------
def test_noll_avvikande_tick_med_borttagning(fem):
    """With the removal pass the incremental dom and the owner channel equal the rebuild every tick.

    The zero is the audited reading: 0 of 40 / 200 / 8 / 8 / 6 ticks diverge.
    """
    for namn in FALL:
        rows = fem[namn]
        divergenta = [r["tick"] for r in rows if r["dom_sha_a"] != r["dom_sha_del"]]
        assert divergenta == []
        assert all(r["owner_del_equal"] for r in rows)


def test_utan_borttagning_avviker_de_tva_borttagningsscenerna(fem):
    """Without the pass the mechanism the pass fixes is visible on exactly the audited ticks:
    the vanished body on 4, 5, 6, 7 and the shrinking pile on 2, 3, 4, 5."""
    vantade = {"disappear_at_4": [4, 5, 6, 7], "shrink": [2, 3, 4, 5]}
    for namn, ticks in vantade.items():
        rows = fem[namn]
        assert [r["tick"] for r in rows if r["dom_sha_a"] != r["dom_sha_nodel"]] == ticks
        assert all(not r["owner_nodel_equal"] for r in rows if r["tick"] in ticks)
    for namn in ("hog", "tre_kroppar", "jump_3cells"):
        assert all(r["dom_sha_a"] == r["dom_sha_nodel"] for r in fem[namn])


def test_borttagningscellerna_ar_exakt_de_tva_fallen(fem):
    """The pass re-judges 234 cells when the body vanishes and 138/113/81/42 when the pile shrinks,
    and costs nothing on the three scenes without removal."""
    assert [r["n_deletion"] for r in fem["disappear_at_4"]] == [0, 0, 0, 0, 234, 0, 0, 0]
    assert [r["n_deletion"] for r in fem["shrink"]] == [0, 0, 138, 113, 81, 42]
    for namn in ("hog", "tre_kroppar", "jump_3cells"):
        assert all(r["n_deletion"] == 0 for r in fem[namn])


def test_dom_cellerna_ur_auditen(fem):
    """The stale cells the naive update leaves: 112 against 176 when the body vanishes, and
    118 / 81 / 40 / 0 against 152 while the pile shrinks. The pass returns the rebuild's numbers."""
    d = {r["tick"]: r for r in fem["disappear_at_4"]}
    assert (d[4]["dom_a"], d[4]["dom_nodel"], d[4]["dom_del"]) == (112, 176, 112)
    s = {r["tick"]: r for r in fem["shrink"]}
    for tick, a in ((2, 118), (3, 81), (4, 40), (5, 0)):
        assert (s[tick]["dom_a"], s[tick]["dom_nodel"], s[tick]["dom_del"]) == (a, 152, a)


def test_agarkanalen_ar_fieldets_ockupans():
    """The owner channel's `>= 0` is the module's own occupancy: it agrees cell for cell with
    `ockupans_fran_partiklar` on the pile's first frame."""
    scen = Hog()
    P, O, _ = scen.particle_state(0)
    agare = TK.agare_full(P, O, scen.centers, scen.radius)
    occ = TK.ockupans_fran_partiklar(P, scen.origin, scen.form, scen.pitch, scen.radius)
    assert np.array_equal((agare >= 0).reshape(scen.form), occ)


def test_borttagna_ar_svansen_av_agarens_moln():
    """The prefix assumption stated in the docstring: the removed particles of a shrunk owner are
    exactly the tail beyond the current count, and an absent owner loses its whole cloud."""
    scen = ShrinkScene()
    P1, O1, _ = scen.particle_state(1)
    P2, O2, _ = scen.particle_state(2)
    bort = TK.borttagna_per_agare(P1, O1, O2)
    assert set(bort) == {0}
    assert bort[0].shape == (120 - 90, 3)
    assert np.array_equal(bort[0], P1[90:120])
    tom = TK.borttagna_per_agare(np.zeros((0, 3)), np.zeros(0, dtype=np.int32),
                                 np.zeros(0, dtype=np.int32))
    assert tom == {}
