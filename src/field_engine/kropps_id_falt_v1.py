#!/usr/bin/env python3
"""One field with an owner index, instead of one scalar field that forgot whose material it is.

A cell full of moving bodies needs distances that are ATTRIBUTED. A single scalar union field
answers "how far to the nearest material", which is the wrong question for contact: at a point on a
gripper's own pad the nearest material is the gripper, so the union field reports a distance to
itself and the contact routine reads a gap that does not exist. Measured on a polishing cell, a
scalar union alone gets 452 contact points where there are 48, a contact-count error of 8.42 and a
gap error of 0.07 m.

This module stores one field with a channel per (owner, block): the base union for "is anything
here", and an owner-indexed channel for "how far to everything EXCEPT this body". Outside an owner's
own window the channel returns a fixed positive bound rather than a computed distance, so a query
far from a body costs nothing and can never be mistaken for contact.

MEASURED on that cell, against analytic references, at a budget of 8.5 %:

    stored                              2 836 836 B
    uniform at the resolution it needs 58 320 276 B      ratio 20.558 212 036
    contact                             44 of 48 points over 16 of 16 pairs
    compliance deviation                3.989 965 914 %  (gate 5 %)
    certificate deviation               3.119 320 120e-10 %
    route                               7.2 m, unchanged
    88 allocation cases bit-identical over two processes

AND THE REPRESENTATION CLAIM THAT DOES NOT SURVIVE. "A scalar union field is not enough" is FALSE as
a statement about representations. The same scalar union plus an EXTERNAL binary body mask gets
48 of 48 points and passes every gate, with a gap error of 7.33e-17 m. What the measured run showed
was that ITS integration of the union was wrong, not that the union cannot carry the job. The honest
form of the finding is narrower and still useful: the body identity has to be stored SOMEWHERE, and
the open question is where it is cheapest -- the mask's own geometry is not charged to the 2.8 MB
above, because the mask calls the source geometry at query time.

The 44 of 48 is also not 44 correct points: it is 10 missing and 6 added, and two support patches
collapse from four contact points to two. A count that is 92 % right can be a support set that is
qualitatively wrong, and under a perturbation the same cell settles 0.318 mm lower with 6 contacts
against the reference's 8.

WHAT DOES NOT TRANSFER. On a second cell -- a bolted assembly rather than a machine cell -- the same
policy passes 0 of 8 budgets from 1 % to 100 %, all of them on the stability gate. The 20x is a
number about that cell and that mesh family, not about the representation, and no uniform grid from
n = 9 to n = 25 passes there either.

ON THE ASSEMBLY THE FIELD IS NOT WHAT EARNS THE NUMBERS. The same allocation reaches three goals
inside 5 % at 886 263 B, which is 4.42 % of the 20 051 151 B uniform. But the exact CAD path computes
the same three goals WITHOUT ANY FIELD in 91.17 ms against the field's 96.06 ms, and it does so by
reading 1 116 069 B of input, not the 5 800 B the report credited it with. For contact margin,
routing clearance and compliance on rigid parts, the field earns nothing; what a field can do and
CAD cannot -- swept volumes, deforming bodies, a sensor field, an occupancy that changes every
10 ms -- is where the case has to be made, and those are not these three goals.

Two further limits on that assembly: the contact-margin reference is MESH-DEPENDENT (an independent
triangle assembly of the same parts gives a 100.20 % relative error on it, at a condition number of
9.7e9), and contact parity against a ray-traced narrow phase is 5 pairs against 3 -- the two bolt
pairs exist only in the field's reading. What stands is the storage ratio, the routing and
compliance goals, and 42 of 42 bit-identical artefacts.
"""
import hashlib
import json
import os
import struct

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "data", "kropps_id_falt_v1")

# The level menu: a block is stored at n^3 samples for one of these n. Level 0 is a single mean.
NIVAER = (1, 2, 3, 5, 9, 17)
UTANFOR_BOUND_M = 0.08          # what an owner channel answers outside that owner's own window
HUVUD_BYTES = 12                # container header: record count and level count


def nivaprover(niva):
    """Samples stored for a block at this level."""
    return int(NIVAER[int(niva)]) ** 3


def lagrade_bytes(nivaer_bas, nivaer_agare):
    """The container's size, as the cost formula it is.

    This is an ACCOUNTING FORMULA over the levels, not a measurement of a file on a disk, and it is
    the number every ratio in this module is built from. It counts four bytes per stored float
    sample, one byte per level entry, eight bytes per owner record, and the header."""
    prov = sum(nivaprover(n) for n in nivaer_bas) + sum(nivaprover(n) for n in nivaer_agare)
    return int(HUVUD_BYTES + 4 * prov + len(nivaer_bas) + len(nivaer_agare)
               + 8 * len(nivaer_agare))


class KroppsIdFalt:
    """A block field with a base union channel and one channel per (owner, block) record.

    poster is an (N, 2) int array of (owner id, block index). Owner ids are the caller's; negative
    ids are conventionally static bodies (the machine, the ground) and non-negative ones the movable
    bodies, but nothing in this class depends on that."""

    def __init__(self, blockform, blockstorlek, ursprung, nivaer_bas, varden_bas,
                 poster, nivaer_agare, varden_agare, utanfor=UTANFOR_BOUND_M):
        self.blockform = tuple(int(v) for v in blockform)
        self.blockstorlek = float(blockstorlek)
        self.ursprung = np.asarray(ursprung, dtype=np.float64)
        self.nivaer_bas = np.asarray(nivaer_bas, dtype=np.uint8)
        self.varden_bas = [np.asarray(v, dtype=np.float32) for v in varden_bas]
        self.poster = np.asarray(poster, dtype=np.int32).reshape(-1, 2)
        self.nivaer_agare = np.asarray(nivaer_agare, dtype=np.uint8)
        self.varden_agare = [np.asarray(v, dtype=np.float32) for v in varden_agare]
        self.utanfor = float(utanfor)
        self.karta = {(int(o), int(b)): i for i, (o, b) in enumerate(self.poster)}
        n_block = int(np.prod(self.blockform))
        if len(self.nivaer_bas) != n_block:
            raise ValueError(f"{len(self.nivaer_bas)} base levels for {n_block} blocks")
        if len(self.nivaer_agare) != len(self.poster):
            raise ValueError("one level per owner record is required")

    # -- addressing --------------------------------------------------------------------------------
    def blockindex(self, P):
        """Block index per query point. A point outside the declared domain RAISES; there is no
        analytic fallback, because a field that quietly answers outside its own support is a field
        whose support cannot be audited."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        ijk = np.floor((P - self.ursprung) / self.blockstorlek).astype(np.int64)
        if np.any(ijk < 0) or np.any(ijk >= np.asarray(self.blockform)):
            raise ValueError("query point outside the field's declared domain")
        nx, ny, nz = self.blockform
        return ijk[:, 0] * (ny * nz) + ijk[:, 1] * nz + ijk[:, 2], ijk

    def _prov(self, varden, niva, index, P, ijk):
        """Nearest-sample read inside a block at its stored level."""
        n = NIVAER[int(niva)]
        v = varden[index].reshape(n, n, n)
        lokal = (P - (self.ursprung + ijk * self.blockstorlek)) / self.blockstorlek
        s = np.clip((lokal * n).astype(np.int64), 0, n - 1)
        return float(v[s[0], s[1], s[2]])

    def query(self, P):
        """The base union channel: distance to the nearest material of any body."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        b, ijk = self.blockindex(P)
        return np.array([self._prov(self.varden_bas, self.nivaer_bas[bi], bi, P[i], ijk[i])
                         for i, bi in enumerate(b)])

    def query_target(self, P, agare):
        """Distance to everything EXCEPT `agare`.

        Outside that owner's stored window the answer is the fixed bound, which is deliberately
        larger than any contact margin the caller should be using: an unstored region must read as
        'certainly not in contact', never as a small number that happens to be the default."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        b, ijk = self.blockindex(P)
        ut = np.full(len(P), self.utanfor)
        for i, bi in enumerate(b):
            j = self.karta.get((int(agare), int(bi)))
            if j is not None:
                ut[i] = self._prov(self.varden_agare, self.nivaer_agare[j], j, P[i], ijk[i])
        return ut

    # -- container ---------------------------------------------------------------------------------
    def serialisera(self):
        """One container, deterministic byte order: header, records, levels, then the samples of the
        base channel and of every owner channel in record order."""
        delar = [struct.pack("<II", len(self.poster), len(self.nivaer_bas)),
                 self.poster.astype("<i4").tobytes(),
                 self.nivaer_bas.tobytes(), self.nivaer_agare.tobytes()]
        for v in self.varden_bas:
            delar.append(np.asarray(v, dtype="<f4").tobytes())
        for v in self.varden_agare:
            delar.append(np.asarray(v, dtype="<f4").tobytes())
        return b"".join(delar)

    def sha256(self):
        return hashlib.sha256(self.serialisera()).hexdigest()

    def bytes(self):
        return lagrade_bytes(self.nivaer_bas, self.nivaer_agare)


def bygg_syntetisk_cell(blockform=(4, 4, 3), blockstorlek=0.1, agare=(-1, 0, 1, 2),
                        fonsterradie=1, niva=3, fro=20260920):
    """A small owner-indexed field to exercise the container on: every owner keeps channels only in
    the blocks within `fonsterradie` of its own seed block, which is what makes the record count
    far smaller than owners x blocks."""
    rng = np.random.default_rng(fro)
    nx, ny, nz = blockform
    n_block = nx * ny * nz
    nivaer_bas = np.full(n_block, niva, dtype=np.uint8)
    varden_bas = [rng.normal(0.0, 0.03, nivaprover(niva)).astype(np.float32)
                  for _ in range(n_block)]
    poster, nivaer_agare, varden_agare = [], [], []
    for k, ow in enumerate(agare):
        seed = np.array([(k * 3) % nx, (k * 2) % ny, k % nz])
        for bi in range(n_block):
            ijk = np.array([bi // (ny * nz), (bi // nz) % ny, bi % nz])
            if np.max(np.abs(ijk - seed)) <= fonsterradie:
                poster.append((ow, bi))
                nivaer_agare.append(niva)
                varden_agare.append(rng.normal(0.02, 0.03, nivaprover(niva)).astype(np.float32))
    return KroppsIdFalt(blockform, blockstorlek, (0.0, 0.0, 0.0), nivaer_bas, varden_bas,
                        np.array(poster), np.array(nivaer_agare, dtype=np.uint8), varden_agare)


def las_allokering(sokvag=None):
    with open(os.path.join(sokvag or DATA, "allokering.json")) as fh:
        return json.load(fh)


def _selftest():
    f = bygg_syntetisk_cell()
    n_block = int(np.prod(f.blockform))
    print(f"synthetic cell: {n_block} blocks, {len(f.poster)} owner records over "
          f"{len(set(int(o) for o, _ in f.poster))} owners, {f.bytes()} B")

    # the container is a function of the field alone
    a, b = f.sha256(), bygg_syntetisk_cell().sha256()
    print(f"container hash reproduced across two builds: {a == b}  {a[:16]}...")

    # an owner's channel is the bound outside its own window, and a real number inside it
    P_in = f.ursprung + (f.poster[0, 1] // (f.blockform[1] * f.blockform[2]) + 0.5) * 0.0
    inne = []
    for ow, bi in f.poster[:1]:
        ijk = np.array([bi // (f.blockform[1] * f.blockform[2]),
                        (bi // f.blockform[2]) % f.blockform[1], bi % f.blockform[2]])
        P_in = f.ursprung + (ijk + 0.5) * f.blockstorlek
        inne.append(float(f.query_target(P_in, int(ow))[0]))
    langt = float(f.query_target(f.ursprung + np.array([0.35, 0.35, 0.25]), 999)[0])
    print(f"owner channel inside its window {inne[0]:+.5f} m, unstored owner reads {langt:.5f} m")

    d = las_allokering()
    m = d["maskincell"]
    print(f"carried: {m['bytes']} B against {m['uniform_bytes']} B uniform = {m['kvot']:.9f}x, "
          f"contact {m['kontakt_antal']}/{m['referens_kontakt_antal']}, "
          f"compliance {100*m['compliance_avvikelse']:.6f} %")
    mk = d["extern_kroppsmask"]
    print(f"carried counter-result: union + external body mask reaches "
          f"{mk['kontakt_antal']}/48 and passes all gates ({mk['passerar_alla']})")
    print(f"carried counter-cell: {d['motcell']['passerande']}/{d['motcell']['av']} budgets pass")

    fel = []
    if a != b:
        fel.append("the container is not a function of the field alone")
    if langt != UTANFOR_BOUND_M:
        fel.append("an unstored owner channel must read the declared bound")
    if not (-0.2 < inne[0] < 0.2) or inne[0] == UTANFOR_BOUND_M:
        fel.append("a stored owner channel must read a sample, not the bound")
    if abs(m["kvot"] - m["uniform_bytes"] / m["bytes"]) > 1e-12:
        fel.append("the carried ratio is not the carried byte counts")
    if mk["kontakt_antal"] != 48 or not mk["passerar_alla"]:
        fel.append("the counter-result must be carried as measured")
    if d["motcell"]["passerande"] != 0:
        fel.append("the counter-cell result must be carried as measured")
    try:
        f.query(np.array([[-1.0, 0.0, 0.0]]))
        fel.append("a query outside the domain must raise")
    except ValueError:
        pass
    if fel:
        raise SystemExit("FAIL: " + "; ".join(fel))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
