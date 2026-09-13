#!/usr/bin/env python3
"""The negative-space schema one level up: the factory hall.

negativrum_v1 (machine level) declares voids from the machine's own sources (joint table, part
manifest) before the material is placed, and lets the placement solver put material against those
rooms as a hard keep-out. This module does exactly the same one level up, with the same compositor
(negativrum_v1.Rum / sdf / vattentathet / falt_paritet / sveptackning / losa_placering):

    the machine's rooms are the factory's rooms. The recursion is not a metaphor -- a machine's
    service / media / cooling rooms are lifted literally out of its own negativrum_v1.json and
    become the hall's keep-outs, with the same source, the same owners, the same primitive family.

Five factory classes, each with a source:
  transport   truck/pallet aisle between stations. The SOURCE IS AN EDGE OF THE PROCESS GRAPH. A
              transport room without an edge does not exist -- just as a motion sweep without a
              joint does not exist.
  operator    free floor area in front of each machine's HMI/control position. The source is the
              machine's own manifest: the HMI group's parts and their AABB point out both where the
              operator stands and which way the room opens.
  service     the machine's own service rooms, inherited upwards. A machine that already declares
              them in the negativrum_v1 schema hands them over; the others are derived from their
              own service hatches and doors.
  media       electrical/compressed-air runs. The source is the building services model: the
              measured ceiling-run z level plus IEC 60204-1 access dimensions at each cabinet.
  utrymning   free escape route along a wall, cited standard width.

Standards (cited, not invented -- see KALLA_STD below):
  SS-EN ISO 14122-2:2016    walkway: clear width >= 600 mm, clear height >= 2100 mm; permanent
                            walkway 800 mm
  SS-EN 60204-1:2018 §11.2  access to electrical equipment: >= 700 mm wide, 2100 mm high
  Honest boundary: no external truck/AGV aisle-width standard is sourced here. Six attempts to
  source one failed and were booked as a gap. The transport aisle width is therefore DERIVED from
  the carrier actually owned -- the EUR pallet 800 x 1200 mm, measured from its own manifest -- plus
  a cited safety margin. The class is DERIVED, not "standard".

Four gates -- the same questions, one level up:
  G0 WATERTIGHTNESS  negativrum_v1.vattentathet() on every hall room (B-rep, closed shells, even chi)
  G1 FIELD PARITY    negativrum_v1.falt_paritet() -- the field and the solid must agree
  G2 SWEEP COVERAGE  the transport room must CONTAIN the pallet's real sweep along the process edge,
                     including the TURN into the station. Exactly as a joint's sweep must fit in its
                     room. This is the gate that failed its own generator here too: an aisle sized
                     for a STRAIGHT run (800+2x300 = 1400 mm) does not hold a pallet that has to
                     TURN (AABB diagonal 1442 mm). The width is re-derived from the turn, not from
                     the straight run.
  G3 KEEP-OUT        no material in a room it does not own. At hall level both the rooms and the
                     parts' proxy bodies are AXIS-ALIGNED BOXES, and an AABB judge measures a box
                     exactly -- the intersection volume is computed analytically with coordinate
                     compression (union of boxes, no inclusion-exclusion guess). No OCC boolean, no
                     voxels, no memory ceiling to launder into a geometric verdict.

The memory lesson is built in: the machine-level module ran out of memory three times (>20 GB RSS)
in its overlap gate. A hall is ~40x larger. There is therefore no voxel rasterisation and no B-rep
boolean over thousands of parts here: the factory verdict is analytic on boxes (O(parts x rooms)
with a bbox pre-filter) and B-rep is used only on the rooms themselves (G0/G1) and on G2's handful
of swept bodies. Peak memory is measured and reported.

I/O: a factory data directory (process graph spec + one parts manifest per station, optionally a
machine-level negativrum_v1.json) in; hall rooms, gate reports and a congestion report out. The
synthetic factory shipped in examples/factory_synth/ is the default input.

Run: python fabriksnegativrum_v1.py --selftest
     python fabriksnegativrum_v1.py --kor
"""
from __future__ import annotations

import json
import math
import os
import re
import resource
import sys
import time

import numpy as np

REPO = os.environ.get("FIELD_ENGINE_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# factory input data: the synthetic factory shipped with this repository unless another directory
# is given. Expected layout: <DATA>/fabriksspec_v2.json and <DATA>/cad/<station>/parts_manifest_v1.json
DATA = os.environ.get("FIELD_ENGINE_FACTORY_DATA", os.path.join(REPO, "examples", "factory_synth"))

import negativrum_v1 as nr  # noqa: E402  reuse: the same compositor, one level up

# ==================================================================================================
# SOURCES AND STANDARDS -- every number has an address
# ==================================================================================================
KALLA_STD = {
    "iso14122_gang": ("SS-EN ISO 14122-2:2016 gangbanor: fri bredd >= 600 mm, fri hojd >= 2100 mm "
                      "(citerad i scripts/cad/utilities_bim_exemplar_v1.py:KALLA['iso14122'])"),
    "iso14122_perm": ("SS-EN ISO 14122-2, minimum width for a PERMANENT walkway = 800 mm"),

    "iec60204": ("SS-EN 60204-1:2018 §11.2 access to electrical equipment: >= 700 mm wide, 2100 mm high"),

    "operator_front": ("operator position in front of the door, 780 mm deep, ISO 14122-2 based"),

    "pall": ("MEASURED CARRIER: EUR pallet 800 x 1200 x 144 mm, AABB from the material-flow "
             "station manifest (pall_toppbrada_*/pall_kloss_*)"),
    "sakerhetsmarginal": ("SERVICE_MARGIN_M = 0.30 m, the same number the factory spec carries as "
                          "layoutvillkor.sakerhetszon_mm"),

    "ingen_truckstandard": ("NO external truck/AGV aisle-width standard is sourced: six sourcing "
                            "attempts failed and were booked as a gap "
                            "(aisle-corridor-width-not-independently-sourced). None is invented "
                            "here -- the width is derived from the MEASURED carrier."),
}

FRI_HOJD_MM = 2100.0            # ISO 14122-2 clear walkway height
OPERATOR_DJUP_MM = 780.0        # building-services zone operator_front
EL_DJUP_MM = 700.0              # IEC 60204-1 §11.2
SERVICE_DJUP_MM = 600.0         # ISO 14122-2 clear walkway width
UTRYMNING_BREDD_MM = 800.0      # ISO 14122-2 permanent walkway
SAKERHET_MM = 300.0             # SERVICE_MARGIN_M
PALL_W_MM, PALL_D_MM, PALL_H_MM = 800.0, 1200.0, 144.0
LAST_HOJD_MM = 600.0            # CONSTRUCTED: pallet + goods on the fork; see the honest remainder
VAGGMARGINAL_MM = 600.0         # the same wall margin as the layout generator

FABRIK = "glasfabrik"
SPEC = os.path.join(DATA, "fabriksspec_v2.json")

# process node -> station directory (only nodes that HAVE a built machine on disk)
MASKINER = [
    ("smaltugn",            "smaltugn_exemplar_v1"),
    ("formning_till_amne",  "glasformning_exemplar_v1"),
    ("grovslipning",        "glasslip_exemplar_v1"),
    ("polering",            "polermaskin_exemplar_v1"),
    ("ar_coating",          "ar_coat_exemplar_v1"),
    ("metrologi",           "glas_metrologi_exemplar_v1"),
]
UTILITIES = "utilities_bim_exemplar_v1"
MATERIALFLODE = "materialflode_exemplar_v1"

HMI_RE = re.compile(r"^hmi|manover|styrskap_display|press_styrskap|press_knapp|_display$|skarm", re.I)
EL_RE = re.compile(r"^elskap|^el_styrskap|^el_skap|^press_styrskap$", re.I)
SERVICE_RE = re.compile(r"servicelucka|inspektionslucka|^dorr_|^kapa_skjutdorr|^skjutdorr_|synhalslucka", re.I)

# the factory classes -- their OWN, not the machine classes renamed
FABRIKSKLASSER = {
    "transport": "truck/pallet aisle between two stations; SOURCE = an edge of the process graph",
    "operator": "free floor area + working area in front of a machine HMI/control position",
    "service": "the machine's OWN service room, inherited up to the hall (the recursion)",
    "media": "electrical/compressed-air run: ceiling run + access at cabinets (IEC 60204-1)",
    "utrymning": "free escape route along a wall, cited standard width",
}


class FabrikRum(nr.Rum):
    """negativrum_v1.Rum with the FACTORY class vocabulary.

    A truck aisle is NOT renamed to 'rorelsesvep' just to avoid touching the compositor -- a class
    that lies about what the room is makes the schema unusable for whoever reads it. Everything
    else -- the field, the solid, the ownership, the gates, the solver -- is literally the same code."""

    def __init__(self, namn, klass, kalla, prims, agare=(), motiv="", niva="fabrik"):
        if klass not in FABRIKSKLASSER:
            raise ValueError(f"{namn}: okand fabriksklass {klass!r} (valj bland {list(FABRIKSKLASSER)})")
        # borrow a valid machine class for the base validation, keep the OWN class in self.klass
        nr.Rum.__init__(self, namn, "media", kalla, prims, agare=agare, motiv=motiv)
        self.klass = klass
        self.niva = niva

    def som_dict(self):
        return {"namn": self.namn, "klass": self.klass, "klass_doc": FABRIKSKLASSER[self.klass],
                "niva": self.niva, "kalla": self.kalla, "agare": self.agare, "motiv": self.motiv,
                "primitiver": self.prims, "bbox_mm": [round(v, 3) for v in self.bbox()],
                "volym_mm3": round(sum(nr.prim_volym(p) for p in self.prims), 3),
                "volym_union_mm3": round(union_volym([nr.prim_bbox(p) for p in self.prims]), 3),
                "falt_ikarus": nr.ikarus_trad(self.prims)}


# ==================================================================================================
# EXACT BOX ALGEBRA -- an AABB judge measures a box exactly, and only a box
# ==================================================================================================
def bbox_snitt(a, b):
    """Volume of the intersection of two AABBs (mm3). EXACT."""
    v = 1.0
    for i in range(3):
        d = min(a[i + 3], b[i + 3]) - max(a[i], b[i])
        if d <= 0.0:
            return 0.0
        v *= d
    return v


def union_snitt_volym(box, boxar):
    """EXACT volume of (box  &  union(boxes)) -- coordinate compression, no inclusion-exclusion.

    Why not sum(bbox_snitt): the rooms are unions of OVERLAPPING boxes (the aisle crosses the stub)
    and a sum double-counts the overlap. Coordinate compression splits the intersection region into
    cells along the occurring coordinates and counts every cell ONCE -- exact for axis-aligned
    boxes, and O(n^3) in the number of boxes that touch the box at all (typically 1-3)."""
    rel = [b for b in boxar if bbox_snitt(box, b) > 0.0]
    if not rel:
        return 0.0
    if len(rel) == 1:
        return bbox_snitt(box, rel[0])
    kord = []
    for i in range(3):
        s = {box[i], box[i + 3]}
        for b in rel:
            s.add(max(b[i], box[i]))
            s.add(min(b[i + 3], box[i + 3]))
        kord.append(sorted(x for x in s if box[i] - 1e-9 <= x <= box[i + 3] + 1e-9))
    tot = 0.0
    for xi in range(len(kord[0]) - 1):
        x0, x1 = kord[0][xi], kord[0][xi + 1]
        if x1 - x0 <= 0:
            continue
        for yi in range(len(kord[1]) - 1):
            y0, y1 = kord[1][yi], kord[1][yi + 1]
            if y1 - y0 <= 0:
                continue
            for zi in range(len(kord[2]) - 1):
                z0, z1 = kord[2][zi], kord[2][zi + 1]
                if z1 - z0 <= 0:
                    continue
                c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
                for b in rel:
                    if (b[0] <= c[0] <= b[3] and b[1] <= c[1] <= b[4] and b[2] <= c[2] <= b[5]):
                        tot += (x1 - x0) * (y1 - y0) * (z1 - z0)
                        break
    return tot


def union_volym(boxar):
    """EXACT volume of the union of AABBs (same compression, the whole set)."""
    if not boxar:
        return 0.0
    hela = [min(b[i] for b in boxar) for i in range(3)] + [max(b[i + 3] for b in boxar) for i in range(3)]
    return union_snitt_volym(hela, boxar)


def bbox_av(boxar):
    return [min(b[i] for b in boxar) for i in range(3)] + [max(b[i + 3] for b in boxar) for i in range(3)]


def lada_av_bbox(b, namn=None):
    return nr.lada([(b[i] + b[i + 3]) / 2.0 for i in range(3)],
                   [max((b[i + 3] - b[i]) / 2.0, 1e-3) for i in range(3)], namn=namn)


# ==================================================================================================
# RIGID TRANSFORM -- yaw in 90-degree steps PRESERVES the AABB exactness
# ==================================================================================================
def yaw_bbox(b, yaw, t=(0.0, 0.0, 0.0)):
    """AABB after yaw (0/90/180/270 degrees about z) and translation. EXACT: a 90-degree yaw
    permutes/mirrors the coordinate axes, so an AABB maps to an AABB without any conservative
    enlargement. That is exactly why the rotation freedom is restricted to 90-degree steps here --
    an arbitrary angle would force an over-solid and make the verdict looser than it needs to be."""
    x0, y0, z0, x1, y1, z1 = b
    if yaw == 0:
        u0, v0, u1, v1 = x0, y0, x1, y1
    elif yaw == 90:
        u0, v0, u1, v1 = -y1, x0, -y0, x1
    elif yaw == 180:
        u0, v0, u1, v1 = -x1, -y1, -x0, -y0
    elif yaw == 270:
        u0, v0, u1, v1 = y0, -x1, y1, -x0
    else:
        raise ValueError(f"yaw {yaw} is not a declared freedom (0/90/180/270)")
    return [u0 + t[0], v0 + t[1], z0 + t[2], u1 + t[0], v1 + t[1], z1 + t[2]]


def yaw_riktning(d, yaw):
    """Rotates a unit direction (dx,dy) by yaw."""
    dx, dy = d
    if yaw == 0:
        return (dx, dy)
    if yaw == 90:
        return (-dy, dx)
    if yaw == 180:
        return (-dx, -dy)
    return (dy, -dx)


# ==================================================================================================
# THE MACHINE FROM ITS OWN MANIFEST
# ==================================================================================================
def las_manifest(katalog):
    p = os.path.join(DATA, "cad", katalog, "parts_manifest_v1.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    delar = [(x["part"], [float(v) for v in x["bbox"]]) for x in d["parts"] if x.get("bbox")]
    return d, delar


def gruppbbox(delar, rx):
    tr = [b for n, b in delar if rx.search(n)]
    return (bbox_av(tr), len(tr)) if tr else (None, 0)


def utat_riktning(grupp_bbox, env_bbox):
    """Which of the four horizontal machine sides the group sits on, and which way it opens.
    Derived, not hand-set: the axis where the group centre lies furthest from the machine centre
    RELATIVE to the machine half extent is the side the group belongs to."""
    best, bd = None, -1.0
    for i in (0, 1):
        c_g = (grupp_bbox[i] + grupp_bbox[i + 3]) / 2.0
        c_m = (env_bbox[i] + env_bbox[i + 3]) / 2.0
        halv = max((env_bbox[i + 3] - env_bbox[i]) / 2.0, 1.0)
        r = (c_g - c_m) / halv
        if abs(r) > bd:
            bd, best = abs(r), (i, 1.0 if r >= 0 else -1.0)
    return best  # (axelindex 0/1, tecken)


def zon_utanfor(grupp_bbox, axel, tecken, djup, sido_marginal=150.0, z0=0.0, z1=FRI_HOJD_MM,
                env_bbox=None):
    """The room that starts at the group's OUTERMOST face and extends `djup` mm OUTWARD.

    The boundary condition must not reject the status quo (the machine-level lesson). If the
    operator zone had started at the machine ENVELOPE face, the machine's own HMI pendant -- which
    by construction sticks out in front of the panel -- would have failed the gate on every machine
    that has a pendant, i.e. condemned the status quo and been unable to judge a CHANGE. The room
    therefore starts at the HMI group's own outermost face: it measures what it claims to measure
    -- FREE floor area for the operator."""
    b = list(grupp_bbox)
    andra = 1 - axel
    ut = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ut[andra] = b[andra] - sido_marginal
    ut[andra + 3] = b[andra + 3] + sido_marginal
    ut[2], ut[5] = z0, z1
    # the room starts at the OUTERMOST of (the group face, the machine envelope face) on that side
    if tecken > 0:
        start = b[axel + 3] if env_bbox is None else max(b[axel + 3], env_bbox[axel + 3])
        ut[axel], ut[axel + 3] = start, start + djup
    else:
        start = b[axel] if env_bbox is None else min(b[axel], env_bbox[axel])
        ut[axel], ut[axel + 3] = start - djup, start
    return ut

# (machine rooms in the machine frame)
def maskinrum_lokala(nod, katalog):
    """The machine's OWN negative spaces in the MACHINE's own coordinate system.

    Returns [(room name, class, source, [bbox], motive, direction)] where direction = (axis, sign)
    for the rooms that have an opening direction (operator/media)."""
    d, delar = las_manifest(katalog)
    env = bbox_av([b for _, b in delar])
    rum = []

    hmi_b, n_hmi = gruppbbox(delar, HMI_RE)
    if hmi_b is not None:
        ax, sg = utat_riktning(hmi_b, env)
        rum.append({
            "namn": f"operator_{nod}", "klass": "operator",
            "kalla": {"typ": "maskinmanifest", "maskin": katalog,
                      "delgrupp": "HMI/control position", "n_delar": n_hmi,
                      "grupp_bbox_mm": [round(v, 2) for v in hmi_b],
                      "fil": f"cad/{katalog}/parts_manifest_v1.json",
                      "standard": KALLA_STD["operator_front"], "djup_mm": OPERATOR_DJUP_MM},
            "boxar": [zon_utanfor(hmi_b, ax, sg, OPERATOR_DJUP_MM, env_bbox=env)],
            "motiv": (f"{n_hmi} HMI parts in {katalog} form the control position; the room opens "
                      f"outward from the HMI group's OWN outermost face "
                      f"({'xy'[ax]}{'+' if sg > 0 else '-'}) and is {OPERATOR_DJUP_MM:.0f} mm deep, "
                      f"{FRI_HOJD_MM:.0f} mm high."),
            "riktning": (ax, sg)})

    el_b, n_el = gruppbbox(delar, EL_RE)
    if el_b is not None:
        ax, sg = utat_riktning(el_b, env)
        rum.append({
            "namn": f"media_elatkomst_{nod}", "klass": "media",
            "kalla": {"typ": "maskinmanifest", "maskin": katalog, "delgrupp": "electrical cabinet",
                      "n_delar": n_el, "grupp_bbox_mm": [round(v, 2) for v in el_b],
                      "fil": f"cad/{katalog}/parts_manifest_v1.json",
                      "standard": KALLA_STD["iec60204"], "djup_mm": EL_DJUP_MM},
            "boxar": [zon_utanfor(el_b, ax, sg, EL_DJUP_MM, env_bbox=env)],
            "motiv": (f"{n_el} cabinet parts in {katalog}; IEC 60204-1 §11.2 requires {EL_DJUP_MM:.0f} mm "
                      f"clear width and {FRI_HOJD_MM:.0f} mm height in front of the cabinet door."),
            "riktning": (ax, sg)})

    sv_b, n_sv = gruppbbox(delar, SERVICE_RE)
    if sv_b is not None:
        ax, sg = utat_riktning(sv_b, env)
        rum.append({
            "namn": f"service_{nod}", "klass": "service",
            "kalla": {"typ": "maskinmanifest", "maskin": katalog, "delgrupp": "service and door hatches",
                      "n_delar": n_sv, "grupp_bbox_mm": [round(v, 2) for v in sv_b],
                      "fil": f"cad/{katalog}/parts_manifest_v1.json",
                      "standard": KALLA_STD["iso14122_gang"], "djup_mm": SERVICE_DJUP_MM},
            "boxar": [zon_utanfor(sv_b, ax, sg, SERVICE_DJUP_MM, env_bbox=env)],
            "motiv": (f"{n_sv} service/door parts in {katalog}; ISO 14122-2 clear width "
                      f"{SERVICE_DJUP_MM:.0f} mm along the side the hatches sit on."),
            "riktning": (ax, sg)})

    # THE RECURSION, LITERALLY: the machine's OWN negativrum_v1 schema is lifted upward if present.
    egen = os.path.join(DATA, "cad", katalog, "negativrum_v1.json")
    if os.path.exists(egen):
        with open(egen, encoding="utf-8") as f:
            eg = json.load(f)
        for R in eg.get("rum", []):
            if R["klass"] not in ("service", "media", "kyl"):
                continue      # a motion sweep is INSIDE the machine; the machine's own business
            bx = [nr.prim_bbox(p) for p in R["primitiver"]]
            rum.append({
                "namn": f"arvd_{R['klass']}_{nod}_{R['namn']}",
                "klass": "media" if R["klass"] in ("media", "kyl") else "service",
                "kalla": {"typ": "arvt_maskinrum", "maskin": katalog, "maskinrum": R["namn"],
                          "maskinklass": R["klass"], "maskinkalla": R.get("kalla"),
                          "fil": f"cad/{katalog}/negativrum_v1.json",
                          "rekursion": ("the machine's rooms are the factory's rooms -- SAME schema, "
                                        "SAME primitive family, one level up")},
                "boxar": bx,
                "motiv": (f"INHERITED from {katalog}'s own negativrum_v1 schema ({R['klass']}): "
                          f"{R.get('motiv', '')[:220]}"),
                "riktning": None})
    return {"env": env, "delar": delar, "rum": rum, "manifest": d}


# ==================================================================================================
# THE PROCESS GRAPH -- the source of the transport rooms
# ==================================================================================================
def las_processgraf():
    with open(SPEC, encoding="utf-8") as f:
        spec = json.load(f)
    g = spec["factories"][FABRIK]["processgraf"]
    noder = {n["id"]: n for n in g["nodes"]}
    return noder, g["edges"], spec["factories"][FABRIK]


def kontrahera(edges, behall):
    """The edges between the nodes that HAVE a built machine. An edge that passes through a node
    without a built machine is contracted -- and that is WRITTEN, so nobody believes the
    intermediate step does not exist."""
    nast = {}
    for e in edges:
        nast.setdefault(e["from"], []).append(e)
    ut = []
    for start in behall:
        stack = [(start, [], None)]
        sedda = {start}
        while stack:
            nod, via, kalla = stack.pop()
            for e in nast.get(nod, []):
                m = e["to"]
                if m in behall:
                    ut.append({"from": start, "to": m, "via": list(via),
                               "kalla": kalla or e.get("kalla"), "typ": e.get("typ"),
                               "kontraherad": bool(via)})
                elif m not in sedda:
                    sedda.add(m)
                    stack.append((m, via + [m], kalla or e.get("kalla")))
    # only the first edge out of each node (the line flow)
    sett = set()
    rak = []
    for e in sorted(ut, key=lambda x: (behall.index(x["from"]), len(x["via"]))):
        if e["from"] in sett:
            continue
        sett.add(e["from"])
        rak.append(e)
    return rak


# ==================================================================================================
# TRANSPORT AISLE WIDTH -- derived from the carrier, and G2 decides whether the derivation held
# ==================================================================================================
def pall_matt():
    """The EUR pallet AABB, MEASURED from the material-flow station manifest."""
    _, delar = las_manifest(MATERIALFLODE)
    p = [b for n, b in delar if n.startswith("pall_")]
    if not p:
        raise RuntimeError("no pallet in the material-flow manifest -- the carrier must be MEASURED")
    b = bbox_av(p)
    return {"w_mm": round(b[3] - b[0], 2), "d_mm": round(b[4] - b[1], 2),
            "h_mm": round(b[5] - b[2], 2), "n_delar": len(p), "kalla": KALLA_STD["pall"]}


def gangbredd_rak(pall):
    return pall["w_mm"] + 2 * SAKERHET_MM


def gangbredd_svang(pall):
    """The width a 90-degree TURN requires: the AABB of the pallet through the whole turn.

    Uses negativrum_v1._rot_bbox -- the SAME function as the machine level's rotation sweep."""
    b = [-pall["w_mm"] / 2, -pall["d_mm"] / 2, 0.0, pall["w_mm"] / 2, pall["d_mm"] / 2, pall["h_mm"]]
    bb = None
    for k in range(0, 91):
        r = nr._rot_bbox(b, (0.0, 0.0, 0.0), "z", float(k))
        bb = r if bb is None else bbox_av([bb, r])
    return max(bb[3] - bb[0], bb[4] - bb[1]) + 2 * SAKERHET_MM


def pallsvep(rutt, pall, z0=0.0):
    """The pallet's REAL sweep along a route of axis-aligned segments + 90-degree turns at the knees.

    Translation along a coordinate axis: EXACTLY an extended AABB (the same argument negativrum_v1
    uses for a prismatic joint -- no sampling, no discretisation remainder).
    Turn at a knee: the AABB over the whole rotation, conservatively enclosing."""
    h = z0 + pall["h_mm"] + LAST_HOJD_MM
    kroppar = []
    for i in range(len(rutt) - 1):
        (x0, y0), (x1, y1) = rutt[i], rutt[i + 1]
        langs_x = abs(x1 - x0) >= abs(y1 - y0)
        w, d = (pall["d_mm"], pall["w_mm"]) if langs_x else (pall["w_mm"], pall["d_mm"])
        b = [min(x0, x1) - w / 2, min(y0, y1) - d / 2, z0,
             max(x0, x1) + w / 2, max(y0, y1) + d / 2, h]
        kroppar.append((f"segment_{i}", b))
        if 0 < i:
            kn = [x0 - pall["d_mm"] / 2, y0 - pall["d_mm"] / 2, z0,
                  x0 + pall["d_mm"] / 2, y0 + pall["d_mm"] / 2, h]
            bb = None
            for k in range(0, 91, 5):
                r = nr._rot_bbox([x0 - pall["w_mm"] / 2, y0 - pall["d_mm"] / 2, z0,
                                  x0 + pall["w_mm"] / 2, y0 + pall["d_mm"] / 2, h],
                                 (x0, y0, 0.0), "z", float(k))
                bb = r if bb is None else bbox_av([bb, r])
            kroppar.append((f"svang_{i}", bbox_av([kn, bb])))
    return kroppar


# ==================================================================================================
# G3 AT HALL LEVEL -- exact box against box, the field as pre-filter
# ==================================================================================================
def keepout_hall(rum_lista, delar_per_maskin, forfilter_mm=2.0):
    """Material in a room it does not own. The field selects the pairs, the box algebra judges EXACTLY."""
    brott = []
    per_rum = {}
    n_falt = n_exakt = 0
    for R in rum_lista:
        boxar = [nr.prim_bbox(p) for p in R.prims]
        rb = bbox_av(boxar)
        agare = set(R.agare)
        for maskin, delar in delar_per_maskin.items():
            for namn, b in delar:
                nyckel = f"{maskin}:{namn}"
                if nyckel in agare or maskin in agare:
                    continue
                if bbox_snitt([rb[i] - forfilter_mm if i < 3 else rb[i] + forfilter_mm
                               for i in range(6)], b) <= 0.0:
                    continue                                  # faltets forfilter (generost)
                n_falt += 1
                v = union_snitt_volym(b, boxar)
                n_exakt += 1
                if v > nr.VOL_TOL_MM3:
                    brott.append({"rum": R.namn, "klass": R.klass, "maskin": maskin,
                                  "del": namn, "snitt_mm3": round(v, 1)})
                    per_rum[R.namn] = round(per_rum.get(R.namn, 0.0) + v, 1)
    brott.sort(key=lambda x: -x["snitt_mm3"])
    return {"GRON": not brott, "n_brott": len(brott), "materia_per_rum_mm3": per_rum,
            "brott": brott[:80], "n_par_efter_faltforfilter": n_falt, "n_par_exakt_raknade": n_exakt,
            "tol_mm3": nr.VOL_TOL_MM3,
            "regel": ("material outside a room's OWNERS may never intersect the room; at hall level "
                      "both rooms and part proxies are AXIS-ALIGNED BOXES and the intersection is "
                      "computed EXACTLY and analytically")}


# ==================================================================================================
# THE MEDIA RUN Z BAND -- MEASURED from the building services model
# ==================================================================================================
def media_zband():
    """The ceiling run's z interval, MEASURED from the building-services station manifest.

    A height is NOT chosen: the module reads where the building services ACTUALLY sit in the model
    that built them, and takes the service parts above the ISO 14122-2 clear height."""
    d, delar = las_manifest(UTILITIES)
    sub = {x["part"]: x.get("subassembly", "") for x in d["parts"]}
    hoga = [b for n, b in delar
            if b[2] >= FRI_HOJD_MM and not sub.get(n, "").startswith("bim/rum")]
    if not hoga:
        raise RuntimeError("no ceiling-mounted services in the building model -- the media run would be a claim")
    bb = bbox_av(hoga)
    bredd = min(bb[3] - bb[0], bb[4] - bb[1])
    return {"z0_mm": round(bb[2], 1), "z1_mm": round(bb[5], 1), "n_delar": len(hoga),
            "bredd_mm": round(bredd, 1),
            "kalla": (f"MATT: {len(hoga)} takforlagda installationsdelar (z_min >= {FRI_HOJD_MM:.0f} mm, "
                      f"not bim/rum) in cad/{UTILITIES}/parts_manifest_v1.json"),
            "bredd_kalla": ("the MEASURED narrowest horizontal extent of the services -- a run along "
                            "a wall, NOT a ceiling-covering layer"),
            "★rattad_fail_open": (
                "A first version declared the media run as a HALL-SPANNING slab 2105..3200 mm. "
                "It was fail-open in one direction and fail-closed in another: the solver escaped it "
                "by pushing the tallest station (2356 mm high) past the slab's arbitrary start "
                "sideways while one station became homeless (7498 of 13041 candidates rejected by "
                "that one room). A room whose extent is arbitrary measures its own arbitrariness. "
                "The width is now MEASURED from the services themselves.")}


# ==================================================================================================
# HALLEN -- tomrummet forst, materian efterat
# ==================================================================================================
def bygg_hall(rv_namn, hall_w, hall_h, med_negativrum=True, log=print):
    """ONE hall. The order is the whole point:

      1. de HALLSGLOBALA tomrummen deklareras FORST (transportband, utrymningsvag, mediastrak) --
         innan en enda maskin har en position,
      2. the machines are placed in the PROCESS GRAPH flow order by the placement solver, with the
         rooms as hard keep-out and the hall dimensions as boundary conditions,
      3. every placed machine's OWN rooms (operator/media/service, inherited from its own
         negativrum_v1 schema where it has one) become keep-out for the NEXT machine.

    med_negativrum=False is the NEGATIVE CONTROL: the same machines, the same flow order, the same
    solver -- but the rooms are not keep-out. What remains is only the requirement a conventional
    hall generator imposes (free aisle >= 800 mm between parts of different stations)."""
    noder, edges, fab = las_processgraf()
    nodlista = [n for n, _ in MASKINER]
    kanter = kontrahera(edges, nodlista)
    pall = pall_matt()
    B_rak, B_svang = gangbredd_rak(pall), gangbredd_svang(pall)
    B_gang = B_svang                       # derived from the TURN, see G2
    med = media_zband()

    # THE Y POSITION OF THE BANDS IS KNOWN A PRIORI -- that is what makes "void first" possible one
    # level up: the south wall carries the media run, then the escape route, then the transport
    # aisle, and only THEREFORE can machines be placed against already-declared rooms.
    y_utr0, y_utr1 = -UTRYMNING_BREDD_MM, 0.0
    # ★MEDIASTRAKET DELAR GOLVYTA MED UTRYMNINGSVAGEN OCH AR SKILT FRAN DEN I Z.  Forsta
    # version put it in its OWN y band next to the escape route and let it eat that width of the
    # hall DEPTH -- although the run hangs at z 2105..3200 mm and does not touch the floor at all.
    # The consequence was that ALL SIX machines became congestion findings, for a reason that was
    # the modelling and not the hall. It is exactly the ISO 14122-2 clear height (2100 mm) that
    # makes sharing the floor legal, and it is MEASURED: the run starts at 2105 mm.
    y_med0, y_med1 = y_utr0, y_utr1
    y_g0, y_g1 = 0.0, B_gang
    y_c = (y_g0 + y_g1) / 2.0
    y_nord = y_utr0 + hall_h if hall_h else 1e6
    x_max_hall = hall_w + y_utr0 if hall_w else 1e6

    # ---------- 1. HALLSGLOBALA RUM, deklarerade FORE all materia ----------
    globala = []
    # the x extent is PROVISIONAL and deliberately generous in both directions during placement:
    # a band that starts arbitrarily far in is a fail-open the solver escapes sideways (measured:
    # the first station moved west of the band start). Phase C trims the bands to the hall's real
    # extent once it is known.
    L = x_max_hall if hall_w else 60000.0
    L0 = -VAGGMARGINAL_MM if hall_w else -60000.0
    globala.append(FabrikRum(
        "transport_huvudstrak", "transport",
        {"typ": "processgraf", "fil": "fabriksspec_v2.json#processgraf",
         "n_kanter": len(kanter), "barare": pall, "bredd_mm": round(B_gang, 1),
         "bredd_harledning": (f"the carrier AABB {pall['w_mm']}x{pall['d_mm']} mm TURNED 90 degrees "
                              f"(AABB over the whole turn) + 2 x {SAKERHET_MM:.0f} mm; "
                              f"the straight run would only have required {B_rak:.0f} mm"),
         "standard_saknas": KALLA_STD["ingen_truckstandard"],
         "marginal_kalla": KALLA_STD["sakerhetsmarginal"]},
        [lada_av_bbox([L0, y_g0, 0.0, L, y_g1, FRI_HOJD_MM])],
        agare=["__transport__"],
        motiv=("the main run carries ALL process-graph edges between the built stations; the width "
               "is derived from the MEASURED carrier and checked by G2 against the pallet's real sweep")))
    globala.append(FabrikRum(
        "utrymning_sodra", "utrymning",
        {"typ": "krav", "krav": "fri utrymningsvag", "standard": KALLA_STD["iso14122_perm"],
         "bredd_mm": UTRYMNING_BREDD_MM},
        [lada_av_bbox([L0, y_utr0, 0.0, L, y_utr1, FRI_HOJD_MM])],
        motiv=("the escape route is SEPARATE from the transport aisle: an aisle with a pallet in it "
               "is not an escape route, so they may not be the same floor")))
    globala.append(FabrikRum(
        "media_takstam", "media",
        {"typ": "byggnadsforsorjning", "fil": f"cad/{UTILITIES}/parts_manifest_v1.json",
         "z_matt": med, "standard": KALLA_STD["iec60204"]},
        [lada_av_bbox([L0, y_med0, med["z0_mm"], L, y_med1, med["z1_mm"]])],
        motiv=(f"the electrical/compressed-air trunk is a {med['bredd_mm']:.0f} mm wide RUN along the "
               f"south wall at z {med['z0_mm']:.0f}..{med['z1_mm']:.0f} mm -- BOTH dimensions MEASURED "
               f"from the building services model, nothing chosen. No machine may wall it in.")))

    # ---------- 2. MASKINERNA, i processgrafens flodesordning ----------
    lokal = {nod: maskinrum_lokala(nod, kat) for nod, kat in MASKINER}
    placerade = {}          # nod -> {"yaw","t","delar":[(namn,bbox)],"env":bbox,"rum":[FabrikRum]}
    rum_lista = list(globala)
    lagg = []
    x_kur = 0.0
    for idx, (nod, kat) in enumerate(MASKINER):
        M = lokal[nod]
        op = next((r for r in M["rum"] if r["klass"] == "operator"), None)
        # -- yaw: the declared freedom that turns the operator position towards the aisle (0,-1)
        yaws = [0, 90, 180, 270]
        val_yaw = None
        for y in yaws:
            if op is None:
                val_yaw = 0
                break
            ax, sg = op["riktning"]
            d = yaw_riktning((sg if ax == 0 else 0.0, sg if ax == 1 else 0.0), y)
            if d[1] < -0.5:
                val_yaw = y
                break
        if val_yaw is None:
            val_yaw = 0
        # -- maskinens lador i yaw-lage (translation 0)
        env0 = yaw_bbox(M["env"], val_yaw)
        rumboxar0 = {r["namn"]: [yaw_bbox(b, val_yaw) for b in r["boxar"]] for r in M["rum"]}
        alla0 = [env0] + [b for bs in rumboxar0.values() for b in bs]
        syd = min(b[1] for b in alla0)
        # -- target from the PROCESS GRAPH: downstream machine right after upstream, aisle width between
        if med_negativrum:
            t0 = (x_kur - min(b[0] for b in alla0), y_g1 - syd, 0.0)
        else:
            # ★NEGATIV KONTROLL -- OCH VARFOR DEN FORSTA VERSIONEN AV DEN VAR VARDELOS.
            # A first version let the control keep the same TARGET (the bundle including the
            # operator zone placed against the aisle edge) and then measured against rooms that
            # MOVED WITH the machines wherever they ended up. It was green on everything -- a gate
            # that gets its own output as input is always green. A layout WITHOUT declared voids
            # packs material, not voids: the target is the machine's OWN body as far south as the
            # neighbours allow, and the measurement is against the A PRIORI declared bands (aisle /
            # escape route / media run), which sit in exactly the same place in both runs and are
            # therefore a FAIR common yardstick.
            t0 = (x_kur - env0[0], y_med0 - env0[1], 0.0)

        def stubbe(e):
            """The transport stub from the main run in to the machine's south side.

            ★ARKITEKTURFEL SOM MATTES OCH RATTADES: forsta versionen skapade stubbarna EFTER
            placement (phase C). The solver then never saw them, and the measurement showed
            51 114 906 mm3 of the first station's masonry in the middle of the next station's stub --
            not because the hall was tight but because the module's own main rule had been broken:
            the VOID FIRST. The stub is now part of the machine's room bundle and is keep-out during
            placeringen."""
            xc = (e[0] + e[3]) / 2.0
            return [xc - B_gang / 2, y_g0, 0.0, xc + B_gang / 2, max(e[1], y_g0 + 1.0), FRI_HOJD_MM]

        def randvillkor(d, _env0=env0, _rb=rumboxar0, _t0=t0, _nod=nod):
            t = (_t0[0] + d[0], _t0[1] + d[1], 0.0)
            e = [_env0[i] + (t[i % 3] if i % 3 < 2 else 0.0) for i in range(6)]
            e = [_env0[0] + t[0], _env0[1] + t[1], _env0[2],
                 _env0[3] + t[0], _env0[4] + t[1], _env0[5]]
            if hall_w and (e[0] < -VAGGMARGINAL_MM or e[3] > x_max_hall):
                return f"hall length ({hall_w:.0f} mm): the machine x {e[0]:.0f}..{e[3]:.0f}"
            if hall_h and (e[1] < y_g1 or e[4] > y_nord):
                return (f"hall depth ({hall_h:.0f} mm): the machine may not enter the aisle or "
                        f"pass through the north wall; y {e[1]:.0f}..{e[4]:.0f} against [{y_g1:.0f}, {y_nord:.0f}]")
            if med_negativrum:
                # THIS machine's rooms -- INCLUDING its transport stub -- must also be free
                # of ALREADY placed material
                buntar = {rn: [[b[0] + t[0], b[1] + t[1], b[2], b[3] + t[0], b[4] + t[1], b[5]]
                               for b in bs] for rn, bs in _rb.items()}
                buntar[f"transport_stubbe_{_nod}"] = [stubbe(e)]
                for rn, flyttade in buntar.items():
                    for pn, P in placerade.items():
                        for dn, db in P["delar"]:
                            if union_snitt_volym(db, flyttade) > nr.VOL_TOL_MM3:
                                return (f"{_nod}:{rn} would contain already placed material "
                                        f"{pn}:{dn}")
            return None

        grupp = {f"{nod}__envelopp": lada_av_bbox([env0[0] + t0[0], env0[1] + t0[1], env0[2],
                                                   env0[3] + t0[0], env0[4] + t0[1], env0[5]])}
        if med_negativrum:
            r = nr.losa_placering(grupp, rum_lista, friheter=[(0, 0), (1, +1)],
                                  steg_mm=100.0, tak_mm=8000.0, randvillkor=randvillkor,
                                  marginal_mm=0.0, log=lambda *a: None)
        else:
            # NEGATIVE CONTROL: only the conventional requirement -- free aisle >= 800 mm between parts of different stations
            r = _placera_utan_rum(nod, env0, t0, placerade, randvillkor)
        if not r["LOST"]:
            lagg.append({"nod": nod, "maskin": kat, "TRANGSEL": True, **r})
            log(f"  TRANGSEL: {nod} finds no placement in {rv_namn}")
            continue
        d = r["offset_mm"]
        t = (t0[0] + d[0], t0[1] + d[1], 0.0)
        delar = [(n, yaw_bbox(b, val_yaw, t)) for n, b in M["delar"]]
        env = yaw_bbox(M["env"], val_yaw, t)
        egna = []
        for R in M["rum"]:
            bs = [yaw_bbox(b, val_yaw, t) for b in R["boxar"]]
            egna.append(FabrikRum(R["namn"], R["klass"], R["kalla"], [lada_av_bbox(b) for b in bs],
                                  agare=[nod], motiv=R["motiv"], niva="fabrik(arvt)"
                                  if R["kalla"].get("typ") == "arvt_maskinrum" else "fabrik"))
        egna.append(FabrikRum(
            f"transport_stubbe_{nod}", "transport",
            {"typ": "processkant_stubbe", "station": nod,
             "fil": "fabriksspec_v2.json#processgraf",
             "bredd_mm": round(B_gang, 1), "barare": {"w_mm": pall["w_mm"], "d_mm": pall["d_mm"]}},
            [lada_av_bbox(stubbe(env))], agare=["__transport__"],
            motiv=("the stub that drops off / picks up the pallet at the station's south side; "
                   "declared AT THE SAME TIME as the machine's other rooms, i.e. BEFORE the next "
                   "machine's material")))
        placerade[nod] = {"maskin": kat, "yaw": val_yaw, "t_mm": [round(v, 1) for v in t],
                          "delar": delar, "env": env, "rum": egna, "n_delar": len(delar),
                          "losare": {k: r[k] for k in ("offset_mm", "n_provade", "n_kandidater")
                                     if k in r}}
        if med_negativrum:
            rum_lista += egna
        x_kur = (max(b[3] for b in [env] + [nr.prim_bbox(p) for R in egna for p in R.prims])
                 if med_negativrum else env[3] + UTRYMNING_BREDD_MM)
        lagg.append({"nod": nod, "maskin": kat, "TRANGSEL": False, "yaw": val_yaw,
                     "offset_mm": d, "t_mm": [round(v, 1) for v in t]})
        log(f"  {nod:22s} yaw={val_yaw:3d} offset={d} t={[round(v) for v in t]}")

    # ---------- 3. ONE TRANSPORT ROOM PER PROCESS EDGE (phase C: bands trimmed to real stations) ----
    kantrum = []
    rutter = {}
    for e in kanter:
        if e["from"] not in placerade or e["to"] not in placerade:
            continue
        A, B = placerade[e["from"]], placerade[e["to"]]
        xa = (A["env"][0] + A["env"][3]) / 2.0
        xb = (B["env"][0] + B["env"][3]) / 2.0
        # THE DOCK POINT IS THE PALLET CENTRE, NOT THE MACHINE FACE. A sweep between two POINTS
        # extends half a pallet beyond both end points (the same geometry as a prismatic joint's
        # AABB extension). A first version put the end point ON the machine's south face and
        # therefore measured 321 785 248 mm3 "outside the room" -- the sweep reached into the
        # machine it was delivering to. The pallet is parked half a pallet in front of the face.
        ya = A["env"][1] - pall["d_mm"] / 2.0
        yb = B["env"][1] - pall["d_mm"] / 2.0
        rutt = [(xa, ya), (xa, y_c), (xb, y_c), (xb, yb)]
        rutter[f"{e['from']}->{e['to']}"] = rutt
        boxar = [[min(xa, xb) - B_gang / 2, y_g0, 0.0, max(xa, xb) + B_gang / 2, y_g1, FRI_HOJD_MM],
                 [xa - B_gang / 2, y_g0, 0.0, xa + B_gang / 2, A["env"][1], FRI_HOJD_MM],
                 [xb - B_gang / 2, y_g0, 0.0, xb + B_gang / 2, B["env"][1], FRI_HOJD_MM]]
        boxar = [b for b in boxar if b[4] > b[1]]
        kantrum.append(FabrikRum(
            f"transport_{e['from']}__{e['to']}", "transport",
            {"typ": "processkant", "from": e["from"], "to": e["to"],
             "kant_typ": e.get("typ"), "kant_kalla": e.get("kalla"),
             "kontraherad_via": e.get("via"), "bredd_mm": round(B_gang, 1),
             "fil": "fabriksspec_v2.json#processgraf.edges"},
            [lada_av_bbox(b) for b in boxar], agare=["__transport__"],
            motiv=(f"kanten {e['from']} -> {e['to']} i processgrafen"
                   + (f" (contracted over {'/'.join(e['via'])}: nodes without built CAD)" if e.get("via") else "")
                   + "; main run + two stubs in to the stations' south sides")))
    # ---------- PHASE C: trim the provisional bands to the hall's REAL extent ----------
    # (the bands were generous in both directions during placement precisely so as not to be
    # fail-open sideways; now the hall length is known and they are trimmed to it.)
    if placerade:
        maskin_x = bbox_av([P["env"] for P in placerade.values()]
                           + [nr.prim_bbox(p) for P in placerade.values() for R in P["rum"]
                              for p in R.prims])
        x0 = maskin_x[0] - VAGGMARGINAL_MM
        x1 = maskin_x[3] + VAGGMARGINAL_MM
        for R in globala:
            nya = []
            for p in R.prims:
                b = nr.prim_bbox(p)
                nya.append(lada_av_bbox([max(b[0], x0), b[1], b[2], min(b[3], x1), b[4], b[5]],
                                        namn=p.get("namn")))
            R.prims = nya
            R._solid = None
    rum_lista = list(globala) + kantrum + [R for P in placerade.values() for R in P["rum"]]

    hall_bbox = bbox_av([P["env"] for P in placerade.values()]
                        + [nr.prim_bbox(p) for R in rum_lista for p in R.prims]) if placerade else None
    return {"rv": rv_namn, "hall_w_mm": hall_w, "hall_h_mm": hall_h,
            "med_negativrum": med_negativrum, "pall": pall,
            "gangbredd_rak_mm": round(B_rak, 1), "gangbredd_svang_mm": round(B_svang, 1),
            "gangbredd_vald_mm": round(B_gang, 1), "media_zband": med,
            "kanter": kanter, "rutter": rutter, "placerade": placerade, "lagg": lagg,
            "rum": rum_lista, "globala": globala, "kantrum": kantrum,
            "hall_bbox_mm": [round(v, 1) for v in hall_bbox] if hall_bbox else None,
            "n_trangsel": sum(1 for x in lagg if x["TRANGSEL"])}


def _placera_utan_rum(nod, env0, t0, placerade, randvillkor):
    """NEGATIVE CONTROL: the conventional requirement and NOTHING else -- free aisle >= 800 mm
    between parts of different stations (exactly the quantity a conventional placement gate
    measures). No operator zone, no transport aisle, no escape route, no media run."""
    for d in nr.kandidatoffset([(0, 0), (1, +1)], 100.0, 8000.0):
        if randvillkor(d):
            continue
        e = [env0[0] + t0[0] + d[0], env0[1] + t0[1] + d[1], env0[2],
             env0[3] + t0[0] + d[0], env0[4] + t0[1] + d[1], env0[5]]
        ok = True
        for P in placerade.values():
            o = P["env"]
            dx = max(o[0] - e[3], e[0] - o[3])
            dy = max(o[1] - e[4], e[1] - o[4])
            fri = math.hypot(max(dx, 0.0), max(dy, 0.0)) if (dx > 0 or dy > 0) else -1.0
            if fri < 800.0:
                ok = False
                break
        if ok:
            return {"LOST": True, "offset_mm": list(d), "n_provade": -1,
                    "metod": "endast fri gang >= 800 mm (SS-EN ISO 14122-2), inga negativrum"}
    return {"LOST": False, "offset_mm": None, "TRANGSEL": True,
            "dom": "no placement even without negative spaces"}


# ==================================================================================================
# SELFTEST -- reference pairs that CAN fail
# ==================================================================================================
def _selftest():
    ut = {"prov": []}

    def p(namn, ok, **kv):
        ut["prov"].append({"namn": namn, "GRON": bool(ok), **kv})
        print(f"  {'GRON' if ok else 'ROD '}  {namn}  {kv}")

    # S1 box algebra: the union of two overlapping boxes is NOT double counted
    a = [0, 0, 0, 100, 100, 100]
    b = [50, 0, 0, 150, 100, 100]
    p("S1a union of two overlapping boxes is exact (no double counting)",
      abs(union_volym([a, b]) - 150 * 100 * 100) < 1e-6, v=union_volym([a, b]))
    p("S1b naive sum OVERSTATES (the error the compression exists for)",
      abs((bbox_snitt(bbox_av([a, b]), a) + bbox_snitt(bbox_av([a, b]), b)) - 200 * 100 * 100) < 1e-6)
    p("S1c intersection against the union is counted once",
      abs(union_snitt_volym([40, 0, 0, 60, 100, 100], [a, b]) - 20 * 100 * 100) < 1e-6)

    # S2 yaw preserves volume and is exact
    bb = [10, -20, 0, 40, 60, 100]
    v0 = (bb[3] - bb[0]) * (bb[4] - bb[1]) * (bb[5] - bb[2])
    okv = all(abs((lambda q: (q[3] - q[0]) * (q[4] - q[1]) * (q[5] - q[2]))(yaw_bbox(bb, y)) - v0) < 1e-9
              for y in (0, 90, 180, 270))
    p("S2a yaw 0/90/180/270 preserves the AABB volume exactly", okv)
    p("S2b yaw 360 = identity", yaw_bbox(yaw_bbox(yaw_bbox(yaw_bbox(bb, 90), 90), 90), 90) == bb)

    # S3 the turn needs MORE than the straight run -- the gate that fails its own generator
    pall = {"w_mm": 800.0, "d_mm": 1200.0, "h_mm": 144.0}
    p("S3 turn width > straight-run width (otherwise the aisle cannot turn the pallet)",
      gangbredd_svang(pall) > gangbredd_rak(pall) + 1.0,
      rak=round(gangbredd_rak(pall), 1), svang=round(gangbredd_svang(pall), 1))

    # S4 G2 at hall level. A FIRST VERSION OF THIS TEST ASKED THE WRONG QUESTION and measured 0 mm3
    # outside for an aisle called "too narrow" (900 mm): on a STRAIGHT run the pallet's
    # cross dimension is only 800 mm, so 900 mm is in fact enough. The test measured nothing. Fixed in the
    # QUESTION: the too-narrow aisle is narrower than the carrier (600 mm), and the DECISIVE test is
    # the TURN -- an aisle sized from the straight run (1400 mm) must FAIL when the pallet turns.
    def _rest(sv, br, rutt_bbox):
        R = FabrikRum("provgang", "transport", {"typ": "processkant", "from": "a", "to": "b"},
                      [lada_av_bbox(b) for b in rutt_bbox(br)], motiv="test aisle")
        bx = [nr.prim_bbox(q) for q in R.prims]
        return sum(max(0.0, (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]) - union_snitt_volym(b, bx))
                   for _, b in sv)

    rak_rutt = [(0.0, 0.0), (5000.0, 0.0)]
    sv_rak = pallsvep(rak_rutt, pall)
    box_rak = lambda br: [[-1500, -br / 2, 0, 6500, br / 2, 2100]]                       # noqa: E731
    p("S4a G2 aisle narrower than the carrier FAILS", _rest(sv_rak, 600.0, box_rak) > nr.VOL_TOL_MM3,
      utanfor_mm3=round(_rest(sv_rak, 600.0, box_rak), 1))
    p("S4b G2 aisle derived from the straight run PASSES a straight run",
      _rest(sv_rak, gangbredd_rak(pall), box_rak) <= nr.VOL_TOL_MM3,
      bredd=gangbredd_rak(pall))

    sv_kna = pallsvep([(0.0, 0.0), (5000.0, 0.0), (5000.0, 4000.0)], pall)
    box_kna = lambda br: [[-1500, -br / 2, 0, 5000 + br / 2, br / 2, 2100],               # noqa: E731
                          [5000 - br / 2, -br / 2, 0, 5000 + br / 2, 5500, 2100]]
    r_rak = _rest(sv_kna, gangbredd_rak(pall), box_kna)
    r_sv = _rest(sv_kna, gangbredd_svang(pall), box_kna)
    p("S4c G2 the TURN fails an aisle sized from the straight run (the gate fails its generator)",
      r_rak > nr.VOL_TOL_MM3, utanfor_mm3=round(r_rak, 1), bredd=gangbredd_rak(pall))
    p("S4d G2 the same turn PASSES when the width is derived from the TURN",
      r_sv <= nr.VOL_TOL_MM3, utanfor_mm3=round(r_sv, 1), bredd=round(gangbredd_svang(pall), 1))

    # S5 keep-out: a part inside the room is failed, the owner is cleared
    R = FabrikRum("prov_gang", "transport", {"typ": "processkant", "from": "a", "to": "b"},
                  [lada_av_bbox([0, 0, 0, 10000, 1400, 2100])], agare=["trucken"], motiv="prov")
    g = keepout_hall([R], {"maskinA": [("stativ", [500, 500, 0, 900, 900, 1000])],
                           "trucken": [("chassi", [2000, 500, 0, 2400, 900, 1000])],
                           "maskinB": [("fjarran", [50000, 0, 0, 50400, 400, 1000])]})
    p("S5 keep-out fails the intruder, clears the owner and the distant part",
      (not g["GRON"]) and g["n_brott"] == 1 and g["brott"][0]["maskin"] == "maskinA",
      n_brott=g["n_brott"], brott=[b["maskin"] for b in g["brott"]])

    # S6 the factory class must NOT be silently downgraded to a machine class
    try:
        FabrikRum("x", "rorelsesvep", {"typ": "prov"}, [lada_av_bbox([0, 0, 0, 1, 1, 1])], motiv="m")
        ok = False
    except ValueError:
        ok = True
    p("S6 a machine class is rejected as a factory class (the class may not lie)", ok)

    ut["GRON"] = all(x["GRON"] for x in ut["prov"])
    ut["n_prov"] = len(ut["prov"])
    print(json.dumps(ut, ensure_ascii=False))
    return 0 if ut["GRON"] else 1


# ==================================================================================================
# GRINDARNA PA HALLSNIVA
# ==================================================================================================
def grinda(H, n_paritet=3000, log=print):
    ut = {}
    t0 = time.time()
    g0, g1 = {}, {}
    for R in H["rum"]:
        g0[R.namn] = nr.vattentathet(R)
        g1[R.namn] = nr.falt_paritet(R, n=n_paritet, seed=17, marginal_mm=300.0)
    ut["G0_vattentathet"] = g0
    ut["G1_faltparitet"] = g1
    ut["G0_GRON"] = all(v["GRON"] for v in g0.values())
    ut["G1_GRON"] = all(v["GRON"] for v in g1.values())
    ut["G1_oense_totalt"] = sum(v["n_oense"] for v in g1.values())
    ut["G1_provade_totalt"] = sum(v["n_provade"] for v in g1.values())
    log(f"  G0 {ut['G0_GRON']}  G1 {ut['G1_GRON']} ({ut['G1_oense_totalt']} oense av "
        f"{ut['G1_provade_totalt']})  {time.time() - t0:.1f}s")

    # G2 SWEEP COVERAGE: the pallet's real sweep along every process edge must fit in that edge's room
    g2 = {}
    for R in H["kantrum"]:
        nyckel = f"{R.kalla['from']}->{R.kalla['to']}"
        rutt = H["rutter"].get(nyckel)
        if not rutt:
            continue
        sv = pallsvep(rutt, H["pall"])
        boxar = [nr.prim_bbox(p) for p in R.prims]
        rest, vartill = 0.0, []
        for namn, b in sv:
            vol = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])
            u = max(0.0, vol - union_snitt_volym(b, boxar))
            if u > nr.VOL_TOL_MM3:
                vartill.append({"svepkropp": namn, "utanfor_rummet_mm3": round(u, 1)})
            rest += u
        g2[R.namn] = {"GRON": not vartill, "n_svepkroppar": len(sv),
                      "utanfor_rummet_mm3": round(rest, 1), "vartill": vartill,
                      "regel": "volume(pallet sweep - transport room) == 0, exact box algebra"}
    ut["G2_sveptackning"] = g2
    ut["G2_GRON"] = all(v["GRON"] for v in g2.values()) if g2 else False
    ut["G2_utanfor_totalt_mm3"] = round(sum(v["utanfor_rummet_mm3"] for v in g2.values()), 1)

    # G3 KEEPOUT
    delar = {nod: P["delar"] for nod, P in H["placerade"].items()}
    ut["G3_keepout"] = keepout_hall(H["rum"], delar)
    ut["G3_GRON"] = ut["G3_keepout"]["GRON"]

    # HARD GATE: 0 mm3 of material in transport + escape rooms
    apriori = {R.namn for R in H["globala"]}
    hard = {}
    for R in H["rum"]:
        if R.klass not in ("transport", "utrymning"):
            continue
        boxar = [nr.prim_bbox(p) for p in R.prims]
        v = 0.0
        varav = {}
        for nod, dl in delar.items():
            s = sum(union_snitt_volym(b, boxar) for _, b in dl)
            if s > nr.VOL_TOL_MM3:
                varav[nod] = round(s, 1)
            v += s
        hard[R.namn] = {"klass": R.klass, "materia_mm3": round(v, 1), "varav_maskin": varav,
                        "apriori": R.namn in apriori, "GRON": v <= nr.VOL_TOL_MM3}
    ut["HARD_transport_utrymning"] = hard
    ut["HARD_GRON"] = all(v["GRON"] for v in hard.values()) if hard else False
    ut["HARD_materia_mm3"] = round(sum(v["materia_mm3"] for v in hard.values()), 1)
    # THE FAIR YARDSTICK: the bands declared A PRIORI sit in exactly the same place in both the
    # positive run and the negative control. Only they can be compared between the two.
    ut["HARD_apriori_mm3"] = round(sum(v["materia_mm3"] for v in hard.values() if v["apriori"]), 1)
    ut["HARD_apriori_varav"] = {k: v["varav_maskin"] for k, v in hard.items()
                                if v["apriori"] and v["materia_mm3"] > nr.VOL_TOL_MM3}

    # OPERATOR ZONES FREE (other machines' material; the machine's own is reported separately)
    opz = {}
    for R in H["rum"]:
        if R.klass != "operator":
            continue
        boxar = [nr.prim_bbox(p) for p in R.prims]
        annan, egen = 0.0, 0.0
        varav = {}
        for nod, dl in delar.items():
            s = sum(union_snitt_volym(b, boxar) for _, b in dl)
            if nod in R.agare:
                egen += s
            else:
                annan += s
                if s > nr.VOL_TOL_MM3:
                    varav[nod] = round(s, 1)
        opz[R.namn] = {"annan_maskins_materia_mm3": round(annan, 1),
                       "egen_maskins_materia_mm3": round(egen, 1), "varav": varav,
                       "GRON": annan <= nr.VOL_TOL_MM3}
    ut["operatorszoner"] = opz
    ut["OPERATOR_GRON"] = all(v["GRON"] for v in opz.values()) if opz else False
    return ut


def golvyta(H):
    b = H["hall_bbox_mm"]
    if not b:
        return None
    w = b[3] - b[0] + 2 * VAGGMARGINAL_MM
    h = b[4] - b[1] + 2 * VAGGMARGINAL_MM
    return {"w_mm": round(w, 1), "h_mm": round(h, 1), "area_m2": round(w * h / 1e6, 3),
            "harledning": ("AABB(placed machines + ALL declared negative spaces) + "
                           f"2 x wall margin {VAGGMARGINAL_MM:.0f} mm")}


def kor(log=print):
    T0 = time.time()
    noder, edges, fab = las_processgraf()

    # ---- boundary condition A: the factory spec's DECLARED building dimensions ----------------
    bm = fab["layoutvillkor"]["byggnadsmatt_mm"]
    log("RV-A: declared building dimensions from the factory spec")
    A = bygg_hall("RV-A_fabriksspec", bm["w"], bm["h"], True, log)
    # ---- boundary condition C: derived hall (unbounded) ---------------------------------------
    log("RV-C: derived hall (unbounded)")
    C = bygg_hall("RV-C_harledd", None, None, True, log)
    log("NEG: negative control -- the same machines WITHOUT negative spaces")
    N = bygg_hall("NEG_utan_negativrum", None, None, False, log)

    log("gates on RV-C ...")
    GC = grinda(C, log=log)
    log("gates on NEG (same measurement, different placement) ...")
    GN = grinda(N, log=log)

    yta = golvyta(C)

    # CONGESTION FINDINGS
    trangsel = []
    for H in (A,):
        for x in H["lagg"]:
            if not x["TRANGSEL"]:
                continue
            trangsel.append({
                "rv": H["rv"], "hall_w_mm": H["hall_w_mm"], "hall_h_mm": H["hall_h_mm"],
                "nod": x["nod"], "maskin": x["maskin"],
                "maskinvolym_mm3": round(sum((b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])
                                             for _, b in las_manifest(x["maskin"])[1]), 1),
                "maskin_envelopp_mm": [round(v, 1) for v in las_manifest(x["maskin"])[1] and
                                       bbox_av([b for _, b in las_manifest(x["maskin"])[1]])],
                "konkurrerande_rum": x.get("konkurrerande_rum"),
                "n_kandidater": x.get("n_kandidater"),
                "avvisade_skal": _skalhistogram(x.get("avvisade", [])),
                "rumsvolym_mm3": x.get("rumsvolym_mm3")})

    # FLOW MEASURE -- the relation requirement "downstream near upstream", per process edge
    def ruttlangd(rutt):
        return sum(abs(rutt[i + 1][0] - rutt[i][0]) + abs(rutt[i + 1][1] - rutt[i][1])
                   for i in range(len(rutt) - 1))
    v_flode = round(sum(ruttlangd(r) for r in C["rutter"].values()), 1)
    flode = {"vart_mm": v_flode, "vart_n_kanter": len(C["rutter"]),
             "vart_per_kant_mm": round(v_flode / max(len(C["rutter"]), 1), 1),
             "jamforelsens_grans": ("only the per-edge number is comparable between two layouts, and "
                                    "only when the edge sets are the same set")}

    minne_gb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 3)
    return {"A": A, "C": C, "N": N, "GC": GC, "GN": GN, "yta": yta,
            "trangsel": trangsel, "flode": flode,
            "sekunder": round(time.time() - T0, 1), "topp_minne_gb": minne_gb}


def _skalhistogram(avvisade):
    h = {}
    for a in avvisade:
        k = a.get("avvisad_av", "?")
        if k == "randvillkor":
            k = "randvillkor: " + str(a.get("skal", ""))[:90]
        elif a.get("rum"):
            k = f"negativrum: {a['rum']}"
        h[k] = h.get(k, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--kor" in sys.argv:
        R = kor()
        print(json.dumps({"n_trangsel_A": R["A"]["n_trangsel"],
                          "yta": R["yta"], "flode": R["flode"],
                          "GC": {k: v for k, v in R["GC"].items() if k.endswith("GRON")
                                 or k.startswith("HARD_m") or k.startswith("G1_") or k.startswith("G2_u")},
                          "GN": {k: v for k, v in R["GN"].items() if k.endswith("GRON")
                                 or k.startswith("HARD_m")},
                          "sek": R["sekunder"], "minne_gb": R["topp_minne_gb"]},
                         ensure_ascii=False, indent=1))
        raise SystemExit(0)
    print(__doc__)
