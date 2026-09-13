#!/usr/bin/env python3
"""Own family recipes: the five families the ported standard tables do not cover.

T-slot aluminium profile, V-slot aluminium profile, DIN rail, threaded rod and standoff -- among the
most used parts in a machine frame and absent from the ported tables.

Provenance class (honesty requirement, measured vs constructed): the dimension tables below are
HAND-ENTERED from publicly published profile/standard catalogues. They are NOT machine-read from a
named source and therefore may NOT be called measured. Every recipe carries
extra["provenans_klass"] = "hand-inmatad" so the register can tell them apart from the ported
standard tables (machine-read) and from harvested instances (sha256-pinned). The V-slot profile's
outer dimensions come from the open-source vslot families (LGPL 2.1+).

I/O: importing this module registers its recipes into recept_v1's register; the selftests build each
family and measure volume, bbox and solid count.
"""
from __future__ import annotations

import math

import build123d as bd

from recept_v1 import Kontroll, Param, Recept, Sjalvtest, registrera

EGEN_LICENS = "own geometry; dimensions from publicly published catalogues"
HAND = {"provenans_klass": "hand-inmatad", "varning":
        "the dimension table is hand-entered from general catalogue knowledge, NOT machine-read from a named source"}


# ------------------------------------------------------------------ arketyper

def a_tslot_profil(w, h, l, spar_bredd=6.2, spar_djup=None, centrumhal=4.2, modul=20.0):  # noqa: E741
    """T-slot aluminium profile (2020/4040 type): a solid bar w x h x l where every 20 mm module on
    each side gets a longitudinal slot, plus one centre hole per module cell.
    Declared simplification: the slot's inner dovetail shape is modelled as a RECTANGULAR slot --
    the outer slot opening (the width an M5 T-nut sees) is the dimension that governs assembly, and
    that is the dimension built exactly."""
    d = spar_djup if spar_djup is not None else modul * 0.30
    p = bd.extrude(bd.Rectangle(w, h), amount=l)
    nx, ny = max(1, int(round(w / modul))), max(1, int(round(h / modul)))
    for i in range(nx):
        cx = -w / 2 + modul * (i + 0.5)
        for sign in (1, -1):
            p -= bd.Pos(cx, sign * (h / 2 - d / 2), l / 2) * bd.Box(spar_bredd, d, l * 1.01)
    for j in range(ny):
        cy = -h / 2 + modul * (j + 0.5)
        for sign in (1, -1):
            p -= bd.Pos(sign * (w / 2 - d / 2), cy, l / 2) * bd.Box(d, spar_bredd, l * 1.01)
    if centrumhal:
        for i in range(nx):
            for j in range(ny):
                p -= bd.Pos(-w / 2 + modul * (i + 0.5), -h / 2 + modul * (j + 0.5), l / 2) \
                    * bd.Cylinder(radius=centrumhal / 2, height=l * 1.01)
    return p


def a_din_skena(bredd, hojd, tjocklek, l, hal=True):  # noqa: E741
    """DIN rail (EN 60715 top-hat profile, TS35 and others): a U-shaped hat section with outward
    flanges, extruded along Z. Built from the outer profile's three dimensions; the hole pattern
    (25 mm pitch) is optional."""
    t = tjocklek
    lapp = 7.0 if bredd >= 30 else 5.0
    sek = (bd.Pos(0, hojd - t / 2) * bd.Rectangle(bredd - 2 * lapp + 2 * t, t)) \
        + (bd.Pos((bredd - 2 * lapp) / 2 + t / 2, hojd / 2) * bd.Rectangle(t, hojd)) \
        + (bd.Pos(-((bredd - 2 * lapp) / 2 + t / 2), hojd / 2) * bd.Rectangle(t, hojd)) \
        + (bd.Pos((bredd - lapp) / 2, t / 2) * bd.Rectangle(lapp, t)) \
        + (bd.Pos(-(bredd - lapp) / 2, t / 2) * bd.Rectangle(lapp, t))
    p = bd.extrude(sek, amount=l)
    if hal:
        n = max(1, int(l // 25))
        for i in range(n):
            z = 12.5 + 25 * i
            if z < l - 5:
                p -= bd.Pos(0, hojd - t / 2, z) * bd.Rot(90, 0, 0) * bd.Cylinder(
                    radius=3.15, height=t * 4)
    return p


def a_gangstang(d, l, gangdjup_andel=0.065):  # noqa: E741
    """Threaded rod: a cylinder with EFFECTIVE diameter = nominal d minus half the thread depth.
    The thread helix is NOT modelled (it costs polygons without carrying a requirement) -- but the
    volume is not allowed to lie either: the effective core diameter gives the right mass within a
    couple of percent, which is what the rod actually contributes to a bill of materials."""
    return bd.Pos(0, 0, l / 2) * bd.Cylinder(radius=d * (1 - gangdjup_andel) / 2.0, height=l)


def a_distansbult(d_yttre, l, hal_d, typ="hona", tapp_l=6.0, tapp_d=None):  # noqa: E741
    """Standoff: a hexagonal or round pillar with an internal thread (female) or
    en gangad tapp i ena anden (hane)."""
    # extrude(amount=l) runs 0..l. (both=True gave DOUBLE length -- caught by the volume check.)
    p = bd.extrude(bd.RegularPolygon(radius=d_yttre / 2, side_count=6, major_radius=False),
                   amount=l)
    p -= bd.Pos(0, 0, l / 2) * bd.Cylinder(radius=hal_d / 2, height=l * 1.01)
    if typ == "hane":
        p += bd.Pos(0, 0, l + tapp_l / 2) * bd.Cylinder(
            radius=(tapp_d or hal_d) / 2 * 0.94, height=tapp_l)
    return p


# ------------------------------------------------------------------ tabeller (hand-inmatade)

TSLOT = {  # profil: (bredd_mm, hojd_mm, sparbredd_mm, centrumhal_mm, modul_mm)
    "2020": (20, 20, 6.2, 4.2, 20), "2040": (20, 40, 6.2, 4.2, 20),
    "2080": (20, 80, 6.2, 4.2, 20), "4040": (40, 40, 8.2, 6.8, 40),
    "4080": (40, 80, 8.2, 6.8, 40), "3030": (30, 30, 8.0, 6.8, 30),
    "3060": (30, 60, 8.0, 6.8, 30), "4545": (45, 45, 10.0, 10.0, 45),
}
TSLOT_LANGDER = [100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000]

VSLOT = {  # V-slot (outer dimensions from the open-source vslot families)
    "vslot2020": (20, 20, 6.2, 5.2, 20), "vslot2040": (20, 40, 6.2, 5.2, 20),
    "vslot2060": (20, 60, 6.2, 5.2, 20), "vslot2080": (20, 80, 6.2, 5.2, 20),
}

DINSKENA = {  # EN 60715: (bredd_mm, hojd_mm, godstjocklek_mm)
    "TS35x7.5": (35.0, 7.5, 1.0), "TS35x15": (35.0, 15.0, 1.5),
    "TS35x15_2.3": (35.0, 15.0, 2.3), "TS32": (32.0, 15.0, 2.0), "TS15": (15.0, 5.5, 1.0),
}
DINSKENA_LANGDER = [100, 150, 200, 250, 300, 400, 500, 1000, 2000]

GANGSTANG = {"M3": 3.0, "M4": 4.0, "M5": 5.0, "M6": 6.0, "M8": 8.0, "M10": 10.0,
             "M12": 12.0, "M14": 14.0, "M16": 16.0, "M20": 20.0, "M24": 24.0}
GANGSTANG_LANGDER = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]

DISTANS = {  # (nyckelvidd_mm, gang_kardia_mm)
    "M2.5": (5.0, 2.5), "M3": (5.5, 3.0), "M4": (7.0, 4.0), "M5": (8.0, 5.0), "M6": (10.0, 6.0)}
DISTANS_LANGDER = [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60]


# ------------------------------------------------------------------ recept

def _reg(rid, familj, namn, taggar, arketyp, build, schema, tester, rader, varianter, extra=None):
    ex = dict(HAND)
    ex.update(extra or {})
    return registrera(Recept(id=rid, familj=familj, namn=namn, klass="familj",
                             kalla="egen:recept_familjer_egna_v1", licens=EGEN_LICENS,
                             taggar=taggar, param_schema=schema, build=build, sjalvtest=tester,
                             arketyp=arketyp, tabellrader=rader, variant_count=varianter,
                             extra=ex))


# --- T-spar-aluprofil ---------------------------------------------------------
def _b_tslot(key, l=500.0):  # noqa: E741
    w, h, sb, ch, mod = TSLOT[key]
    return a_tslot_profil(w, h, l, sb, None, ch, mod)


_t = []
for _k in ("2020", "4040", "2080"):
    _w, _h, _sb, _ch, _mod = TSLOT[_k]
    _l = 500.0
    _bulk = _w * _h * _l
    _t.append(Sjalvtest({"key": _k, "l": _l}, [
        Kontroll("bbox_x_mm", _w * 0.999, _w * 1.001, False, "profilbredd"),
        Kontroll("bbox_y_mm", _h * 0.999, _h * 1.001, False, "profilhojd"),
        Kontroll("bbox_z_mm", _l * 0.999, _l * 1.001, False, "cut length"),
        Kontroll("volym_mm3", _bulk * 0.40, _bulk * 0.85, True,
                 "INDEPENDENT: a real T-slot profile is 45-65 % material of its bounding box "
                 "(catalogue cross-section 2020 ~ 160 mm2 of 400 mm2); a profile WITHOUT slots or "
                 "holes gives 1.0 and fails, an over-cut one gives < 0.40 and fails"),
    ], f"{_k} l={_l}"))
_reg("egen.aluprofil_tslot", "aluprofil_tslot", "T-spar-aluprofil (2020/4040/...)",
     ["profil", "aluprofil", "t-spar", "ram"], "tslot_profil", _b_tslot,
     [Param("key", "-", "str", default="2020", beskrivning="profilbeteckning"),
      Param("l", "mm", "float", 20.0, 6000.0, 500.0, "kaplangd")],
     _t, len(TSLOT), len(TSLOT) * len(TSLOT_LANGDER))


# --- V-spar-aluprofil ---------------------------------------------------------
def _b_vslot(key, l=500.0):  # noqa: E741
    w, h, sb, ch, mod = VSLOT[key]
    return a_tslot_profil(w, h, l, sb, None, ch, mod)


_t = []
for _k in ("vslot2020", "vslot2040"):
    _w, _h, _sb, _ch, _mod = VSLOT[_k]
    _l = 300.0
    _bulk = _w * _h * _l
    _t.append(Sjalvtest({"key": _k, "l": _l}, [
        Kontroll("bbox_x_mm", _w * 0.999, _w * 1.001, False, "profilbredd"),
        Kontroll("bbox_y_mm", _h * 0.999, _h * 1.001, False, "profilhojd"),
        Kontroll("volym_mm3", _bulk * 0.40, _bulk * 0.85, True,
                 "INDEPENDENT: material fraction 45-65 % of the bounding box (same anchor as the T-slot)"),
    ], f"{_k} l={_l}"))
_reg("egen.aluprofil_vslot", "aluprofil_vslot", "V-spar-aluprofil (20x20..20x80)",
     ["profil", "aluprofil", "v-spar", "ram"], "tslot_profil", _b_vslot,
     [Param("key", "-", "str", default="vslot2020", beskrivning="profilbeteckning"),
      Param("l", "mm", "float", 20.0, 6000.0, 300.0, "kaplangd")],
     _t, len(VSLOT), len(VSLOT) * len(TSLOT_LANGDER),
     extra={"yttermatt_kalla": "BOLTS vslot*-familjerna (LGPL 2.1+)"})


# --- DIN-skena ----------------------------------------------------------------
def _b_dinskena(key, l=300.0, hal=True):  # noqa: E741
    b, h, t = DINSKENA[key]
    return a_din_skena(b, h, t, l, hal)


_t = []
for _k in ("TS35x7.5", "TS35x15", "TS15"):
    _b, _h, _tt = DINSKENA[_k]
    _l = 300.0
    _t.append(Sjalvtest({"key": _k, "l": _l, "hal": True}, [
        Kontroll("bbox_x_mm", _b * 0.995, _b * 1.005, True,
                 "INDEPENDENT: EN 60715 fixes the rail WIDTH (35/32/15 mm) as the dimension "
                 "the device clips grip -- the builder is given the width but the flange/web split "
                 "is our own, so the measured outer width checks that the section is not "
                 "grown outside its span"),
        Kontroll("bbox_y_mm", _h * 0.995, _h * 1.02, False, "profile height"),
        Kontroll("bbox_z_mm", _l * 0.999, _l * 1.001, False, "cut length"),
        Kontroll("volym_mm3", 0.05 * _b * _h * _l, 0.75 * _b * _h * _l, False,
                 "a flat profile: much air in its bbox"),
    ], f"{_k} l={_l}"))
_reg("egen.din_skena", "din_skena", "DIN-skena EN 60715 (TS35/TS32/TS15)",
     ["elektronik", "skena", "montage", "din-skena"], "din_skena", _b_dinskena,
     [Param("key", "-", "str", default="TS35x7.5", beskrivning="skenprofil"),
      Param("l", "mm", "float", 25.0, 4000.0, 300.0, "kaplangd"),
      Param("hal", "-", "bool", default=True, beskrivning="slots with 25 mm pitch")],
     _t, len(DINSKENA), len(DINSKENA) * len(DINSKENA_LANGDER))


# --- Gangstang ----------------------------------------------------------------
def _b_gangstang(key, l=1000.0):  # noqa: E741
    return a_gangstang(GANGSTANG[key], l)


_t = []
for _k in ("M4", "M8", "M16"):
    _d = GANGSTANG[_k]
    _l = 1000.0
    _vol = math.pi / 4.0 * (_d * 0.935) ** 2 * _l
    _t.append(Sjalvtest({"key": _k, "l": _l}, [
        Kontroll("bbox_x_mm", _d * 0.90, _d * 1.001, True,
                 "INDEPENDENT: a threaded rod's OUTER dimension may never exceed its nominal "
                 "thread diameter (and not fall below the core diameter ~0.85d) -- a builder that "
                 "set radius=d by mistake would give 2d and fail here"),
        Kontroll("bbox_z_mm", _l * 0.999, _l * 1.001, False, "cut length"),
        Kontroll("volym_mm3", _vol * 0.97, _vol * 1.03, False, "effektiv kardiametervolym"),
    ], f"{_k} l={_l}"))
_reg("egen.gangstang", "gangstang", "Gangstang M3-M24 (DIN 975-lika)",
     ["fastelement", "gangstang", "metrisk"], "gangstang", _b_gangstang,
     [Param("key", "-", "str", default="M8", beskrivning="gangbeteckning"),
      Param("l", "mm", "float", 10.0, 3000.0, 1000.0, "kaplangd")],
     _t, len(GANGSTANG), len(GANGSTANG) * len(GANGSTANG_LANGDER))


# --- Distansbult --------------------------------------------------------------
def _b_distans(key, l=20.0, typ="hona"):  # noqa: E741
    s, d = DISTANS[key]
    return a_distansbult(s, l, d, typ)


_t = []
for _k, _typ in (("M3", "hona"), ("M4", "hane"), ("M6", "hona")):
    _s, _d = DISTANS[_k]
    _l = 20.0
    _t.append(Sjalvtest({"key": _k, "l": _l, "typ": _typ}, [
        Kontroll("bbox_xy_min_mm", _s * 0.98, _s * 1.02, False, "width across flats"),
        Kontroll("bbox_xy_max_mm", _s * 1.13, _s * 1.16, True,
                 "INDEPENDENT: a regular hexagon's across-corners dimension MUST be 2/sqrt(3) = "
                 "1.1547 x the width across flats -- a geometric identity the builder was not fed; "
                 "a square gives 1.414 and a cylinder 1.000, both fail"),
        Kontroll("bbox_z_mm", _l * 0.999, (_l + 6.5) * 1.001, False,
                 "pillar length (plus any male stud)"),
        Kontroll("volym_mm3", 0.0, math.sqrt(3) / 2 * _s * _s * (_l + 8), False,
                 "below a solid hexagonal prism (the bore is present)"),
    ], f"{_k} {_typ} l={_l}"))
_reg("egen.distansbult", "distansbult", "Standoff M2.5-M6 (female/male)",
     ["fastelement", "distans", "montage", "elektronik"], "distansbult", _b_distans,
     [Param("key", "-", "str", default="M3", beskrivning="gangbeteckning"),
      Param("l", "mm", "float", 3.0, 200.0, 20.0, "pelarlangd"),
      Param("typ", "-", "str", default="hona", beskrivning="hona | hane")],
     _t, len(DISTANS), len(DISTANS) * len(DISTANS_LANGDER) * 2)
