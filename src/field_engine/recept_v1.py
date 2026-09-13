#!/usr/bin/env python3
"""The recipe library: the smallest reusable CAD unit.

A RECIPE has five parts and none of them is optional:

  1. BUILD FUNCTION  -- a parametric function returning a build123d Solid/Compound. Family recipes
                        build the geometry themselves (own OCC builders); instance recipes point at
                        a harvested STEP file with a sha256.
  2. PARAM SCHEMA    -- every parameter with a UNIT, a type, min/max bounds and a default. A
                        parameter without a unit is a number without a carried quantity: forbidden.
  3. SELFTEST        -- >= 1 (target 3) parameter sets that are built and MEASURED (volume / bbox /
                        solid count) against an EXPECTATION. Fail-closed: a recipe whose selftest
                        does not pass is never registered (it lands in the register's `avvisade`
                        list with its error message -- silence is not allowed).
  4. PROVENANCE      -- source (repository / URL / own) + LICENCE, stored per recipe. An unclear
                        licence means the recipe is not harvested at all.
  5. TAGS            -- fastener / bearing / profile / cut / tube / electronics / ... : what makes
                        the library searchable.

Two recipe classes, deliberately kept apart in the accounting (never merged into ONE number):
  * FAMILY RECIPES (class "familj"): a parametric builder plus a standard table. Its part coverage
    = table rows x free combinations. This is where the multiplication towards a large library sits.
  * INSTANCE RECIPES (class "instans"): ONE harvested, concrete part identified by sha256. Part
    coverage = 1 per recipe.

Selftest honesty: every check carries the flag `oberoende` (independent). A check is INDEPENDENT
only when the expectation comes from a table column the builder did NOT use (over-determination,
e.g. a hex nut's measured across-corners dimension against the standard's e_min when the builder was
only given the width across flats). The other checks are build integrity (they catch crashes,
degenerate geometry, unit errors) and are reported as such. Both are counted separately in --status;
neither is called "validation of the standard".

I/O: recipe modules register themselves into an in-memory register; --bygg-register runs every
selftest and writes the register JSON under artifacts/ next to this module, --status prints the
honest counts from that register. Instance recipes live in a separate JSON-lines file so the main
register stays readable.

CLI:
  python recept_v1.py --status            # honest counts from the register
  python recept_v1.py --bygg-register     # run ALL selftests, write the register
  python recept_v1.py --sjalvtest <id>    # one recipe, verbose
build123d is needed for --bygg-register/--sjalvtest; --status only reads JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

REPO = os.environ.get("FIELD_ENGINE_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
REGISTER_PATH = os.path.join(_ART, "recept_register_v1.json")
# Instance recipes can be tens of thousands of rows -- they live in their own jsonl (one row per
# recipe) so the main register stays readable. The main register carries their COUNT and this path.
INSTANS_PATH = os.path.join(_ART, "recept_instanser_v1.jsonl")

# ------------------------------------------------------------------ datamodell


@dataclass
class Param:
    """A recipe parameter. `enhet` (unit) is MANDATORY -- a number without a unit is not a measure."""

    namn: str
    enhet: str
    typ: str = "float"          # float | int | str | bool
    minv: float | None = None
    maxv: float | None = None
    default: Any = None
    beskrivning: str = ""

    def som_dict(self):
        return {"namn": self.namn, "enhet": self.enhet, "typ": self.typ,
                "min": self.minv, "max": self.maxv, "default": self.default,
                "beskrivning": self.beskrivning}


@dataclass
class Kontroll:
    """En sjalvtest-kontroll: matt storhet vs forvantat intervall."""

    storhet: str                 # "volym_mm3" | "bbox_x_mm" | ... | "solids"
    lo: float
    hi: float
    oberoende: bool = False      # True = forvantan kommer fran en kolumn byggaren EJ anvande
    motivering: str = ""


@dataclass
class Sjalvtest:
    params: dict
    kontroller: list[Kontroll]
    etikett: str = ""


@dataclass
class Recept:
    id: str
    familj: str
    namn: str
    klass: str                   # "familj" | "instans"
    kalla: str
    licens: str
    taggar: list[str]
    param_schema: list[Param] = field(default_factory=list)
    build: Callable | None = None
    sjalvtest: list[Sjalvtest] = field(default_factory=list)
    arketyp: str = ""            # vilken egen OCC-byggare familjen anvander
    standard: str | None = None
    tabellrader: int = 0
    variant_count: int = 1       # antal delvarianter receptet tacker
    extra: dict = field(default_factory=dict)

    def som_dict(self):
        return {"id": self.id, "familj": self.familj, "namn": self.namn, "klass": self.klass,
                "kalla": self.kalla, "licens": self.licens, "taggar": self.taggar,
                "arketyp": self.arketyp, "standard": self.standard,
                "tabellrader": self.tabellrader, "variant_count": self.variant_count,
                "param_schema": [p.som_dict() for p in self.param_schema],
                "n_sjalvtest": len(self.sjalvtest), "extra": self.extra}


REGISTER: dict[str, Recept] = {}
# Instance recipes: one concrete harvested part. Stored as flat dicts (no build function in memory --
# the build step is "read the sha256-pinned STEP file"), but the same FAIL-CLOSED requirement holds.
INSTANS_REGISTER: list[dict] = []
INSTANS_AVVISADE: list[dict] = []

# mandatory fields of an instance recipe -- if one is missing it is NEVER registered
_INSTANS_KRAV = ("id", "namn", "kalla", "licens", "taggar", "sha256", "kalla_url", "matt_klass")


def registrera_instans(d: dict) -> bool:
    """Fail-closed metadata gate for an instance recipe. Returns True if it was accepted.
    `matt_klass` must be either
      "geometri" -- the STEP file exists locally and volume/bbox are MEASURED (volym_mm3/bbox_mm), or
      "metadata" -- only sha256 + byte size are known; no geometry is measured (and none is claimed)."""
    saknas = [k for k in _INSTANS_KRAV if not d.get(k)]
    if saknas:
        INSTANS_AVVISADE.append({"id": d.get("id"), "skal": f"missing fields: {saknas}"})
        return False
    if "OKLAR" in str(d["licens"]).upper() or "UNKNOWN" in str(d["licens"]).upper():
        INSTANS_AVVISADE.append({"id": d["id"], "skal": "oklar licens"})
        return False
    if d["matt_klass"] not in ("geometri", "metadata"):
        INSTANS_AVVISADE.append({"id": d["id"], "skal": f"okand matt_klass {d['matt_klass']}"})
        return False
    if d["matt_klass"] == "geometri":
        v = d.get("volym_mm3")
        bb = d.get("bbox_mm") or []
        if not v or v <= 0 or len(bb) != 3 or min(bb) <= 0:
            INSTANS_AVVISADE.append({"id": d["id"],
                                     "skal": f"geometrigrind: volym={v} bbox={bb}"})
            return False
    d.setdefault("klass", "instans")
    d.setdefault("variant_count", 1)
    INSTANS_REGISTER.append(d)
    return True


def registrera(r: Recept) -> Recept:
    """Put the recipe in the in-memory register. This is NOT the approval -- approval happens in
    bygg_register(), where the selftest has to pass (fail-closed)."""
    if r.id in REGISTER:
        raise ValueError(f"dubblett-id i receptregistret: {r.id}")
    if not r.sjalvtest:
        raise ValueError(f"{r.id}: a recipe without a selftest may not even be defined")
    for p in r.param_schema:
        if not p.enhet:
            raise ValueError(f"{r.id}: parameter {p.namn} saknar enhet")
    if not r.licens or "OKLAR" in r.licens.upper() or "UNKNOWN" in r.licens.upper():
        raise ValueError(f"{r.id}: unclear licence -- may not be registered")
    REGISTER[r.id] = r
    return r


# ------------------------------------------------------------------ matning


def mat_form(shape) -> dict:
    """Mat en build123d-form: volym (mm3), bbox-sidor (mm), antal solids."""
    import build123d as bd

    if isinstance(shape, (list, tuple)):
        shape = bd.Compound(children=list(shape)) if len(shape) > 1 else shape[0]
    bb = shape.bounding_box()
    try:
        solids = len(shape.solids())
    except Exception:  # noqa: BLE001
        solids = 1
    x, y, z = float(bb.size.X), float(bb.size.Y), float(bb.size.Z)
    return {
        "volym_mm3": float(shape.volume),
        "bbox_x_mm": x, "bbox_y_mm": y, "bbox_z_mm": z,
        # xy max/min = across-corners resp. across-flats for a Z-axis hexagon: the quantity the
        # standard's e column can cross-check without the builder having seen it.
        "bbox_xy_max_mm": max(x, y), "bbox_xy_min_mm": min(x, y),
        "bbox_diag_mm": float((x ** 2 + y ** 2 + z ** 2) ** 0.5),
        "solids": float(solids),
    }


def kor_sjalvtest(r: Recept) -> dict:
    """Build the recipe with every selftest parameter set and compare against the checks.
    Returns {status, fall:[...], n_kontroller, n_oberoende}. A single failed case => the whole
    recipe FAILs."""
    fall = []
    ok = True
    n_ober = 0
    for st in r.sjalvtest:
        rec = {"etikett": st.etikett, "params": st.params}
        t0 = time.time()
        try:
            shape = r.build(**st.params)
            matt = mat_form(shape)
            rec["matt"] = matt
            kres = []
            for k in st.kontroller:
                v = matt.get(k.storhet)
                p = v is not None and (k.lo <= v <= k.hi)
                if k.oberoende:
                    n_ober += 1
                kres.append({"storhet": k.storhet, "matt": v, "lo": k.lo, "hi": k.hi,
                             "pass": bool(p), "oberoende": k.oberoende,
                             "motivering": k.motivering})
                if not p:
                    ok = False
            rec["kontroller"] = kres
            rec["status"] = "PASS" if all(x["pass"] for x in kres) else "FAIL"
        except Exception as e:  # noqa: BLE001
            ok = False
            rec["status"] = "FAIL"
            rec["fel"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc()[-800:]
        rec["sek"] = round(time.time() - t0, 3)
        fall.append(rec)
    return {"status": "PASS" if ok else "FAIL", "fall": fall,
            "n_kontroller": sum(len(x.kontroller) for x in r.sjalvtest),
            "n_oberoende": n_ober}


# ------------------------------------------------------------------ register build


def _ladda_kallor():
    """Import all recipe source modules. Every module registers its recipes on import."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # MEASURED TRAP: when this file runs as __main__, `from recept_v1 import ...` in the source
    # modules creates a SECOND copy of the module with its OWN empty REGISTER -- the register then
    # held 0 recipes although every module registered correctly. Bind the name to THIS module first.
    sys.modules.setdefault("recept_v1", sys.modules[__name__])
    moduler = []
    for m in ("recept_familjer_egna_v1",):
        try:
            __import__(m)
            moduler.append({"modul": m, "status": "OK"})
        except Exception as e:  # noqa: BLE001
            moduler.append({"modul": m, "status": "IMPORTFEL", "fel": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-600:]})
    return moduler


def bygg_register(bara: str | None = None) -> dict:
    moduler = _ladda_kallor()
    godkanda, avvisade = [], []
    t0 = time.time()
    items = sorted(REGISTER.values(), key=lambda r: r.id)
    if bara:
        items = [r for r in items if bara in r.id or bara == r.familj]
    for r in items:
        res = kor_sjalvtest(r)
        d = r.som_dict()
        d["sjalvtest"] = {"status": res["status"], "n_kontroller": res["n_kontroller"],
                          "n_oberoende": res["n_oberoende"],
                          "fall": [{"etikett": f.get("etikett"), "status": f["status"],
                                    "matt": f.get("matt"), "fel": f.get("fel"),
                                    "kontroller": f.get("kontroller")} for f in res["fall"]]}
        if res["status"] == "PASS":
            godkanda.append(d)
        else:
            avvisade.append(d)
    with open(INSTANS_PATH, "w") as f:
        for d in sorted(INSTANS_REGISTER, key=lambda x: x["id"]):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    reg = {
        "version": "recept_register_v1",
        "genererad": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "byggsek": round(time.time() - t0, 1),
        "moduler": moduler,
        "grind": "FAIL-CLOSED: only recipes with a PASS selftest are in `recept`",
        "recept": godkanda,
        "avvisade": avvisade,
        "instans_fil": os.path.relpath(INSTANS_PATH, REPO),
        "instanser": _instanssammanfattning(),
        "instanser_avvisade": INSTANS_AVVISADE[:200],
        "n_instanser_avvisade": len(INSTANS_AVVISADE),
    }
    reg["rakning"] = rakna(reg)
    if not bara:
        with open(REGISTER_PATH, "w") as f:
            json.dump(reg, f, indent=1, ensure_ascii=False)
    return reg


def _instanssammanfattning() -> dict:
    per: dict[str, dict] = {}
    for d in INSTANS_REGISTER:
        k = d["kalla"]
        e = per.setdefault(k, {"n": 0, "geometri": 0, "metadata": 0, "licenser": {}, "taggar": {}})
        e["n"] += 1
        e[d["matt_klass"]] += 1
        e["licenser"][d["licens"]] = e["licenser"].get(d["licens"], 0) + 1
        for t in d["taggar"]:
            e["taggar"][t] = e["taggar"].get(t, 0) + 1
    for e in per.values():
        e["taggar"] = dict(sorted(e["taggar"].items(), key=lambda x: -x[1])[:12])
    return {"tot": len(INSTANS_REGISTER),
            "geometri_matta": sum(1 for d in INSTANS_REGISTER if d["matt_klass"] == "geometri"),
            "endast_metadata": sum(1 for d in INSTANS_REGISTER if d["matt_klass"] == "metadata"),
            "per_kalla": per}


# ------------------------------------------------------------------ rakenskap


def rakna(reg: dict) -> dict:
    """THE HONEST COUNT. Four numbers that must NOT be merged:
      familjer            -- antal familjerecept (parametrisk byggare + tabell)
      delvarianter        -- SUM(variant_count) over familjerecepten = familj x tabellrader
      instanser           -- number of harvested instance recipes (one concrete part with sha256)
      delteckning_tot     -- delvarianter + instanser = vagen mot 10 000
    """
    rec = reg["recept"]
    fam = [r for r in rec if r["klass"] == "familj"]
    inst = reg.get("instanser", {"tot": 0, "geometri_matta": 0, "endast_metadata": 0,
                                 "per_kalla": {}})
    n_inst = inst["tot"]
    per_kalla: dict[str, dict] = {}
    for r in rec:
        k = r["kalla"].split(":")[0]
        d = per_kalla.setdefault(k, {"familjer": 0, "instanser": 0, "delvarianter": 0,
                                     "licenser": {}})
        d["familjer"] += 1
        d["delvarianter"] += r["variant_count"]
        d["licenser"][r["licens"]] = d["licenser"].get(r["licens"], 0) + 1
    per_familjetagg: dict[str, int] = {}
    for r in rec:
        for t in r["taggar"]:
            per_familjetagg[t] = per_familjetagg.get(t, 0) + r["variant_count"]
    n_ober_recept = sum(1 for r in rec if r["sjalvtest"]["n_oberoende"] > 0)
    for k, e in inst["per_kalla"].items():
        d = per_kalla.setdefault(k, {"familjer": 0, "instanser": 0, "delvarianter": 0,
                                     "licenser": {}})
        d["instanser"] += e["n"]
        d["delvarianter"] += e["n"]
        for lic, n in e["licenser"].items():
            d["licenser"][lic] = d["licenser"].get(lic, 0) + n
    dv = sum(r["variant_count"] for r in fam)
    return {
        "familjer": len(fam),
        "instanser": n_inst,
        "instanser_geometri_matta": inst["geometri_matta"],
        "instanser_endast_metadata": inst["endast_metadata"],
        "delvarianter_ur_familjer": dv,
        "delteckning_tot": dv + n_inst,
        "avvisade": len(reg["avvisade"]),
        "instanser_avvisade": reg.get("n_instanser_avvisade", 0),
        "sjalvtest_pass_andel": round(len(rec) / max(1, len(rec) + len(reg["avvisade"])), 4),
        "n_kontroller_tot": sum(r["sjalvtest"]["n_kontroller"] for r in rec),
        "n_oberoende_kontroller": sum(r["sjalvtest"]["n_oberoende"] for r in rec),
        "recept_med_oberoende_kontroll": n_ober_recept,
        "andel_recept_med_oberoende_kontroll": round(n_ober_recept / max(1, len(rec)), 4),
        "per_kalla": per_kalla,
        "delvarianter_per_tagg": dict(sorted(per_familjetagg.items(), key=lambda x: -x[1])),
        # THE TWO HONEST NUMBERS. The large number includes catalogue entries that are sha256-
        # PINNED but whose STEP has NOT been downloaded and whose geometry has NOT been measured.
        # That is a real part coverage (every entry is an identified, licence-clear part with a
        # downloadable URL) but it is NOT geometry in hand. Both are therefore ALWAYS reported,
        # and the coverage target is measured against the STRICTER number.
        "delteckning_geometriskt_realiserbar": dv + inst["geometri_matta"],
        "mal_10000": {
            "nu_geometriskt_realiserbar": dv + inst["geometri_matta"],
            "kvar_till_10000": max(0, 10000 - (dv + inst["geometri_matta"])),
            "andel_geometrisk": round((dv + inst["geometri_matta"]) / 10000, 4),
            "nu_inkl_sha256_pinnad_katalog": dv + n_inst,
            "andel_inkl_katalog": round((dv + n_inst) / 10000, 4),
        },
    }


def status(as_json=False):
    if not os.path.exists(REGISTER_PATH):
        print("NO REGISTER -- run --bygg-register first", file=sys.stderr)
        return 1
    with open(REGISTER_PATH) as f:
        reg = json.load(f)
    r = reg["rakning"]
    if as_json:
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return 0
    print(f"RECIPE REGISTER v1  (generated {reg['genererad']}, {reg['byggsek']} s)")
    print(f"  family recipes ............. {r['familjer']}")
    print(f"  part variants from families  {r['delvarianter_ur_familjer']}  (family x table rows)")
    print(f"  harvested instance recipes . {r['instanser']}  "
          f"(geometry measured {r['instanser_geometri_matta']}, "
          f"metadata+sha256 only {r['instanser_endast_metadata']})")
    print(f"  COVERAGE, geometry in hand . {r['delteckning_geometriskt_realiserbar']}   "
          f"({100 * r['mal_10000']['andel_geometrisk']:.1f} % of 10 000, "
          f"remaining {r['mal_10000']['kvar_till_10000']})")
    print(f"  COVERAGE incl. catalogue ... {r['delteckning_tot']}   "
          f"({100 * r['mal_10000']['andel_inkl_katalog']:.1f} % of 10 000) "
          f"-- the extra {r['instanser_endast_metadata']} are sha256-pinned catalogue entries "
          f"whose STEP is NOT downloaded and whose geometry is NOT measured")
    print(f"  selftest PASS fraction ..... {100 * r['sjalvtest_pass_andel']:.1f} %  "
          f"(rejected fail-closed: families {r['avvisade']}, "
          f"instances {r['instanser_avvisade']})")
    print(f"  checks in total ............ {r['n_kontroller_tot']}  "
          f"of which independent (over-determined): {r['n_oberoende_kontroller']}")
    print(f"  recipes with >=1 independent {r['recept_med_oberoende_kontroll']} "
          f"({100 * r['andel_recept_med_oberoende_kontroll']:.1f} %)")
    print("  per source:")
    for k, d in sorted(r["per_kalla"].items(), key=lambda x: -x[1]["delvarianter"]):
        lic = ", ".join(f"{a} x{b}" for a, b in d["licenser"].items())
        print(f"    {k:44s} fam={d['familjer']:4d} inst={d['instanser']:5d} "
              f"delvarianter={d['delvarianter']:6d}  [{lic}]")
    print("  part variants per tag (top 12):")
    for t, n in list(r["delvarianter_per_tagg"].items())[:12]:
        print(f"    {t:28s} {n}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--bygg-register", action="store_true")
    ap.add_argument("--sjalvtest", metavar="ID")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.sjalvtest:
        reg = bygg_register(bara=a.sjalvtest)
        print(json.dumps({"recept": reg["recept"], "avvisade": reg["avvisade"]},
                         indent=1, ensure_ascii=False)[:20000])
        return 0 if not reg["avvisade"] else 1
    if a.bygg_register:
        reg = bygg_register()
        for m in reg["moduler"]:
            if m["status"] != "OK":
                print(f"MODULFEL {m['modul']}: {m.get('fel')}", file=sys.stderr)
        print(json.dumps(reg["rakning"], indent=1, ensure_ascii=False))
        return 0
    return status(as_json=a.json)


if __name__ == "__main__":
    sys.exit(main())
