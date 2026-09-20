#!/usr/bin/env python3
"""A grammar reader for requirement text: regular expressions, no model, as the baseline to beat.

The recipe chain needs a requirement tree at its head -- named, typed, unit-carrying fields that the
op schema can be driven from. The obvious way to get one out of a sentence a person wrote is a
language model. This module is the control that makes that an empirical question instead of an
assumption: a bilingual (Swedish/English) pattern grammar over the eleven fields the recipe chain
actually consumes, costing about half a millisecond per intention and no checkpoint at all.

MEASURED, on a corpus of 40 hand-written intentions with hand-annotated requirement trees
(30 expected ACCEPT, 10 expected REJECT), carried here as data:

    fields extracted correctly, over the accepted set   139 / 195   (0.7128)
    per field: material 30/30, thickness 23/25, length 18/24, width 18/24, hole_count 10/20,
               hole_diameter 9/20, max_deflection 8/10, test_force 8/10, diameter 5/6,
               pitch_x 5/11, pitch_y 5/8, and height 0/5 and radius 0/2 -- two field names the
               grammar has no pattern for at all, which is where a sixth of the misses sit.

and, with the requirement tree then passed through a claim-type checker (SI unit grammar,
dimensional typing, positivity, and a cross-field test that a hole cannot be wider than the plate):

    accepted 29 / 40, rejected 11 / 40, verdict accuracy 0.925, false accepts 1, false rejects 2
    recipe -> valid exact solid, end to end                26 / 40   (0.65)

The last number is the one that matters, and it is the same 0.65 the typed 0.5B model reached on the
same corpus: on this set the model adds nothing end to end. What the model half needs is a checkpoint,
a GPU and 48.6 ms per intention; this needs `re`.

THE CAVEAT THAT TRAVELS WITH 0.65, and it is a real one: this grammar was written AFTER the 40
intentions were written, so the 0.65 is partly a fit to its own corpus. Re-measured on 20 intentions
it had never seen, the same grammar gets every field right in 6 of 15 accepted cases, hits 50 of 66
fields, and falsely accepts 1 of 5 cases it should have rejected. Read 0.65 as an upper bound for
this phrasing distribution, not as a yield.

The verdict leg above (29 accepted, 0.925) was measured with the decorrelation graph engine's
claim-type checker, which lives beside this repository and is not a dependency here. What this module
reproduces on its own, and what its selftest gates, is the extraction leg: 139 of 195 fields.
"""
import json
import os
import re

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                    "data", "cadbank_v1")

_RAKNEORD = {"en": 1, "ett": 1, "tva": 2, "två": 2, "tre": 3, "fyra": 4, "fem": 5, "sex": 6,
             "sju": 7, "atta": 8, "åtta": 8, "nio": 9, "tio": 10,
             "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10}

_TAL = r"\d+(?:[.,]\d+)?"
_M_DIM = re.compile(rf"({_TAL})\s*(?:×|x|\*)\s*({_TAL})(?:\s*(?:×|x|\*)\s*({_TAL}))?\s*([a-zA-Z]+)?")
_M_TJOCK_A = re.compile(r"(?:tjock(?:lek)?|godstjocklek|thickness|thick|height|höjd)"
                        rf"\s*(?:av|is|of)?\s*(-?{_TAL})\s*([a-zA-Z]+)?", re.IGNORECASE)
_M_TJOCK_B = re.compile(rf"(-?{_TAL})\s*([a-zA-Z]+)?\s*(?:tjock|thick|godstjocklek|thickness)",
                        re.IGNORECASE)
_M_DIAM_A = re.compile(rf"(?:diameter|radie|radius)\s*(?:av|is|of)?\s*({_TAL})\s*([a-zA-Z]+)?",
                       re.IGNORECASE)
_M_DIAM_B = re.compile(rf"({_TAL})\s*([a-zA-Z]+)?\s*(?:diameter|radie|radius)", re.IGNORECASE)
_M_HAL = re.compile(r"\b(en|ett|två|tre|fyra|fem|sex|sju|åtta|nio|tio|one|two|three|four|five|six|"
                    rf"seven|eight|nine|ten|\d+)\s*(?:M(\d+)|(?:({_TAL})\s*([a-zA-Z]+)?))?"
                    r"\s*(?:-\s*)?(?:fästhål|hörnhål|tvärhål|hål|holes|bolt holes)", re.IGNORECASE)
_M_DELNING = re.compile(rf"({_TAL})\s*(?:×|x|\*)\s*({_TAL})"
                        r"\s*(?:-delning|mm\s*(?:delning|pitch|pattern|spacing))", re.IGNORECASE)
_M_MATERIAL = re.compile(r"\b(aluminium|aluminum|stål|steel|mässing|brass|bronze|koppar|copper|"
                         r"titanium)\b", re.IGNORECASE)
_M_NEDBOJ = re.compile(r"(?:böja mer än|utböjning|nedböjning|"
                       rf"deflection(?: must not exceed| <=| under)?)\s*({_TAL})\s*([a-zA-Z]+)?",
                       re.IGNORECASE)
_M_KRAFT = re.compile(rf"(?:under|vid|at|lastkapacitet)\s*(?:last|load)?\s*(?:av|of)?\s*({_TAL})"
                      r"\s*([a-zA-Z]+)?", re.IGNORECASE)
_M_MASSA = re.compile(rf"mass\s*(?:of|is)?\s*({_TAL})\s*([a-zA-Z]+)?", re.IGNORECASE)
_M_KVOT = re.compile(rf"ratio\s*(?:of|is)?\s*({_TAL})\s*([a-zA-Z]+)?", re.IGNORECASE)


def _f(s):
    return float(s.replace(",", "."))


def _falt(typ, varde, enhet, tolerans, relation="exact"):
    return {"type": typ, "value": varde, "unit": enhet, "tolerance": tolerans, "relation": relation}


def las_krav(text):
    """One intention in, a requirement tree out: {field name: {type, value, unit, tolerance, relation}}.

    Fields the recipe chain consumes: length, width, thickness, diameter, hole_count, hole_diameter,
    pitch_x, pitch_y, material, max_deflection, test_force, plus mass and stiffness_ratio, which exist
    so that a dimensionally impossible claim has somewhere to land and be rejected."""
    falt = {}

    m = _M_DIM.search(text)
    if m:
        e = m.group(4) or "mm"
        falt["length"] = _falt("dimension", _f(m.group(1)), e, 0.5)
        falt["width"] = _falt("dimension", _f(m.group(2)), e, 0.5)
        if m.group(3):
            falt["thickness"] = _falt("dimension", _f(m.group(3)), e, 0.2)

    m = _M_TJOCK_A.search(text) or _M_TJOCK_B.search(text)
    if m:
        falt["thickness"] = _falt("dimension", _f(m.group(1)), m.group(2) or "mm", 0.2)

    m = _M_DIAM_A.search(text) or _M_DIAM_B.search(text)
    if m:
        falt["diameter"] = _falt("dimension", _f(m.group(1)), m.group(2) or "mm", 0.1)

    m = _M_HAL.search(text)
    if m:
        ord_ = m.group(1).lower()
        antal = _RAKNEORD.get(ord_, int(ord_) if ord_.isdigit() else 1)
        falt["hole_count"] = _falt("count", antal, "", 0)
        if m.group(2):                                     # an M-thread designation carries the size
            falt["hole_diameter"] = _falt("dimension", float(m.group(2)), "mm", 0.1)
        elif m.group(3):
            falt["hole_diameter"] = _falt("dimension", _f(m.group(3)), m.group(4) or "mm", 0.1)

    m = _M_DELNING.search(text)
    if m:
        falt["pitch_x"] = _falt("dimension", _f(m.group(1)), "mm", 0.2)
        falt["pitch_y"] = _falt("dimension", _f(m.group(2)), "mm", 0.2)

    m = _M_MATERIAL.search(text)
    if m:
        falt["material"] = _falt("material", m.group(1).lower(), "", None)

    m = _M_NEDBOJ.search(text)
    if m:
        falt["max_deflection"] = _falt("deflection", _f(m.group(1)), m.group(2) or "mm", None, "<=")

    m = _M_KRAFT.search(text)
    if m:
        falt["test_force"] = _falt("force", _f(m.group(1)), m.group(2) or "", None)

    m = _M_MASSA.search(text)
    if m:
        falt["mass"] = _falt("mass", _f(m.group(1)), m.group(2) or "", None)

    m = _M_KVOT.search(text)
    if m:
        falt["stiffness_ratio"] = _falt("ratio", _f(m.group(1)), m.group(2) or "", None)

    return falt


def falttraff(facit, last):
    """Did the reader recover this field? Numeric within the annotation's own tolerance (never
    tighter than 1e-4), strings case-insensitively equal."""
    fv, lv = facit.get("value"), last.get("value")
    tol = facit.get("tolerance") or 0.0
    if isinstance(fv, (int, float)) and isinstance(lv, (int, float)):
        return abs(float(fv) - float(lv)) <= max(tol, 1e-4)
    return str(fv).lower() == str(lv).lower()


def las_intentioner(sokvag=None):
    with open(os.path.join(sokvag or DATA, "intentioner.json"), encoding="utf-8") as fh:
        return json.load(fh)["intentions"]


def matt_extraktion(intentioner):
    """Score the extraction leg over the intentions annotated ACCEPT: per-field hits and the total.

    Only the accepted set is scored, because a tree that should be refused has no correct extraction
    to be measured against."""
    traffar, totalt = {}, {}
    for post in intentioner:
        if post["expected_verdict"] != "ACCEPT":
            continue
        last = las_krav(post["text"])
        for namn, facit in post["ground_truth"].items():
            totalt[namn] = totalt.get(namn, 0) + 1
            if namn in last and falttraff(facit, last[namn]):
                traffar[namn] = traffar.get(namn, 0) + 1
    summa_t, summa_n = sum(traffar.values()), sum(totalt.values())
    return {"traffar": summa_t, "falt": summa_n,
            "andel": summa_t / summa_n if summa_n else 0.0,
            "per_falt": {k: (traffar.get(k, 0), v) for k, v in sorted(totalt.items())}}


def _selftest():
    ints = las_intentioner()
    m = matt_extraktion(ints)
    print(f"intentions {len(ints)}")
    print(f"fields recovered {m['traffar']}/{m['falt']} = {m['andel']:.4f}")
    for namn, (h, t) in m["per_falt"].items():
        print(f"  {namn:18s} {h:3d}/{t:3d}")
    if (m["traffar"], m["falt"]) != (139, 195):
        raise SystemExit(f"FAIL: extraction leg is {m['traffar']}/{m['falt']}, measured 139/195")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
