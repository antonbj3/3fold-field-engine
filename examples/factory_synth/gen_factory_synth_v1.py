#!/usr/bin/env python3
"""Generate the synthetic factory input used by fabriksnegativrum_v1.py.

Writes, next to this script:
  fabriksspec_v2.json                          process graph (nodes + edges) and layout constraints
  cad/<station>/parts_manifest_v1.json         one part manifest per station: {"parts":[{"part","bbox",
                                               "subassembly"}]} with bbox = [x0,y0,z0,x1,y1,z1] in mm
  cad/polering_exemplar_v1/negativrum_v1.json  one station that already declares its own machine-level
                                               negative spaces, so the recursion (machine room -> hall
                                               room) has something to lift

Every number here is declared, not measured: six process stations on a line, each a box-shaped machine
with an HMI pendant, an electrical cabinet and a service hatch, plus a building-services station whose
ceiling run sits above the ISO 14122-2 clear height and a material-flow station that carries one EUR
pallet (800 x 1200 x 144 mm).

Run: python gen_factory_synth_v1.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# station id -> (directory, machine length x, depth y, height z, has own negativrum_v1 schema)
STATIONS = [
    ("smaltugn",           "smaltugn_exemplar_v1",       3200.0, 2400.0, 2600.0, False),
    ("formning_till_amne", "glasformning_exemplar_v1",   2600.0, 2000.0, 2200.0, False),
    ("grovslipning",       "glasslip_exemplar_v1",       2200.0, 1800.0, 2000.0, False),
    ("polering",           "polermaskin_exemplar_v1",    2400.0, 1900.0, 2100.0, True),
    ("ar_coating",         "ar_coat_exemplar_v1",        3000.0, 2200.0, 2400.0, False),
    ("metrologi",          "glas_metrologi_exemplar_v1", 1800.0, 1600.0, 1800.0, False),
]
# an intermediate node with no built machine, so the edge contraction has something to contract
EXTRA_NODES = [("transportband", "conveyor between grinding and polishing")]


def _box(x0, y0, z0, x1, y1, z1):
    return [float(x0), float(y0), float(z0), float(x1), float(y1), float(z1)]


def machine_manifest(station, w, d, h):
    """Parts of one machine in its own frame: body, frame, HMI pendant, cabinet, service hatch."""
    parts = [
        {"part": "stativ", "subassembly": "ram", "bbox": _box(0, 0, 0, w, d, h * 0.15)},
        {"part": "hus", "subassembly": "ram", "bbox": _box(0, 0, h * 0.15, w, d, h)},
        {"part": "hmi_pendang", "subassembly": "styr",
         "bbox": _box(w * 0.35, -320.0, 900.0, w * 0.65, 0.0, 1700.0)},
        {"part": "hmi_skarm", "subassembly": "styr",
         "bbox": _box(w * 0.40, -340.0, 1150.0, w * 0.60, -300.0, 1500.0)},
        {"part": "elskap_huvud", "subassembly": "el",
         "bbox": _box(w, d * 0.30, 200.0, w + 400.0, d * 0.70, 2000.0)},
        {"part": "servicelucka_bak", "subassembly": "ram",
         "bbox": _box(w * 0.30, d, 400.0, w * 0.70, d + 60.0, 1600.0)},
    ]
    return {"schema": "parts_manifest_v1", "station": station, "parts": parts}


def utilities_manifest():
    """Building services: a ceiling-mounted run above the clear height plus two room shells."""
    parts = [
        {"part": "kabelstege_tak", "subassembly": "el/strak",
         "bbox": _box(-2000.0, 0.0, 2105.0, 30000.0, 900.0, 2450.0)},
        {"part": "tryckluftsror_tak", "subassembly": "tryckluft/strak",
         "bbox": _box(-2000.0, 150.0, 2500.0, 30000.0, 700.0, 3200.0)},
        {"part": "hallgolv", "subassembly": "bim/rum",
         "bbox": _box(-3000.0, -3000.0, -200.0, 40000.0, 20000.0, 0.0)},
        {"part": "yttervagg_syd", "subassembly": "bim/rum",
         "bbox": _box(-3000.0, -3200.0, 0.0, 40000.0, -3000.0, 6000.0)},
    ]
    return {"schema": "parts_manifest_v1", "station": "utilities", "parts": parts}


def materialflow_manifest():
    """One EUR pallet, 800 x 1200 x 144 mm, as the measured carrier."""
    parts = [
        {"part": "pall_toppbrada_1", "subassembly": "pall", "bbox": _box(0, 0, 122.0, 800.0, 1200.0, 144.0)},
        {"part": "pall_kloss_1", "subassembly": "pall", "bbox": _box(0, 0, 22.0, 145.0, 145.0, 122.0)},
        {"part": "pall_kloss_2", "subassembly": "pall", "bbox": _box(655.0, 1055.0, 22.0, 800.0, 1200.0, 122.0)},
        {"part": "pall_bottenbrada_1", "subassembly": "pall", "bbox": _box(0, 0, 0.0, 800.0, 145.0, 22.0)},
    ]
    return {"schema": "parts_manifest_v1", "station": "materialflode", "parts": parts}


def machine_negativrum(station, w, d, h):
    """A machine-level negativrum_v1 file: one service room and one cooling room, in the machine frame."""
    def lada(cx, cy, cz, hx, hy, hz, namn):
        return {"typ": "lada", "center": [cx, cy, cz], "half": [hx, hy, hz], "namn": namn}
    return {
        "schema": "negativrum_v1", "maskin": station,
        "rum": [
            {"namn": "service_bakre_lucka", "klass": "service",
             "kalla": {"typ": "krav", "krav_id": "service_access"},
             "motiv": "tool access in front of the rear service hatch",
             "primitiver": [lada(w * 0.5, d + 330.0, 1000.0, w * 0.2, 300.0, 600.0, "service_box")]},
            {"namn": "kylluft_utblas", "klass": "kyl",
             "kalla": {"typ": "krav", "krav_id": "cooling_outlet"},
             "motiv": "cooling-air outlet path that may not be walled in",
             "primitiver": [lada(w + 250.0, d * 0.15, h * 0.8, 250.0, 200.0, 300.0, "kyl_box")]},
        ],
    }


def spec():
    nodes = [{"id": s, "namn": s, "typ": "process"} for s, _, _, _, _, _ in STATIONS]
    nodes += [{"id": n, "namn": n, "typ": "transport", "kommentar": c} for n, c in EXTRA_NODES]
    order = ["smaltugn", "formning_till_amne", "grovslipning", "transportband", "polering",
             "ar_coating", "metrologi"]
    edges = [{"from": a, "to": b, "typ": "materialflode", "kalla": "process line order"}
             for a, b in zip(order, order[1:])]
    return {"schema": "fabriksspec_v2",
            "factories": {"glasfabrik": {
                "processgraf": {"nodes": nodes, "edges": edges},
                "layoutvillkor": {"byggnadsmatt_mm": {"w": 26000.0, "h": 14000.0},
                                  "sakerhetszon_mm": 300.0}}}}


def main():
    os.makedirs(os.path.join(HERE, "cad"), exist_ok=True)
    for station, katalog, w, d, h, egen in STATIONS:
        out = os.path.join(HERE, "cad", katalog)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "parts_manifest_v1.json"), "w") as f:
            json.dump(machine_manifest(station, w, d, h), f, indent=1)
        if egen:
            with open(os.path.join(out, "negativrum_v1.json"), "w") as f:
                json.dump(machine_negativrum(station, w, d, h), f, indent=1)
    for katalog, payload in (("utilities_bim_exemplar_v1", utilities_manifest()),
                             ("materialflode_exemplar_v1", materialflow_manifest())):
        out = os.path.join(HERE, "cad", katalog)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "parts_manifest_v1.json"), "w") as f:
            json.dump(payload, f, indent=1)
    with open(os.path.join(HERE, "fabriksspec_v2.json"), "w") as f:
        json.dump(spec(), f, indent=1)
    print("wrote synthetic factory under", HERE)


if __name__ == "__main__":
    main()
