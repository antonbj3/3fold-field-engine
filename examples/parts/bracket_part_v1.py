#!/usr/bin/env python3
"""Part module for the bracket recipe, implementing the part_harness_v1 contract.

build(theta) executes examples/recipes/bracket_v1_recipe.json through recept_exec_v1; objectives are
the solid volume and an analytic deflection proxy for a plate in bending; the single constraint keeps
that deflection under a declared limit (value <= 0 is feasible).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src", "field_engine", "loop"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import recept_exec_v1 as RX  # noqa: E402

RECIPE_PATH = os.path.join(HERE, "..", "recipes", "bracket_v1_recipe.json")
CONSTS = {"corner_r": 8.0, "slot_w": 30.0, "slot_h": 40.0, "slot_d": 60.0}

BOUNDS = {"plate_w": (120.0, 180.0), "plate_h": (80.0, 120.0), "thk": (6.0, 14.0),
          "hole_d": (6.0, 12.0)}
OBJECTIVE_NAMES = ["volume_mm3", "deflection_proxy_mm"]
CONSTRAINT_NAMES = ["deflection_margin"]

E_MPA = 71000.0      # aluminium
LOAD_N = 800.0       # declared reference load at the free edge
DEFLECTION_LIMIT_MM = 0.6


def _recipe() -> Dict[str, Any]:
    with open(RECIPE_PATH) as f:
        return json.load(f)


def build(theta: Dict[str, float]):
    """Builds the bracket for one theta and returns the solid."""
    return RX.execute(_recipe(), theta, CONSTS)


def _deflection_proxy(theta: Dict[str, float]) -> float:
    """Tip deflection of a cantilever plate strip, delta = F L^3 / (3 E I), I = b t^3 / 12."""
    L = theta["plate_w"] / 2.0
    b = theta["plate_h"]
    t = theta["thk"]
    I = b * t ** 3 / 12.0
    return LOAD_N * L ** 3 / (3.0 * E_MPA * I)


def objectives(shape, theta: Dict[str, float]) -> Dict[str, float]:
    """Returns the solid volume in mm^3 and the deflection proxy in mm (both minimised)."""
    return {"volume_mm3": float(shape.volume), "deflection_proxy_mm": _deflection_proxy(theta)}


def constraints(shape, theta: Dict[str, float]) -> Dict[str, float]:
    """Returns the deflection margin; <= 0 is feasible."""
    return {"deflection_margin": _deflection_proxy(theta) - DEFLECTION_LIMIT_MM}


def make_part():
    """Wires the contract into one ParametricPart."""
    from part_harness_v1 import ParametricPart
    return ParametricPart(name="bracket_v1", bounds=BOUNDS, build_fn=build,
                          objectives_fn=objectives, objective_names=OBJECTIVE_NAMES,
                          constraints_fn=constraints, constraint_names=CONSTRAINT_NAMES)


if __name__ == "__main__":
    theta = {"plate_w": 150.0, "plate_h": 100.0, "thk": 10.0, "hole_d": 9.0}
    part = build(theta)
    print(json.dumps({"theta": theta, "objectives": objectives(part, theta),
                      "constraints": constraints(part, theta),
                      "n_solids": len(part.solids())}, indent=1))
