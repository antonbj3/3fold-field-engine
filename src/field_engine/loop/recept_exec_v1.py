#!/usr/bin/env python3
"""Generic recipe executor: a recipe (an ordered, declarative CAD action sequence in JSON) plus
parameters (theta) -> a build123d Part.

The recipe is the generator, not a script per part: adding a new part means adding one JSON file.

Recipe schema (see examples/recipes/ for an example):
{
  "schema": "recept_v1",
  "part": "<namn>",
  "frame_note": "<free text: what local (u,v,w) is relative to the master>",
  "params": {"<namn>": {"bounds": [lo, hi], "roll": "<free text -- what the parameter governs>"}},
  "steps": [
    {"op": "sketch_rect", "on": "XY", "w": "<param-or-number>", "h": "<...>", "fillet_r": "<...>"},
    {"op": "sketch_hole_pattern", "centers_uv": [[u,v], ...], "d": "<param>"},
    {"op": "extrude", "amount": "<param>", "mode": "ADD"},
    {"op": "fillet_vertical_edges", "r": "<param>"}
  ]
}
One "steps" row is one action. Any string value in a step that matches a key in "params" (or in theta
directly) is resolved against theta or the constants; anything else is used literally. There is no free
Python code in a recipe, only the declared ops.

Contract: execute(recipe: dict, theta: dict, consts: dict) -> build123d.Part, in the local frame; the
sketch plane defines the origin.
"""
from __future__ import annotations

from typing import Any, Dict

import build123d as bd

SUPPORTED_OPS = (
    "sketch_rect", "sketch_hole_pattern", "extrude", "fillet_vertical_edges", "fillet_all_edges",
    "box", "cylinder", "boolean_cut",
)


def _resolve(v: Any, theta: Dict[str, float], consts: Dict[str, float]) -> Any:
    if isinstance(v, str):
        if v in theta:
            return theta[v]
        if v in consts:
            return consts[v]
        raise KeyError(f"recept_exec_v1: string value {v!r} matches neither theta nor consts")
    return v


def execute(recipe: Dict[str, Any], theta: Dict[str, float], consts: Dict[str, float] | None = None):
    consts = consts or {}
    if recipe.get("schema") != "recept_v1":
        raise ValueError(f"unknown recipe schema {recipe.get('schema')!r} (expected recept_v1)")
    unknown = [s["op"] for s in recipe["steps"] if s["op"] not in SUPPORTED_OPS]
    if unknown:
        raise ValueError(f"unknown ops in the recipe: {unknown}; extend SUPPORTED_OPS, no free code")

    R = lambda v: _resolve(v, theta, consts)  # noqa: E731

    sketch = None
    part = None
    # Named intermediate results: "as" on box/cylinder/extrude/boolean_cut stores the Part under a
    # name in `named`, so boolean_cut can reference {"target": "<name>", "tool": "<name>"} with no
    # dangling references: `named` only holds Parts an earlier step actually built, and a KeyError is
    # raised otherwise.
    named: Dict[str, Any] = {}
    for step in recipe["steps"]:
        op = step["op"]
        if op == "sketch_rect":
            plane = getattr(bd.Plane, step.get("on", "XY"))
            w, h = R(step["w"]), R(step["h"])
            fr = R(step.get("fillet_r", 0.0)) or 0.0
            with bd.BuildSketch(plane) as sk:
                r = bd.Rectangle(w, h)
                if fr > 0:
                    bd.fillet(r.vertices(), radius=fr)
            sketch = sk.sketch
        elif op == "sketch_hole_pattern":
            if sketch is None:
                raise ValueError("sketch_hole_pattern requires a preceding sketch_rect")
            d = R(step["d"])
            centers = [(R(u), R(v)) for u, v in step["centers_uv"]]
            plane = getattr(bd.Plane, step.get("on", "XY"))
            # Measured kernel defect: the imperative builder idiom
            # ("with BuildSketch(plane): add(sketch); with Locations(*centers): Circle(..., SUBTRACT)")
            # silently subtracts nothing once a hole centre lies more than about 110-120 mm from the
            # sketch origin: centres at 91-110 mm cut correctly, centres beyond 120 mm leave a fully
            # uncut solid with no exception and no warning. The cause is the
            # BuildSketch + Locations + add combination, not the circle size or the plate dimensions.
            # Fix: use the sketch algebra API (plane * Location(...) * Circle(r), subtracted with
            # "sketch - circ"), verified correct for centre distances up to at least 300 mm.
            for u, v in centers:
                loc = plane * bd.Location((u, v, 0))
                sketch = sketch - (loc * bd.Circle(d / 2.0))
        elif op == "extrude":
            amount = R(step["amount"])
            if step.get("negate"):
                amount = -amount
            mode = getattr(bd.Mode, step.get("mode", "ADD"))
            with bd.BuildPart() as bp:
                if part is not None:
                    bd.add(part)
                bd.extrude(sketch, amount=amount, mode=mode if part is not None else bd.Mode.ADD)
            part = bp.part
            if step.get("as"):
                named[step["as"]] = part
        elif op == "box":
            # a named primitive: a standalone Part, not merged into `part`, for use as a boolean_cut
            # tool or as a "target" body step of its own.
            w, h, d = R(step["w"]), R(step["h"]), R(step["d"])
            at = step.get("at", [0.0, 0.0, 0.0])
            at = (R(at[0]), R(at[1]), R(at[2]))
            with bd.BuildPart() as bp:
                with bd.Locations(bd.Location(at)):
                    bd.Box(w, h, d)
            prim = bp.part
            if step.get("as"):
                named[step["as"]] = prim
            else:
                part = prim
        elif op == "cylinder":
            r, h = R(step["r"]), R(step["h"])
            at = step.get("at", [0.0, 0.0, 0.0])
            at = (R(at[0]), R(at[1]), R(at[2]))
            with bd.BuildPart() as bp:
                with bd.Locations(bd.Location(at)):
                    bd.Cylinder(r, h)
            prim = bp.part
            if step.get("as"):
                named[step["as"]] = prim
            else:
                part = prim
        elif op == "boolean_cut":
            # target/tool must be names an earlier step actually built (via "as"); KeyError otherwise
            tgt_name, tool_name = step["target"], step["tool"]
            if tgt_name not in named:
                raise KeyError(f"boolean_cut: target {tgt_name!r} is not a previously built named step (as)")
            if tool_name not in named:
                raise KeyError(f"boolean_cut: tool {tool_name!r} is not a previously built named step (as)")
            with bd.BuildPart() as bp:
                bd.add(named[tgt_name])
                bd.add(named[tool_name], mode=bd.Mode.SUBTRACT)
            part = bp.part
            if step.get("as"):
                named[step["as"]] = part
        elif op == "fillet_vertical_edges":
            r = R(step["r"])
            if r and r > 0 and part is not None:
                # edges parallel to the extrude axis (+-Z, the sketch normal) -- the plate's 4 side edges
                vertical = [e for e in part.edges() if abs(e.tangent_at(0).Z) > 0.999]
                if vertical:
                    part = bd.fillet(vertical, radius=r)
        elif op == "fillet_all_edges":
            r = R(step["r"])
            if r and r > 0 and part is not None:
                part = bd.fillet(part.edges(), radius=r)
        else:  # pragma: no cover -- guarded above
            raise ValueError(f"unknown op {op!r}")
    if part is None:
        raise ValueError("the recipe produced no solid")
    return part


if __name__ == "__main__":
    import json
    import sys
    recipe = json.load(open(sys.argv[1]))
    theta = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    p = execute(recipe, theta, {})
    print(f"OK: volume={p.volume:.1f}mm3 n_solids={len(p.solids())}")
    if len(sys.argv) > 3:
        bd.export_step(p, sys.argv[3])
        print(f"wrote {sys.argv[3]}")
