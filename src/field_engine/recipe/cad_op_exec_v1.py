#!/usr/bin/env python3
"""cad_op_exec_v1.py -- deterministic executor: a list of op dicts becomes build123d geometry.

Op dicts are dispatched directly to build123d API calls in memory. No source code is synthesised,
there is no exec(), no eval(), no temporary.py file and no subprocess for geometry generation.

Coverage: of the op types enumerated in cad_op_schema_v1, all but "reference" have a handler that is
exercised by the selftest below -- sketch_2d, sketch_profile, extrude, revolve, loft, sweep, the
three booleans, hole, fillet, chamfer, shell, mirror, pattern_linear, pattern_circular, draft,
sheet_bend, thread_cosmetic and rib. "reference" raises NotImplementedError: an untested handler is
a worse failure class than an honest escalation.

API:
exec_ops(ops: list[dict|OpStep], params: dict|None = None, log_path: str|None = None)
-> {"solids": dict[str, Any], "log": list[dict], "population_warnings": list[dict]}
resolve_expr(expr: str, params: dict) -> float

Per-op log: every executed op appends {op_id, op_type, volymdelta_mm3, wall_ms, status, reason}. For
boolean_cut and hole a volume delta of about zero means the tool removed no material, which is
recorded as status=FAIL, reason=MISSED_CUT rather than passing silently.

ALL-population guard: a fillet/chamfer/draft whose "selector" is omitted acts on the whole edge or
face population of its target. Before such a handler runs, exec_ops calls
geometri_selektor_v1.check_all_population on the current target. If the op declares
"expected_population" and the measured population has drifted, RefusalError(POPULATION_DRIFT) is
raised before OCC is touched, carrying op_id and the op's index in the chain. A recipe without that
key stays backward compatible: the guard warns instead of raising, the measured population is auto-
recorded into the op's log row, and exec_ops returns the undeclared-population warnings drained
during the run as out["population_warnings"].

Symbolic expressions: any value anywhere in an op may be {"expr": "BORE_D/2"} instead of a number,
resolved against the flat `params` dict before the op's guard and handler run, so handlers and
selectors only ever see fully concrete dicts. resolve_expr parses with ast in "eval" mode and walks
a closed node set; every other node raises ValueError naming the disallowed node type. Arithmetic
that fails (division by zero, overflow, a complex result from a fractional power of a negative base)
is re-raised as a legible ValueError naming the sub-expression, the operand values and the params
used. When an op carried expressions its log row gains "expr_resolutions" with one entry per
resolved expression (path, expr, resolved_value, params_used). With params falsy the resolve walk is
skipped entirely and the op dict handed in is passed through unchanged.

Run the selftest with `python cad_op_exec_v1.py`; it prints a JSON report and exits non-zero if a
check fails.
"""
from __future__ import annotations

import ast
import json
import math
import os
import sys
import time
from typing import Any

import build123d as bd
from OCP.BRepOffsetAPI import BRepOffsetAPI_DraftAngle
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt
from OCP.TopoDS import TopoDS

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(_KERNEL_DIR, "artifacts")
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)
from geometri_selektor_v1 import (  # noqa: E402 -- edges/faces are addressed geometrically, never by index
    selector_match, check_all_population, record_expected_population, drain_population_warnings,
    RefusalError, geometric_edge_population, local_wall_thickness_mm, REASON_SCALE_DEGENERATE,
)
from iso_thread_table_v1 import parse_designation as _iso_parse_designation  # noqa: E402

_MISSED_CUT_TOL_MM3 = 1e-6

# --------------------------------------------------------------------------------------------------
# - SYMBOLIC EXPRESSIONS -- see module docstring.
# -------------------------------------------------------------------------------------------------
_EXPR_ALLOWED_BINOPS: dict[type, Any] = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_EXPR_ALLOWED_UNARYOPS: dict[type, Any] = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}


def _expr_eval_node(node: ast.AST, params: dict, expr_str: str):
    """Closed-set AST interpreter -- see module docstring SAFE EVALUATOR. Every branch not explicitly
    handled falls through to the final `raise` (no default-permissive case)."""
    if isinstance(node, ast.Expression):
        return _expr_eval_node(node.body, params, expr_str)
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"expr {expr_str!r}: disallowed constant {v!r} (only int/float literals allowed)")
        return v
    if isinstance(node, ast.Name):
        if node.id not in params:
            raise ValueError(
                f"expr {expr_str!r}: unknown name {node.id!r} -- not present in params "
                f"{sorted(params)} (an undeclared name is a configuration error, "
                f"never silently treated as 0)"
            )
        v = params[node.id]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"expr {expr_str!r}: param {node.id!r} = {v!r} is not numeric")
        return v
    if isinstance(node, ast.BinOp):
        fn = _EXPR_ALLOWED_BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"expr {expr_str!r}: disallowed binary operator {type(node.op).__name__}")
        left = _expr_eval_node(node.left, params, expr_str)
        right = _expr_eval_node(node.right, params, expr_str)
        try:
            sub_repr = ast.unparse(node)
        except Exception:  # noqa: BLE001 -- unparse is cosmetic only, never let it hide the real error
            sub_repr = f"<{type(node.op).__name__} left={left!r} right={right!r}>"
        try:
            result = fn(left, right)
        # an earlier stress-test corpus legC (0.4545 frac_structured): a raw
        # ZeroDivisionError/OverflowError/TypeError/ ValueError from the arithmetic itself must
        # never propagate uncaught -- wrap it into a legible ValueError naming the sub-expression,
        # the operand values, and the params that fed it (never a silent wrong value, per
        # resolve_expr's own contract -- symboliska_selektorer_v1 PROVENANCE).
        except (ZeroDivisionError, OverflowError, TypeError, ValueError) as e:
            raise ValueError(
                f"expr {expr_str!r}: {type(e).__name__} evaluating '{sub_repr}' (left={left!r}, "
                f"right={right!r}) -- {e}; params={params}"
            ) from e
        if isinstance(result, complex):
            # e.g. (-4.0)**0.5 -- Python's ** silently RETURNS a complex number instead of raising;
            # only float(...) downstream would ever surface it (as an opaque TypeError far from the
            # cause). Catch it explicitly, at the exact sub-expression, while left/right are still known.
            raise ValueError(
                f"expr {expr_str!r}: '{sub_repr}' (left={left!r}, right={right!r}) produced a complex "
                f"result -- negative base with a fractional/non-integer exponent has no real-valued "
                f"answer (only real arithmetic is supported); params={params}"
            )
        return result
    if isinstance(node, ast.UnaryOp):
        fn = _EXPR_ALLOWED_UNARYOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"expr {expr_str!r}: disallowed unary operator {type(node.op).__name__}")
        return fn(_expr_eval_node(node.operand, params, expr_str))
    raise ValueError(
        f"expr {expr_str!r}: disallowed syntax node {type(node).__name__} (allowed: numeric "
        f"literals, names-in-params, + - * / // % **, unary + -; no calls/attrs/subscripts/compares/"
        f"boolops/lambdas/comprehensions/string-or-collection-literals)"
    )


def resolve_expr(expr: str, params: dict | None) -> float:
    """Safe arithmetic-only evaluator (module docstring SAFE EVALUATOR): NO eval()/exec(), an
    ast.parse(mode='eval') tree walked by a closed-set interpreter (numeric literals, names-in-params,
    + - * / // % ** and unary +/-, nothing else). Raises ValueError with a legible message naming the
    offending token on an unknown name or a disallowed AST node -- never a silent wrong value."""
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError(f"expr must be a non-empty string, got {expr!r}")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"expr {expr!r}: invalid syntax: {e}") from e
    return float(_expr_eval_node(tree, params or {}, expr))


def _expr_names_used(expr: str) -> list[str]:
    """Names referenced by `expr`, for provenance (module docstring PROVENANCE) -- re-parses rather
    than threading a names-accumulator through _expr_eval_node, since this is cold/log-path-only, not
    hot geometry code."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return []
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})


def _resolve_op_exprs(data: dict, params: dict) -> tuple[dict, list[dict]]:
    """Walk `data` (one op's raw dict) recursively; every nested dict of the EXACT shape {"expr": str}
    is replaced by resolve_expr(expr, params) -- everywhere in the op, not just top-level scalar
    fields, so a selector sub-dict's "radius"/"min_length" (the exact literals benchmark_a_v1.json's
    prior-art finding named) resolve the same way a top-level "diameter" does. Returns (resolved_data,
    provenance) -- provenance is [] when the op has no expr anywhere (the common/legacy case)."""
    provenance: list[dict] = []

    def walk(obj, path: str):
        if isinstance(obj, dict):
            if set(obj.keys()) == {"expr"} and isinstance(obj.get("expr"), str):
                expr = obj["expr"]
                value = resolve_expr(expr, params)
                provenance.append({
                    "path": path, "expr": expr, "resolved_value": value,
                    "params_used": _expr_names_used(expr),
                })
                return value
            return {k: walk(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(obj)]
        return obj

    resolved = walk(data, "")
    return resolved, provenance

# op-types whose selector=None/omitted chain position is an "ALL" selection over the WHOLE target's
# population of this entity kind -- guarded by check_all_population before the destructive handler
# runs (see module docstring, ALL-POPULATIONSVAKTEN). pattern_linear/pattern_circular/mirror/shell
# are NOT "ALL over a selector-addressable population" in this sense (no selector key at all, or --
# shell's faces_to_remove -- an explicit list, never an implicit "everything").
_ALL_CHAIN_ENTITY = {"fillet": "edge", "chamfer": "edge", "draft": "face"}

AXIS_PLANE_ORIGIN = {
    # axis -> (x_dir, z_dir) for a bd.Plane whose NORMAL (z_dir) is that world axis,
    # used to build a hole/cylinder's cross-section perpendicular to the requested axis.
    "Z": ((1, 0, 0), (0, 0, 1)),
    "X": ((0, 1, 0), (1, 0, 0)),
    "Y": ((0, 0, 1), (0, 1, 0)),
}

# axis -> unit direction vector, world axis THROUGH THE GIVEN ORIGIN (default world origin) --
# used by revolve + pattern_circular. A dict {"origin":[..],"direction":[..]} is also accepted for a
# non-world-origin axis; this is NOT a new op-type (still revolve/pattern_circular), just an internal
# representation for the existing "axis" key -- no new transform-op introduced (pilot rule, formverb_v1).
AXIS_VECTOR = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}

# named mirror planes, world-origin-centered (a {"origin","x_dir","z_dir"} dict is also accepted,
# reusing _sketch_plane -- same convention as sketch_2d's own "plane" field).
_MIRROR_PLANES = {"XY": bd.Plane.XY, "XZ": bd.Plane.XZ, "YZ": bd.Plane.YZ}

# sketch_2d "shapes" sub-types that build an OPEN path (bd.BuildLine) instead of a closed profile
# (bd.BuildSketch) -- feeds sweep's path_ref. Still the SAME op-type (sketch_2d); this only extends
# the un-validated freeform "shapes" sub-vocabulary (cad_op_schema_v1 does not enumerate shape
# contents), so the 16-type closed op-vocabulary is untouched.
_PATH_SHAPE_TYPES = {"line", "arc"}


# ---------------------------------------------------------------------------------------------------
# per-op handlers -- each: handler(data: dict, ctx: dict[str, Any], params: dict) -> result_object
# ---------------------------------------------------------------------------------------------------
def _sketch_plane(plane_spec: dict | None) -> bd.Plane:
    if not plane_spec:
        return bd.Plane.XY
    origin = tuple(plane_spec.get("origin", (0, 0, 0)))
    x_dir = tuple(plane_spec.get("x_dir", (1, 0, 0)))
    z_dir = tuple(plane_spec.get("z_dir", (0, 0, 1)))
    return bd.Plane(origin=origin, x_dir=x_dir, z_dir=z_dir)


def _mode_of(shape: dict) -> bd.Mode:
    m = shape.get("mode", "add")
    if m == "add":
        return bd.Mode.ADD
    if m == "subtract":
        return bd.Mode.SUBTRACT
    raise ValueError(f"invalid shape mode {m!r} (must be 'add' or 'subtract')")


def _build_path(plane: bd.Plane, shapes: list):
    """sweep's path_ref target: an OPEN wire (bd.BuildLine), local 2D coords in `plane` -- see
    _PATH_SHAPE_TYPES docstring at module top."""
    with bd.BuildLine(plane) as bl:
        for shape in shapes:
            stype = shape.get("type")
            if stype == "line":
                bd.Line(tuple(shape["start"]), tuple(shape["end"]))
            elif stype == "arc":
                bd.CenterArc(
                    tuple(shape["center"]), shape["radius"],
                    shape.get("start_angle_deg", 0.0), shape["arc_size_deg"],
                )
            else:
                raise NotImplementedError(f"path shape type {stype!r} (pilot scope: line|arc)")
    return bl.line


def _handle_sketch_2d(data: dict, ctx: dict, params: dict):
    plane = _sketch_plane(data.get("plane"))
    shapes = data["shapes"]
    shape_types = {s.get("type") for s in shapes}
    if shape_types and shape_types <= _PATH_SHAPE_TYPES:
        return _build_path(plane, shapes)
    if shape_types & _PATH_SHAPE_TYPES:
        raise ValueError(
            f"sketch_2d: cannot mix path shape types {_PATH_SHAPE_TYPES} with profile shape types "
            f"in one op (got {sorted(shape_types)})"
        )
    with bd.BuildSketch(plane) as sk:
        for shape in data["shapes"]:
            stype = shape.get("type")
            bmode = _mode_of(shape)
            center = tuple(shape.get("center", (0, 0)))
            if stype == "rectangle":
                w, h = shape["width"], shape["height"]
                if center == (0, 0):
                    bd.Rectangle(w, h, mode=bmode)
                else:
                    with bd.Locations(center):
                        bd.Rectangle(w, h, mode=bmode)
            elif stype == "circle":
                r = shape["radius"]
                if center == (0, 0):
                    bd.Circle(r, mode=bmode)
                else:
                    with bd.Locations(center):
                        bd.Circle(r, mode=bmode)
            else:
                raise NotImplementedError(f"sketch shape type {stype!r} (pilot scope: rectangle|circle)")
    return sk.sketch


def _handle_extrude(data: dict, ctx: dict, params: dict):
    mode = data.get("mode", "new")
    if mode != "new":
        raise NotImplementedError(f"extrude mode {mode!r} (pilot scope: 'new' only)")
    sk = ctx[data["sketch_ref"]]
    taper = data.get("taper", 0.0)  # U86.1 cheap draft layer -- build123d.extrude() already took this
    return bd.extrude(sk, amount=data["amount"], taper=taper)


def _handle_loft(data: dict, ctx: dict, params: dict):
    mode = data.get("mode", "new")
    if mode != "new":
        raise NotImplementedError(f"loft mode {mode!r} (pilot scope: 'new' only)")
    refs = data["sketch_refs"]
    if len(refs) < 2:
        raise ValueError("loft requires >=2 sketch_refs")
    ruled = data.get("ruled", True)
    faces = [ctx[r].faces()[0] for r in refs]
    # MEASURED KERNEL BUG: the earlier implementation lofted PAIRWISE and fused the segments with
    # Python "+" for three or more sketch_refs. Each two-section segment is its own closed solid
    # sharing an exactly coincident end face with the next one; BRepCheck_Analyzer accepts both
    # solids individually, but a later boolean_union/boolean_cut against the fused body consistently
    # raised "ValueError: Null TopoDS_Shape object" (reproduced on a three-profile loft: every
    # downstream union or cut failed, even against a trivial cylindrical boss that otherwise unions
    # cleanly). Root cause: the internal fuse leaves a non-manifold, doubled seam at the intermediate
    # profile which a later boolean solver does not tolerate, even though the shape looks valid in
    # isolation. Fix: bd.loft() natively accepts two or more sections in ONE call and builds a single
    # ThruSections body with no internal fuse -- no seam, the same result as before for the
    # two-section case (measured: identical volume and solid count at N=2), and downstream unions
    # succeed at N=3. Fully backward compatible.
    return bd.loft(faces, ruled=ruled)


def _handle_boolean_union(data: dict, ctx: dict, params: dict):
    result = ctx[data["base_ref"]]
    for t in data["tool_refs"]:
        result = result + ctx[t]
    return result


def _handle_boolean_cut(data: dict, ctx: dict, params: dict):
    result = ctx[data["base_ref"]]
    for t in data["tool_refs"]:
        result = result - ctx[t]
    return result


def _handle_boolean_intersect(data: dict, ctx: dict, params: dict):
    result = ctx[data["base_ref"]]
    for t in data["tool_refs"]:
        result = result & ctx[t]
    return result


def _handle_hole(data: dict, ctx: dict, params: dict):
    target = ctx[data["target_ref"]]
    center = tuple(data["center"])
    diameter = data["diameter"]
    axis = data.get("axis", "Z")
    through = data.get("through", False)
    if axis not in AXIS_PLANE_ORIGIN:
        raise NotImplementedError(f"hole axis {axis!r} (pilot scope: X|Y|Z)")
    if through:
        bb = target.bounding_box()
        extent = {"X": bb.max.X - bb.min.X, "Y": bb.max.Y - bb.min.Y, "Z": bb.max.Z - bb.min.Z}[axis]
        depth = max(extent * 2.0, diameter * 2.0)
    else:
        if "depth" not in data:
            raise ValueError("hole requires 'depth' when through=False")
        depth = data["depth"]
    shift = depth / 2.0 if through else 0.0
    if axis == "Z":
        origin = (center[0], center[1], center[2] - shift)
    elif axis == "X":
        origin = (center[0] - shift, center[1], center[2])
    else:
        origin = (center[0], center[1] - shift, center[2])
    x_dir, z_dir = AXIS_PLANE_ORIGIN[axis]
    plane = bd.Plane(origin=origin, x_dir=x_dir, z_dir=z_dir)
    with bd.BuildSketch(plane) as sk:
        bd.Circle(diameter / 2.0)
    cyl = bd.extrude(sk.sketch, amount=depth)
    return target - cyl


def _axis_from_spec(axis_spec) -> bd.Axis:
    """Resolve an 'axis' field (str 'X'|'Y'|'Z' through world origin, or a
    {"origin":[..],"direction":[..]} dict) to a bd.Axis. Shared by revolve + pattern_circular."""
    if isinstance(axis_spec, str):
        if axis_spec not in AXIS_VECTOR:
            raise NotImplementedError(f"axis {axis_spec!r} (pilot scope: X|Y|Z)")
        return bd.Axis((0, 0, 0), AXIS_VECTOR[axis_spec])
    if isinstance(axis_spec, dict):
        return bd.Axis(tuple(axis_spec.get("origin", (0, 0, 0))), tuple(axis_spec.get("direction", (0, 0, 1))))
    raise ValueError(f"axis must be str 'X'|'Y'|'Z' or dict{{origin,direction}}, got {type(axis_spec).__name__}")


def _edge_sort_key(e):
    """04S1: center+length ALONE collides on 66/50325 (0.13%) real edges -- two distinct edges sharing
    a rounded center+ length (e.g. two edges of an arc/circle pair, or mirrored edges) then fall
    back on OCC's
    *unstable* enumeration order for the tie, reintroducing the exact nondeterminism this key exists
    to remove. Extend the key with geom_type + radius (0-arg-safe, faces without a radius sort as
    -1.0) + a CANONICAL (order-independent) endpoint pair -- an edge's start/end can itself come out
    reversed between two fresh OCC processes for the SAME edge, so sorting the two endpoints before
    including them keeps the key stable under that reversal too, not just under center/length ties.
    """
    try:
        c = e.center()
        try:
            r = round(float(e.radius), 6)
        except Exception:
            r = -1.0
        try:
            p0 = tuple(round(v, 6) for v in tuple(e.start_point()))
            p1 = tuple(round(v, 6) for v in tuple(e.end_point()))
            endpoints = tuple(sorted((p0, p1)))
        except Exception:
            endpoints = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        return (round(c.X, 6), round(c.Y, 6), round(c.Z, 6), round(e.length, 6),
                e.geom_type.name, r, endpoints)
    except Exception:
        return (0.0, 0.0, 0.0, 0.0, "", -1.0, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))


def _resolve_target_edges(target, selector: dict | None) -> list:
    """selector omitted => ALL edges of target (whole-solid fillet/chamfer); selector present =>
    geometri_selektor_v1.selector_match (geometric, index-independent -- see module import).

    SORTED by a geometric key (center+length) before returning -- MEASURED (formlyftet A5.2,
    2026-08-14): shape.edges() enumeration order is NOT stable across process runs for a byte-
    identical op recipe (OCC's underlying TopTools iteration is allocator/pointer-hash-order
    dependent), and bd.fillet()/bd.chamfer() CAN produce a topologically different result (different
    downstream edge COUNT, not just a relabeling) depending on the ORDER edges are handed to the
    builder -- reproduced directly: the SAME 5-op recipe run twice in two fresh processes gave 8 vs
    16 post-fillet ring edges on an identical target. A deterministic input order removes that
    variable; it does not by itself prove the builder is now order-INSENSITIVE, but it makes THIS
    pipeline's own repeated runs reproducible, which non-determinism itself forecloses. Never
    silently reorders the SELECTOR's matched SET (identical set, canonical order).
    """
    if selector is None:
        edges = list(target.edges())
    else:
        edges = list(selector_match(target, selector))
    return sorted(edges, key=_edge_sort_key)


def _resolve_scaled_value(data: dict, edges: list, explicit_key: str, target=None) -> tuple[float, dict]:
    """radius XOR radius_frac. Resolve fillet's `radius` / chamfer's `distance` to a concrete positive
    float. TWO forms:
    - explicit_key present (radius|distance): pass-through, unchanged legacy behaviour.
    - "radius_frac" present: value = (MEASURED min edge length) * radius_frac. THE MEASURED
    POPULATION IS NOT `edges` (the safety-FILTERED population this op will actually touch, e.g.
    after min_length/vertex_safe_k have excluded short/vertex-adjacent edges) -- it is `target`'s
    GEOMETRIC population under the same selector, via
    geometri_selektor_v1.geometric_edge_population, i.e. the selector's geometry-defining filters
    ONLY, with the safety filters (min_length/ max_length/vertex_safe_k) skipped. MEASURED: scaling
    off the SAFETY-filtered population's own min length inflates the resolved radius past what the
    surviving edges can individually tolerate -- a min_length/vertex_safe_k-filtered population's
    min edge length is ALWAYS >= the shape's true local minimum BY CONSTRUCTION of the filter
    itself. `target=None` (caller has no live shape, e.g. a unit test measuring `edges` directly)
    falls back to measuring `edges` itself -- exec_ops' own handlers always pass `target`, so this
    fallback is a defensive-only path, not the production path. NO clamp/cap/floor is applied
    (unchanged from adaptiv_radie_rib_v1: the analytical cube reference value requires
    radius==L*radius_frac EXACTLY, which a cap/floor would silently falsify). XOR is already
    enforced by cad_op_schema_v1.validate_op when a recipe passes through validation first, but this
    function re-checks defensively -- exec_ops is deliberately NOT coupled to validate_op (module
    docstring), so a caller CAN reach this handler with an unvalidated op dict.
    """
    frac = data.get("radius_frac")
    explicit = data.get(explicit_key)
    if frac is None and explicit is None:
        raise ValueError(f"requires either {explicit_key!r} or 'radius_frac' (neither given)")
    if frac is not None and explicit is not None:
        raise ValueError(f"cannot specify both {explicit_key!r} and 'radius_frac' -- exactly one")
    if frac is None:
        value = float(explicit)
        return value, {"mode": "explicit", "resolved_value": value}
    if frac <= 0:
        raise ValueError(f"radius_frac must be > 0, got {frac}")
    selector = data.get("selector")
    if target is not None:
        measure_edges = geometric_edge_population(target, selector)
    else:
        measure_edges = edges
    lens = [float(e.length) for e in measure_edges]
    if not lens:
        raise ValueError("radius_frac requires a nonempty edge population to measure (0 edges given)")
    min_len = min(lens)
    if min_len <= 0:
        raise ValueError(f"radius_frac: measured population min edge length is {min_len} (<=0), cannot scale")
    value = min_len * float(frac)
    resolution = {
        "mode": "radius_frac", "radius_frac": float(frac),
        "measured_min_edge_length_mm": round(min_len, 6),
        "carrier": "selector_geometric_population_pre_safety_filter" if selector is not None else "target_ref_all_edges",
        "n_population_edges": len(lens),
        "n_geometric_population_edges": len(measure_edges),
        "n_safety_filtered_population_edges": len(edges),
        "resolved_value": round(value, 6),
    }
    return value, resolution


_SCALE_DEGENERATE_REL_TOL = 1e-6  # REFRAME DECLARED (discipline #8) -- see below, not the mandate's own
# "~0.5%" suggestion. MEASURED against the ACTUAL OCC crash boundary of an internal recipe fixture
# earlier stress-test corpus/legC_a2_SEGFAULT_radius_equals_shell_ thickness.json (true wall
# thickness 5.0mm, confirmed via BRepExtrema): radius=5.0 SIGSEGVs (exit 139); radius=4.9999/5.0001
# (0.002% off) and 4.999/5.001 (0.02% off) ALL give a clean ValueError, no crash -- the true
# degeneracy is razor-thin (bit-EXACT equality), not a broad 0.5%-wide unsafe band. A 0.5% tolerance
# would WRONGLY reclassify the mandate's OWN acceptance case (4.99/5.01, ~0.2% off, explicitly
# required to keep its previous behaviour, i.e. a plain ValueError) as a RefusalError instead --
# verified: at tol=0.005, rel_dev(4.99,5.0)=0.002 < 0.005 WOULD have wrongly refused it. 1e-6 (~
# 0.0001%) catches the measured exact-match crash (rel_dev=0.0) plus float round-off noise on a
# radius_frac-resolved value (double-precision arithmetic accumulates <<1e-9 relative error, not
# 1e-4), while leaving every measured non-crashing case (>=0.002% off) exactly as before.


def _sigsegv_guard_scale_degenerate(target, edges: list, value: float, verb: str, op_id) -> None:
    """PROCESS_CRASH_SIGSEGV guard. MEASURED repro: a fillet radius EXACTLY equal to the shelled body's
    wall thickness (5.0mm == 5.0mm) crashes the OCC builder with SIGSEGV (exit 139, no Python
    exception --the WORST vagranskorrekthet class, since neither a ValueError nor a RefusalError is
    even raised for the caller to catch). 4.99/5.01mm (0.2% off) instead raise a clean ValueError --
    confirming the degeneracy is at the EXACT boundary, not a broad unsafe band.

    REFRAME DECLARED: the ORIGINAL DECLARED METHOD measured the GLOBAL minimum BRepExtrema distance
    from `edges`' adjacent faces to EVERY OTHER face of `target`. MEASURED FALSE POSITIVE on a
    hollow 90x90 tube with a 6 mm wall and root_radius=6 mm (the same nominal degeneracy): the
    global scan blocked
    a fillet that MEASURABLY does NOT crash OCC (isolated repro, guard disabled: clean success,
    volume unchanged) -- it was matching the tube's OWN opposite-side wall, an unrelated face that
    has nothing to do with the edges actually being filleted. See
    geometri_selektor_v1.local_wall_thickness_mm's docstring for the full mechanism + the re-
    verified true-positive (this SIGSEGV recipe's top-rim edges still measure their OWN local
    ribbon-face partner at exactly 5.0mm and still refuse). DECLARED METHOD
    NOW: measure the LOCAL wall/flange thickness AT `edges` themselves -- geometri_selektor_v1.
    local_wall_thickness_mm scans only each candidate edge's own directly-adjacent faces' own OTHER
    edges (never the whole body). BEFORE the destructive OCC call: if a thickness is measurable
    (None => no parallel-alongside partner found near any candidate edge -- no SIGSEGV risk from
    this mechanism, no guard fires) and `value` is within _SCALE_DEGENERATE_REL_TOL (~0.5%) of it,
    raises RefusalError(reason=SCALE_DEGENERATE) naming BOTH numbers -- a structured refusal instead
    of a crashed interpreter.
    """
    thickness = local_wall_thickness_mm(target, edges, near_hint_mm=value)
    if thickness is None or thickness <= 0:
        return
    rel_dev = abs(value - thickness) / thickness
    if rel_dev > _SCALE_DEGENERATE_REL_TOL:
        return
    diagnosis = {
        "reason": REASON_SCALE_DEGENERATE, "op_id": op_id, "verb": verb,
        "resolved_value_mm": round(value, 6), "measured_local_wall_thickness_mm": round(thickness, 6),
        "rel_dev": round(rel_dev, 6), "tol": _SCALE_DEGENERATE_REL_TOL,
        "note": (
            f"{verb} value {value:.6f}mm is within {_SCALE_DEGENERATE_REL_TOL * 100:.2f}% of the "
            f"measured local wall thickness {thickness:.6f}mm -- the OCC {verb} builder SIGSEGVs "
            f"(exit 139, no exception) at this exact degeneracy; refusing structurally BEFORE the "
            f"destructive call instead of crashing the interpreter"
        ),
    }
    raise RefusalError(0, 0, [], reason=REASON_SCALE_DEGENERATE, diagnosis=diagnosis)


def _handle_fillet(data: dict, ctx: dict, params: dict):
    target = ctx[data["target_ref"]]
    edges = _resolve_target_edges(target, data.get("selector"))
    if not edges:
        raise ValueError("fillet: selector matched 0 edges (nothing to fillet)")
    radius, resolution = _resolve_scaled_value(data, edges, "radius", target=target)
    if radius <= 0:
        raise ValueError(f"fillet requires radius>0, got {radius}")
    _sigsegv_guard_scale_degenerate(target, edges, radius, "fillet", data.get("id"))
    try:
        result = bd.fillet(edges, radius=radius)
    except NotImplementedError:
        raise
    except Exception as e:  # noqa: BLE001 -- normalize OCC's raw exception class to an honest, clear error
        raise ValueError(
            f"fillet failed (radius={radius} likely exceeds the max valid radius for the selected "
            f"{len(edges)} edge(s)): {type(e).__name__}:{e}"
        ) from e
    result._op_value_resolution = resolution  # folded into the log row by exec_ops, see docstring there
    return result


def _handle_chamfer(data: dict, ctx: dict, params: dict):
    target = ctx[data["target_ref"]]
    edges = _resolve_target_edges(target, data.get("selector"))
    if not edges:
        raise ValueError("chamfer: selector matched 0 edges (nothing to chamfer)")
    distance, resolution = _resolve_scaled_value(data, edges, "distance", target=target)
    if distance <= 0:
        raise ValueError(f"chamfer requires distance>0, got {distance}")
    _sigsegv_guard_scale_degenerate(target, edges, distance, "chamfer", data.get("id"))
    try:
        result = bd.chamfer(edges, length=distance)
    except NotImplementedError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"chamfer failed (distance={distance} likely exceeds the max valid length for the "
            f"selected {len(edges)} edge(s)): {type(e).__name__}:{e}"
        ) from e
    result._op_value_resolution = resolution
    return result


def _handle_revolve(data: dict, ctx: dict, params: dict):
    mode = data.get("mode", "new")
    if mode != "new":
        raise NotImplementedError(f"revolve mode {mode!r} (pilot scope: 'new' only)")
    sk = ctx[data["sketch_ref"]]
    axis = _axis_from_spec(data["axis"])
    angle = data.get("angle_deg", 360.0)
    if not (0.0 < angle <= 360.0):
        raise ValueError(f"revolve angle_deg must be in (0,360], got {angle}")
    try:
        return bd.revolve(sk, axis=axis, revolution_arc=angle)
    except NotImplementedError:
        raise
    except Exception as e:  # noqa: BLE001 -- normalize OCC's raw exception (e.g. StdFail_NotDone)
        raise ValueError(
            f"revolve failed (profile likely straddles or crosses the axis): {type(e).__name__}:{e}"
        ) from e


def _handle_sweep(data: dict, ctx: dict, params: dict):
    mode = data.get("mode", "new")
    if mode != "new":
        raise NotImplementedError(f"sweep mode {mode!r} (pilot scope: 'new' only)")
    section = ctx[data["sketch_ref"]]
    path = ctx[data["path_ref"]]
    try:
        result = bd.sweep(section, path=path)
    except NotImplementedError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"sweep failed: {type(e).__name__}:{e}") from e
    vol = getattr(result, "volume", None)
    if vol is not None and abs(vol) < 1e-6:
        # SILENT-WRONG-GEOMETRY GUARD (same class as MISSED_CUT): a profile plane not perpendicular
        # to the path tangent at its start produces a near-zero-volume, topologically "valid" but
        # physically degenerate sweep -- refuse rather than pass it through (measured, formverb_v1).
        raise ValueError(
            "sweep produced a degenerate (~0 volume) result -- profile plane is likely not "
            "perpendicular to the path's tangent at its start"
        )
    return result


def _handle_shell(data: dict, ctx: dict, params: dict):
    target = ctx[data["target_ref"]]
    thickness = data["thickness"]
    if thickness <= 0:
        raise ValueError(f"shell requires thickness>0, got {thickness}")
    faces_spec = data.get("faces_to_remove")
    faces: list = []
    if faces_spec:
        for sel in faces_spec:
            faces.extend(selector_match(target, sel))
    try:
        return target.hollow(faces, -float(thickness), kind=bd.Kind.INTERSECTION)
    except NotImplementedError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"shell failed (thickness={thickness} likely exceeds half the smallest extent): "
            f"{type(e).__name__}:{e}"
        ) from e


def _handle_pattern_linear(data: dict, ctx: dict, params: dict):
    target = ctx[data["target_ref"]]
    count = data["count"]
    spacing = data["spacing"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"pattern_linear requires an integer count>=1, got {count!r}")
    if spacing <= 0:
        raise ValueError(f"pattern_linear requires spacing>0, got {spacing}")
    vec = bd.Vector(*data["direction"])
    if vec.length < 1e-9:
        raise ValueError("pattern_linear requires a nonzero direction vector")
    unit = vec.normalized()
    copies = []
    for i in range(count):
        off = unit * (spacing * i)
        copies.append(bd.Pos(off.X, off.Y, off.Z) * target)
    return bd.Compound(copies)


def _handle_pattern_circular(data: dict, ctx: dict, params: dict):
    target = ctx[data["target_ref"]]
    count = data["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"pattern_circular requires an integer count>=1, got {count!r}")
    angle_total = data.get("angle_deg", 360.0)
    if not (0.0 < angle_total <= 360.0):
        raise ValueError(f"pattern_circular angle_deg must be in (0,360], got {angle_total}")
    axis_spec = data["axis"]
    if isinstance(axis_spec, str):
        if axis_spec not in AXIS_VECTOR:
            raise NotImplementedError(f"axis {axis_spec!r} (pilot scope: X|Y|Z)")
        origin, direction = (0, 0, 0), AXIS_VECTOR[axis_spec]
    elif isinstance(axis_spec, dict):
        origin = tuple(axis_spec.get("origin", (0, 0, 0)))
        direction = tuple(axis_spec.get("direction", (0, 0, 1)))
    else:
        raise ValueError("pattern_circular axis must be str 'X'|'Y'|'Z' or dict{origin,direction}")
    # full 360 => equal division (count instances, no coincident last==first); a partial arc divides
    # angle_deg across (count-1) steps so both ends of the arc are populated
    step = (angle_total / count) if abs(angle_total - 360.0) < 1e-9 else (
        angle_total / (count - 1) if count > 1 else 0.0
    )
    copies = []
    for i in range(count):
        loc = bd.Location(origin, direction, step * i)
        copies.append(loc * target)
    return bd.Compound(copies)


def _handle_mirror(data: dict, ctx: dict, params: dict):
    target = ctx[data["target_ref"]]
    plane_spec = data["plane"]
    if isinstance(plane_spec, str):
        if plane_spec not in _MIRROR_PLANES:
            raise NotImplementedError(f"mirror plane {plane_spec!r} (pilot scope: XY|XZ|YZ)")
        plane = _MIRROR_PLANES[plane_spec]
    elif isinstance(plane_spec, dict):
        plane = _sketch_plane(plane_spec)
    else:
        raise ValueError("mirror plane must be str 'XY'|'XZ'|'YZ' or dict{origin,x_dir,z_dir}")
    try:
        return bd.mirror(target, about=plane)
    except NotImplementedError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"mirror failed: {type(e).__name__}:{e}") from e


def _draft_auto_faces(target, pull_unit: "bd.Vector") -> list:
    """selector omitted -> every PLANE face whose normal is (near-)perpendicular to pull_direction,
    i.e. a face "parallel to the pull direction" per 's spec -- 1deg tolerance, same convention
    geometri_selektor_v1's own direction_tol_deg default uses.
    """
    out = []
    for f in target.faces():
        if f.geom_type.name != "PLANE":
            continue
        n = f.normal_at(f.center()).normalized()
        ang = n.get_angle(pull_unit)
        ang = min(ang, 180.0 - ang)  # a plane's normal has no canonical sign
        if abs(90.0 - ang) <= 1.0:
            out.append(f)
    return out


def _handle_draft(data: dict, ctx: dict, params: dict):
    target = ctx[data["target_ref"]]
    angle_deg = data["angle_deg"]
    if angle_deg == 0:
        raise ValueError("draft requires a nonzero angle_deg")
    if not (-89.0 < angle_deg < 89.0):
        raise ValueError(f"draft angle_deg must be in (-89,89), got {angle_deg}")

    pull_vec = bd.Vector(*data["pull_direction"])
    if pull_vec.length < 1e-9:
        raise ValueError("draft requires a nonzero pull_direction vector")
    pull_unit = pull_vec.normalized()

    np_spec = data["neutral_plane"]
    if "normal" not in np_spec:
        raise ValueError("draft neutral_plane requires a 'normal' key")
    n_origin = tuple(np_spec.get("origin", (0, 0, 0)))
    n_normal = bd.Vector(*np_spec["normal"])
    if n_normal.length < 1e-9:
        raise ValueError("draft neutral_plane normal must be a nonzero vector")
    n_normal = n_normal.normalized()

    selector = data.get("selector")
    if selector is not None:
        faces = list(selector_match(target, selector))
    else:
        faces = _draft_auto_faces(target, pull_unit)
    if not faces:
        raise ValueError("draft: selector/auto-detect matched 0 faces (nothing to draft)")

    drafter = BRepOffsetAPI_DraftAngle(target.wrapped)
    neutral_pln = gp_Pln(gp_Ax3(gp_Pnt(*n_origin), gp_Dir(n_normal.X, n_normal.Y, n_normal.Z)))
    pull_gp = gp_Dir(pull_unit.X, pull_unit.Y, pull_unit.Z)
    angle_rad = math.radians(angle_deg)
    for f in faces:
        drafter.Add(f.wrapped, pull_gp, angle_rad, neutral_pln)
        if not drafter.AddDone():
            raise ValueError(
                f"draft failed (angle_deg={angle_deg}): OCC rejected adding the draft to a selected "
                f"face -- likely a non-planar/non-cylindrical/non-conical face, or a face not "
                f"belonging to target_ref's own solid"
            )
    drafter.Build()
    if not drafter.IsDone():
        # MEASURED: a too-large angle_deg makes AddDone()==True per-face (OCC accepts the LOCAL
        # edit) but Build() leaves IsDone()==False once the drafted faces would self-intersect
        # before reaching the far end -- calling.Shape() past this point raises a raw, harder-to-
        # read OCC Standard_ConstructionError; normalized to an honest ValueError here instead
        # (matches _handle_fillet/_handle_shell's own OCC-exception-normalization pattern).
        raise ValueError(
            f"draft failed (angle_deg={angle_deg}): drafted face(s) self-intersect before reaching "
            f"the far end of target_ref -- reduce angle_deg or the affected extent"
        )
    result_shape = drafter.Shape()
    # MEASURED: BRepOffsetAPI_DraftAngle.Shape() returns a TopoDS_COMPOUND wrapping a single
    # TopoDS_SOLID on a FRESH solid's first draft call, but a chained draft (draft-of-a-draft, e.g.
    # applying a 2nd/3rd/4th draft to an already-drafted result) can return the SOLID directly
    # --downcasting unconditionally to Compound raised Standard_TypeMismatch on that path. Branch on
    # the actual returned ShapeType rather than assuming one shape kind.
    from OCP.TopAbs import TopAbs_ShapeEnum
    stype = result_shape.ShapeType()
    if stype == TopAbs_ShapeEnum.TopAbs_COMPOUND:
        solids = bd.Compound(TopoDS.Compound_s(result_shape)).solids()
    elif stype == TopAbs_ShapeEnum.TopAbs_SOLID:
        solids = [bd.Solid(TopoDS.Solid_s(result_shape))]
    else:
        solids = list(bd.Shape(result_shape).solids())
    if not solids:
        raise ValueError("draft produced zero solids")
    return solids[0]


# --------------------------------------------------------------------------------------------------
# -sheet_bend
# -------------------------------------------------------------------------------------------------
def _handle_sheet_bend(data: dict, ctx: dict, params: dict):
    """Isometric plate bend. PILOT SCOPE: thickness axis MUST be world Z, bend_line MUST be parallel to
    world Y -- the underlying rigid-rotation SIGN convention below was MEASURED to give an EXACT
    volume-conserving join (rel_dev=4.68e-14 at k_factor=0.5) only for this one axis triple
    (t_axis=Z, bd_axis=Y, prog_axis=X); other permutations are NOT verified and raise
    NotImplementedError rather than risk an unmeasured sign flip.

    CONSTRUCTION (3 pieces, fused): fixed part (prog < zone_min, unchanged) + bend-zone wall +
    rotating leg (rigidly translated by -BA then rotated angle_deg about the SAME axis O, so its
    near cross-section exactly coincides with the bend-wall's far cross-section -- MEASURED
    continuity: `fixed + wall + leg` fuses to n_solids==1).

    REFRAME DECLARED: the bend-wall's PHYSICAL swept volume is fixed by GEOMETRY alone (Pappus:
    V_wall = angle_rad * (radius + thickness/2) * A_cross --the TRUE geometric centroid, i.e. an
    implicit k=0.5), independent of the caller's k_factor. The caller's k_factor only sets WHERE the
    flat sheet is cut (BA = angle_rad*(radius+k_factor* thickness), the flat-pattern length). These
    two uses of "k" coincide -- giving the doc's claimed EXACT V(bent)==V(flat) invariant,
    rel_dev<1e-6 -- ONLY at k_factor=0.5. MEASURED: k=0.5 -> rel_dev=4.68e-14; k=0.44 (DIN 6935
    default, real material accounts for stretch our rigid/incompressible CAD model does NOT
    simulate)
    -> rel_dev=1.885e-3; k=0.33 -> 5.34e-3; k=0.0 -> 1.57e-2. The default stays k_factor=0.44
    (declared KONSTRUERAT, source: DIN 6935 / SME Sheet Metal Handbook -- the real-world flat-
    pattern-length standard) because that is what a real bend needs for correct FINAL leg lengths;
    the ~0.19% volume gap at the default is the honestly-measured price of NOT modelling material
    stretch, not a bug -- see platbock_ganga_v1.json REFRAME_LOG for the full 4-point table.
    """
    target = ctx[data["target_ref"]]
    angle_deg = float(data["angle_deg"])
    radius = float(data["radius"])
    thickness = float(data["thickness"])
    k_factor = float(data.get("k_factor", 0.44))
    side = data.get("side", "positive")

    if radius <= 0:
        raise ValueError(f"sheet_bend requires radius>0, got {radius}")
    if radius < thickness:
        raise ValueError(
            f"sheet_bend: bend radius ({radius}mm) < sheet thickness ({thickness}mm) -- below the "
            f"practical minimum inside bend radius (rule of thumb, min radius ~= thickness for most "
            f"sheet materials); refusing rather than building an unmanufacturable/self-crushing bend"
        )
    if thickness <= 0:
        raise ValueError(f"sheet_bend requires thickness>0, got {thickness}")
    if not (0.0 < abs(angle_deg) <= 180.0):
        raise ValueError(f"sheet_bend angle_deg must satisfy 0<|angle_deg|<=180, got {angle_deg}")
    if side != "positive":
        raise NotImplementedError(
            f"sheet_bend side={side!r} (pilot scope: 'positive' only -- the higher-X-coordinate side "
            f"of bend_line rotates, the lower-X side stays fixed)"
        )

    p0 = bd.Vector(*data["bend_line"][0])
    p1 = bd.Vector(*data["bend_line"][1])
    bb = target.bounding_box()
    extents = (bb.max.X - bb.min.X, bb.max.Y - bb.min.Y, bb.max.Z - bb.min.Z)
    t_axis = min(range(3), key=lambda i: extents[i])
    if t_axis != 2:
        raise NotImplementedError(
            f"sheet_bend: target_ref's thinnest bbox extent is along axis index {t_axis} "
            f"(0=X,1=Y,2=Z), not Z -- pilot scope requires the sheet's thickness axis to be world Z "
            f"(extents X={extents[0]:.4f} Y={extents[1]:.4f} Z={extents[2]:.4f})"
        )
    t_extent = extents[2]
    if abs(t_extent - thickness) > max(1e-3, 1e-3 * thickness):
        raise ValueError(
            f"sheet_bend: target_ref's Z-extent ({t_extent:.6f}mm) does not match declared thickness "
            f"({thickness}mm) -- target_ref may not be a flat sheet, or 'thickness' is wrong"
        )
    diff = p1 - p0
    if abs(diff.X) > 1e-6 or abs(diff.Z) > 1e-6 or abs(diff.Y) < 1e-9:
        raise NotImplementedError(
            "sheet_bend: bend_line must be parallel to world Y and lie in the sheet's XY-plane "
            f"(pilot scope) -- got p0={tuple(p0)}, p1={tuple(p1)}"
        )

    x0 = p0.X
    t_lo = bb.min.Z
    prog_lo, prog_hi = bb.min.X, bb.max.X
    bd_lo, bd_hi = bb.min.Y, bb.max.Y
    width = bd_hi - bd_lo
    bd_center = (bd_lo + bd_hi) / 2.0

    angle_rad = math.radians(angle_deg)
    Rn = radius + k_factor * thickness
    BA = abs(angle_rad) * Rn
    zone_min = x0 - BA / 2.0
    zone_max = x0 + BA / 2.0

    avail_neg = zone_min - prog_lo
    avail_pos = prog_hi - zone_max
    if avail_neg < -1e-9 or avail_pos < -1e-9:
        raise ValueError(
            f"sheet_bend: bend allowance zone (BA={BA:.4f}mm, half-width={BA / 2:.4f}mm around "
            f"bend_line at X={x0}) exceeds available flat material on one side (available: "
            f"{avail_neg:.4f}mm fixed-side, {avail_pos:.4f}mm rotating-side) -- radius/thickness/"
            f"k_factor too large for this sheet, or bend_line too close to an edge"
        )

    margin = (prog_hi - prog_lo) + width + t_extent + 1000.0

    def _box_x_range(x_lo: float, x_hi: float) -> bd.Solid:
        box = bd.Solid.make_box(x_hi - x_lo, width + 2 * margin, t_extent + 2 * margin)
        return box.translate((x_lo, bd_lo - margin, t_lo - margin))

    fixed_part = target & _box_x_range(prog_lo - margin, zone_min)
    leg_raw = target & _box_x_range(zone_max, prog_hi + margin)
    leg_shifted = leg_raw.translate((-BA, 0.0, 0.0))

    O = (zone_min, bd_center, t_lo - radius)
    axis = bd.Axis(O, (0.0, 1.0, 0.0))
    leg_bent = leg_shifted.rotate(axis, angle_deg)

    path_plane = bd.Plane(origin=O, x_dir=(0, 0, 1), z_dir=(0, 1, 0))
    with bd.BuildLine(path_plane) as bl:
        bd.CenterArc((0, 0), radius, 0, angle_deg)
    path = bl.line
    p_start = path.edges()[0].position_at(0)
    tan_start = path.edges()[0].tangent_at(0)
    sec_plane = bd.Plane(origin=p_start, x_dir=(0, 0, 1), z_dir=tan_start)
    with bd.BuildSketch(sec_plane) as sk:
        with bd.Locations((thickness / 2.0, 0.0)):
            bd.Rectangle(thickness, width)
    try:
        wall = bd.sweep(sk.sketch, path=path)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"sheet_bend: bend-wall sweep failed: {type(e).__name__}:{e}") from e

    result = fixed_part + wall + leg_bent
    solids = list(result.solids())
    if len(solids) != 1:
        raise ValueError(
            f"sheet_bend: the fixed/wall/leg pieces did not fuse into one solid (got {len(solids)} "
            f"solids) -- a geometric discontinuity, refusing rather than returning a split part"
        )
    return result


# --------------------------------------------------------------------------------------------------
# -thread_cosmetic
# -------------------------------------------------------------------------------------------------
def _handle_thread_cosmetic(data: dict, ctx: dict, params: dict):
    """Metadata-only ISO thread annotation -- geometry is NEVER touched (strongest possible no-op
    invariant: volume/facecount bit-identical before/after, rel_dev==0.0 exactly). selector must
    resolve to exactly one CYLINDER face; designation validated against iso_thread_table_v1's ISO
    724 coarse-series table. Metadata is attached as a python attribute on the (unmodified, same-
    object) returned shape -- `result.thread_cosmetic_meta` -- the discoverable surface a downstream
    consumer (mate_check_v1-style code) reads nominal diameter/pitch from, per this cell's mandate:
    mate_check_ v1.json's own published finding is that its pin/socket mating chain needs nominal
    d/pitch METADATA, never a spiral BRep surface.
    """
    target = ctx[data["target_ref"]]
    designation = data["designation"]
    diameter_mm, pitch_mm = _iso_parse_designation(designation)

    faces = list(selector_match(target, data["selector"]))
    cyl_faces = [f for f in faces if f.geom_type.name == "CYLINDER"]
    if len(cyl_faces) != 1:
        raise ValueError(
            f"thread_cosmetic: selector matched {len(faces)} face(s), {len(cyl_faces)} of them "
            f"CYLINDER -- requires EXACTLY 1 cylindrical face (a cosmetic thread on a non-cylindrical "
            f"or ambiguous face selection is a configuration error, not a silent no-op)"
        )
    face = cyl_faces[0]
    measured_radius = face.radius
    measured_diameter = 2.0 * measured_radius
    if abs(measured_diameter - diameter_mm) > max(0.5, 0.1 * diameter_mm):
        raise ValueError(
            f"thread_cosmetic: designation {designation!r} declares diameter {diameter_mm}mm but the "
            f"selected cylindrical face measures {measured_diameter:.4f}mm diameter -- refusing to "
            f"label a mismatched surface (this is exactly the mislabel-class bug population/selector "
            f"guards in this module exist to prevent)"
        )
    target.thread_cosmetic_meta = {
        "designation": designation, "diameter_mm": diameter_mm, "pitch_mm": pitch_mm,
        "measured_face_diameter_mm": measured_diameter,
    }
    return target


# --------------------------------------------------------------------------------------------------
# -sketch_profile: the sketch becomes a first-class recipe citizen -- name-keyed
# sketch_gcs_v1.SketchGCS graph in, solved profile out, SAME ctx[op_id] reference semantics as
# sketch_2d (see cad_op_schema_v1.py "SKETCH PROFILE" docstring). Dimension values ({"expr":
# "BORE_D/2"} anywhere in the op dict) are ALREADY resolved to concrete floats by exec_ops' own
# generic _resolve_op_exprs walk BEFORE this handler ever runs -- this handler therefore never calls
# resolve_expr itself; the ONE evaluator is reused, not duplicated, exactly per this cell's mandate.
# -----------------------------------------------------------------------------------------------
class SketchProfileRefusal(ValueError):
    """Refusal: an over/under-determined or geometrically-invalid sketch never
    silently proceeds to build a profile from a not-fully-solved SketchGCS. Carries `.op_id`,
    `.status` (SketchGCS's own status vocabulary: OVERBESTAMD_KONFLIKT/UNDERBESTAMD/
    GEOMETRISKT_OGILTIG) and `.diagnosis` (the solver's own structured blame dict, e.g.
    conflicting_ranked / free_directions_named) as REAL attributes -- not just baked into the
    message string -- so a caller can programmatically inspect the refusal, not just log it. The
    message itself also carries a compact JSON rendering of the same diagnosis (exec_ops' generic
    except-Exception log row only stores str(e)), so the structured content survives even when a
    caller only reads the log row's "reason" string."""

    def __init__(self, op_id: str, status: str, diagnosis: dict):
        self.op_id = op_id
        self.status = status
        self.diagnosis = diagnosis
        super().__init__(
            f"sketch_profile {op_id!r}: solve did not reach FULLT_BESTAMD (status={status}) -- "
            f"diagnosis={json.dumps(diagnosis, default=str)}"
        )


_SKETCH_GCS_CONSTRAINT_BUILDERS = {
    "coincident": lambda sk, c, P, L, C, R: sk.coincident(P[c["p1"]], P[c["p2"]], name=c.get("name")),
    "horizontal": lambda sk, c, P, L, C, R: sk.horizontal(L[c["line"]], name=c.get("name")),
    "vertical": lambda sk, c, P, L, C, R: sk.vertical(L[c["line"]], name=c.get("name")),
    "parallel": lambda sk, c, P, L, C, R: sk.parallel(L[c["l1"]], L[c["l2"]], name=c.get("name")),
    "perpendicular": lambda sk, c, P, L, C, R: sk.perpendicular(L[c["l1"]], L[c["l2"]], name=c.get("name")),
    "angle_l2l": lambda sk, c, P, L, C, R: sk.angle_l2l(
        L[c["l1"]], L[c["l2"]], math.radians(c["angle_deg"]), name=c.get("name")),
    "tangent_line_circle": lambda sk, c, P, L, C, R: sk.tangent_line_circle(
        L[c["line"]], C[c["circle"]], name=c.get("name")),
    "tangent_circle_circle": lambda sk, c, P, L, C, R: sk.tangent_circle_circle(
        C[c["c1"]], C[c["c2"]], external=c.get("external", True), name=c.get("name")),
    "distance_p2p": lambda sk, c, P, L, C, R: sk.distance_p2p(
        P[c["p1"]], P[c["p2"]], c["value"], name=c.get("name")),
    "distance_p2l": lambda sk, c, P, L, C, R: sk.distance_p2l(
        P[c["p"]], L[c["line"]], c["value"], name=c.get("name")),
    "radius": lambda sk, c, P, L, C, R: sk.radius(R[c["param"]], c["value"], name=c.get("name")),
    "direction": lambda sk, c, P, L, C, R: sk.direction(
        P[c["p_from"]], P[c["p_to"]], tuple(c["value"]), name=c.get("name")),
    # ---- sketchmotor_sond_v1: the four sketcher constraints v1 lacked --
    "point_on_circle": lambda sk, c, P, L, C, R: sk.point_on_circle(
        P[c["p"]], C[c["circle"]], name=c.get("name")),
    "point_on_line": lambda sk, c, P, L, C, R: sk.point_on_line(
        P[c["p"]], L[c["line"]], name=c.get("name")),
    "concentric": lambda sk, c, P, L, C, R: sk.concentric(C[c["c1"]], C[c["c2"]], name=c.get("name")),
    "equal_radius": lambda sk, c, P, L, C, R: sk.equal_radius(
        R[c["p1"]], R[c["p2"]], name=c.get("name")),
    "symmetric_p2p_line": lambda sk, c, P, L, C, R: sk.symmetric_p2p_line(
        P[c["p1"]], P[c["p2"]], L[c["line"]], name=c.get("name")),
}


def _build_sketch_gcs(data: dict):
    """Name-keyed op dict -> (sketch_gcs_v1.SketchGCS instance, point_ids, positive_param_ids). Pure
    translation layer: every sk.add_*/constraint call below is the SAME sketch_gcs_v1 API the
    already measured (0.2-1.8ms, facits 1-3) -- no re-implementation of the solver or its residuals
    here.
    """
    from sketch_gcs_v1 import SketchGCS

    sk = SketchGCS()
    P: dict[str, int] = {}
    R: dict[str, int] = {}
    L: dict[str, int] = {}
    C: dict[str, int] = {}

    for name, spec in data["points"].items():
        P[name] = sk.add_point(spec["x"], spec["y"], fixed=bool(spec.get("fixed", False)))
    for name, spec in data.get("sketch_params", {}).items():
        R[name] = sk.add_param(spec["value"], fixed=bool(spec.get("fixed", False)))
    for name, (p1, p2) in data.get("lines", {}).items():
        L[name] = sk.add_line(P[p1], P[p2])
    for name, spec in data.get("circles", {}).items():
        C[name] = sk.add_circle(P[spec["center"]], R[spec["radius_param"]])
    A: dict[str, int] = {}
    for name, spec in data.get("arcs", {}).items():  # sketchmotor_sond_v1 (2026-08-17)
        A[name] = sk.add_arc(C[spec["circle"]], P[spec["p_start"]], P[spec["p_end"]],
                             ccw=bool(spec.get("ccw", True)))
    data["_arc_ids"] = A

    for c in data["constraints"]:
        ctype = c["type"]
        builder = _SKETCH_GCS_CONSTRAINT_BUILDERS.get(ctype)
        if builder is None:
            raise ValueError(
                f"sketch_profile: unknown constraint type {ctype!r} -- not one of "
                f"{sorted(_SKETCH_GCS_CONSTRAINT_BUILDERS)}"
            )
        builder(sk, c, P, L, C, R)

    positive_names = data.get("positive_params")
    if positive_names is None:
        # default (schema docstring): every sketch_param used as a circle's radius_param
        positive_names = sorted({spec["radius_param"] for spec in data.get("circles", {}).values()})
    positive_param_ids = tuple(R[name] for name in positive_names)

    return sk, P, positive_param_ids


def _close_wire_or_refuse(op_id: str, edges: list, what: str):
    """THE CLOSURE GATE, shared by arc_chain and mixed_chain (blandprofil_v1 reuses the gate
    sketchmotor_sond_v1 already wrote rather than authoring a second, drifting copy). A solved
    sketch whose segments do not MEET is a SILENT-WRONG-GEOMETRY source, never a pass."""
    wires = bd.Wire.combine(edges)
    if len(wires) != 1 or not wires[0].is_closed:
        raise ValueError(
            f"sketch_profile {op_id!r}: {what} did not close into ONE wire "
            f"(got {len(wires)} wire(s), closed={[w.is_closed for w in wires]}) -- a solved "
            f"sketch whose segments do not meet is a SILENT-WRONG-GEOMETRY source, never a pass")
    return wires[0]


def _check_continuity(op_id: str, edges: list, spec: list) -> list:
    """TANGENCY / CONTINUITY GATE (blandprofil_v1). Two-sided angle band, MEASURED on the built
    edges (not on the solver's claim): |angle between outgoing tangent of segment i and incoming
    tangent of segment i+1| must be <= tol_deg. A declared G1 joint that is not G1 is a refusal."""
    import math as _math
    out = []
    n = len(edges)
    for c in spec:
        i = int(c["at"])
        tol = float(c.get("tol_deg", 1.0))
        if c.get("type", "G1") != "G1":
            raise NotImplementedError(
                f"sketch_profile {op_id!r}: continuity type {c.get('type')!r} (supported: 'G1')")
        if not (0 <= i < n):
            raise ValueError(f"sketch_profile {op_id!r}: continuity.at={i} outside 0..{n-1}")
        a = edges[i]
        b = edges[(i + 1) % n]
        ta = a.tangent_at(1.0)
        # the joint point is a's end; b may be traversed either way -> take the b-end nearest it
        pa = a.position_at(1.0)
        tb = b.tangent_at(0.0) if (b.position_at(0.0) - pa).length <= (b.position_at(1.0) - pa).length \
            else -b.tangent_at(1.0)
        dot = max(-1.0, min(1.0, ta.X * tb.X + ta.Y * tb.Y + ta.Z * tb.Z))
        ang = _math.degrees(_math.acos(dot))
        rec = {"at": i, "angle_deg": round(ang, 9), "tol_deg": tol, "pass": bool(ang <= tol)}
        out.append(rec)
        if not rec["pass"]:
            raise ValueError(
                f"sketch_profile {op_id!r}: declared G1 continuity at segment joint {i} MEASURED "
                f"{ang:.6f} deg > tol {tol} deg -- a declared tangency that is not tangent is a "
                f"refusal, never a silent kink")
    return out


def _mixed_chain_edges(op_id: str, data: dict, sk, point_ids: dict, profile: dict):
    """profile.segment_order -> real build123d Edges (LINE / CIRCLE / BSPLINE), EXACT.

    The lossy alternative this replaces: profile type "polygon" chords every curved edge into
    straight segments. Here an arc becomes a real circular Edge and a spline becomes a real
    interpolated BSPLINE Edge, so the extruded BREP carries CYLINDER / BSPLINE surfaces, not a
    plane fan."""
    import math as _math
    order = profile.get("segment_order")
    if not order or len(order) < 2:
        raise ValueError(
            f"sketch_profile {op_id!r}: mixed_chain needs >=2 segments, got "
            f"{0 if not order else len(order)}")
    lines = data.get("lines", {})
    arcs = data.get("arcs", {})
    splines = data.get("splines", {})
    arc_ids = data.get("_arc_ids") or {}

    def P(name):
        p = sk.points[point_ids[name]]
        return (p["x"], p["y"])

    edges, meta = [], []
    for k, seg in enumerate(order):
        kind = seg.get("kind")
        if kind == "line":
            nm = seg["line"]
            if nm not in lines:
                raise ValueError(f"sketch_profile {op_id!r}: segment {k} names undeclared line {nm!r}")
            a, b = lines[nm]
            pa, pb = P(a), P(b)
            if _math.dist(pa, pb) < 1e-12:
                raise ValueError(
                    f"sketch_profile {op_id!r}: segment {k} line {nm!r} solved DEGENERATE "
                    f"(length {_math.dist(pa, pb):.3e}) -- a zero-length edge is a refusal")
            edges.append(bd.Edge.make_line(bd.Vector(*pa, 0), bd.Vector(*pb, 0)))
            meta.append({"i": k, "kind": "line", "name": nm, "length": _math.dist(pa, pb)})
        elif kind == "arc":
            nm = seg["arc"]
            if nm not in arcs:
                raise ValueError(f"sketch_profile {op_id!r}: segment {k} names undeclared arc {nm!r}")
            s = sk.to_arc_profile_schema([arc_ids[nm]])["segments"][0]
            cx, cy = s["center"]; r = s["r"]
            a0 = _math.degrees(_math.atan2(s["p0"][1] - cy, s["p0"][0] - cx)) % 360.0
            a1 = _math.degrees(_math.atan2(s["p1"][1] - cy, s["p1"][0] - cx)) % 360.0
            sweep = ((a1 - a0) % 360.0) if s["ccw"] else -((a0 - a1) % 360.0)
            if abs(sweep) < 1e-9:
                raise ValueError(
                    f"sketch_profile {op_id!r}: segment {k} arc {nm!r} solved to a ZERO sweep -- "
                    f"a degenerate arc is a refusal, never a silent point")
            am = _math.radians(a0 + sweep / 2.0)
            mid = (cx + r * _math.cos(am), cy + r * _math.sin(am))
            edges.append(bd.Edge.make_three_point_arc(
                bd.Vector(*s["p0"], 0), bd.Vector(*mid, 0), bd.Vector(*s["p1"], 0)))
            meta.append({"i": k, "kind": "arc", "name": nm, "r": r, "sweep_deg": sweep})
        elif kind == "spline":
            nm = seg["spline"]
            if nm not in splines:
                raise ValueError(
                    f"sketch_profile {op_id!r}: segment {k} names undeclared spline {nm!r}")
            spec = splines[nm]
            if "poles" in spec:
                # EXACT NURBS carry-over (blandprofil_v1): poles/knots/mults/degree taken
                # verbatim from the source B-rep curve. This is the ONLY lossless route for a
                # general freeform boundary -- interpolating through sampled points is an
                # APPROXIMATION and is measured as such (see brep_kontur_extrakt_v1).
                from OCP.Geom import Geom_BSplineCurve
                from OCP.TColgp import TColgp_Array1OfPnt
                from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
                from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
                from OCP.gp import gp_Pnt
                poles = spec["poles"]
                pa = TColgp_Array1OfPnt(1, len(poles))
                for i, (px, py) in enumerate(poles, 1):
                    pa.SetValue(i, gp_Pnt(float(px), float(py), 0.0))
                ka = TColStd_Array1OfReal(1, len(spec["knots"]))
                for i, kv in enumerate(spec["knots"], 1):
                    ka.SetValue(i, float(kv))
                ma = TColStd_Array1OfInteger(1, len(spec["mults"]))
                for i, mv in enumerate(spec["mults"], 1):
                    ma.SetValue(i, int(mv))
                wts = spec.get("weights")
                if wts:
                    wa = TColStd_Array1OfReal(1, len(wts))
                    for i, wv in enumerate(wts, 1):
                        wa.SetValue(i, float(wv))
                    crv = Geom_BSplineCurve(pa, wa, ka, ma, int(spec["degree"]),
                                            bool(spec.get("periodic", False)))
                else:
                    crv = Geom_BSplineCurve(pa, ka, ma, int(spec["degree"]),
                                            bool(spec.get("periodic", False)))
                e = bd.Edge(BRepBuilderAPI_MakeEdge(crv).Edge())
                if "trim" in spec:
                    t0, t1 = spec["trim"]
                    e = bd.Edge(BRepBuilderAPI_MakeEdge(crv, float(t0), float(t1)).Edge())
                edges.append(e)
                meta.append({"i": k, "kind": "spline", "name": nm, "exact_nurbs": True,
                             "degree": int(spec["degree"]), "n_poles": len(poles)})
                continue
            pts = [P(n) for n in spec["point_order"]]
            if len(pts) < 2:
                raise ValueError(
                    f"sketch_profile {op_id!r}: spline {nm!r} needs >=2 interpolation points")
            tan = spec.get("tangents")
            kw = {}
            if tan:
                kw["tangents"] = [bd.Vector(t[0], t[1], 0) for t in tan]
                kw["scale"] = bool(spec.get("scale_tangents", False))
            edges.append(bd.Edge.make_spline([bd.Vector(*p, 0) for p in pts],
                                             periodic=bool(spec.get("periodic", False)), **kw))
            meta.append({"i": k, "kind": "spline", "name": nm, "n_pts": len(pts)})
        else:
            raise ValueError(
                f"sketch_profile {op_id!r}: segment {k} kind {kind!r} (supported: 'line', 'arc', "
                f"'spline')")
    return edges, meta


def _handle_sketch_profile(data: dict, ctx: dict, params: dict):
    op_id = data["id"]
    sk, point_ids, positive_param_ids = _build_sketch_gcs(data)
    res = sk.solve(positive_params=positive_param_ids)

    if res.status not in ("FULLT_BESTAMD", "FULLT_BESTAMD_REDUNDANT"):
        raise SketchProfileRefusal(op_id, res.status, res.diagnosis)

    profile = data["profile"]
    ptype = profile.get("type")
    if ptype not in ("polygon", "arc_chain", "mixed_chain"):
        raise NotImplementedError(
            f"sketch_profile profile type {ptype!r} (supported: 'polygon', 'arc_chain', "
            f"'mixed_chain')"
        )
    plane = _sketch_plane(data.get("plane"))

    if ptype == "mixed_chain":
        # blandprofil_v1: ARBITRARILY INTERLEAVED line/arc/spline segments.
        edges, seg_meta = _mixed_chain_edges(op_id, data, sk, point_ids, profile)
        wire = _close_wire_or_refuse(op_id, edges, "mixed_chain")
        cont = _check_continuity(op_id, edges, profile.get("continuity") or [])
        result = bd.Sketch() + plane.from_local_coords(bd.Face(wire))
        result._op_mixed_chain_meta = {"segments": seg_meta, "continuity": cont}
    elif ptype == "arc_chain":
        # sketchmotor_sond_v1: tangent-arc boundary -> REAL circular edges. arc_order names arcs
        # declared in the op's "arcs" block; each solved arc becomes one
        # bd.Edge.make_three_point_arc so the extruded face is an exact CYLINDER in the BREP (a
        # polygon profile would have made it a chorded plane fan -- the measured reason boolean-of-
        # primitives could not express part 231's concave R70 blend chain).
        import math as _math
        arc_ids = data.get("_arc_ids") or {}
        segs = sk.to_arc_profile_schema([arc_ids[n] for n in profile["arc_order"]])["segments"]
        if len(segs) < 2:
            raise ValueError(f"sketch_profile {op_id!r}: arc_order needs >=2 arcs, got {len(segs)}")
        edges = []
        for s in segs:
            cx, cy = s["center"]; r = s["r"]
            a0 = _math.degrees(_math.atan2(s["p0"][1] - cy, s["p0"][0] - cx)) % 360.0
            a1 = _math.degrees(_math.atan2(s["p1"][1] - cy, s["p1"][0] - cx)) % 360.0
            sweep = ((a1 - a0) % 360.0) if s["ccw"] else -((a0 - a1) % 360.0)
            am = _math.radians(a0 + sweep / 2.0)
            mid = (cx + r * _math.cos(am), cy + r * _math.sin(am))
            edges.append(bd.Edge.make_three_point_arc(
                bd.Vector(*s["p0"], 0), bd.Vector(*mid, 0), bd.Vector(*s["p1"], 0)))
        wire = _close_wire_or_refuse(op_id, edges, "arc_chain")
        result = bd.Sketch() + plane.from_local_coords(bd.Face(wire))
    else:
        pts = [
            (sk.points[point_ids[name]]["x"], sk.points[point_ids[name]]["y"])
            for name in profile["point_order"]
        ]
        if len(pts) < 3:
            raise ValueError(f"sketch_profile {op_id!r}: profile.point_order needs >=3 points, got {len(pts)}")
        with bd.BuildSketch(plane) as bsk:
            bd.Polygon(*pts, align=None)
        result = bsk.sketch
    result._op_sketch_profile_meta = {
        "status": res.status, "dof": res.dof, "n_unknowns": res.n_unknowns, "n_eqs": res.n_eqs,
        "iterations": res.iterations, "wall_ms": res.wall_s * 1000.0,
    }
    return result


# --------------------------------------------------------------------------------------------------
# -rib: DISPATCH-wiring of formrib_v1.rib_v1, so a rib step is addressable by id/target_ref inside
# an ops chain instead of only as a direct Python call.
# -------------------------------------------------------------------------------------------------
def _handle_rib(data: dict, ctx: dict, params: dict):
    """Lazy import (NOT at module top): formrib_v1.py itself does `from cad_op_exec_v1 import
    exec_ops`, so importing formrib_v1 at cad_op_exec_v1's OWN module-load time -- before `exec_ops`
    is defined further down this same file -- would be a genuine circular import (ImportError:
    cannot import name 'exec_ops', partially-initialized module). Deferring the import to call time
    (after both modules have fully loaded at least once) breaks the cycle; formrib_v1 remains the
    single owner of the rib geometry/boolean-cut logic -- this handler only resolves target_ref +
    unpacks the op dict's keys onto rib_v1's existing positional API.
    """
    from formrib_v1 import rib_v1 as _rib_v1

    target = ctx.get(data["target_ref"])
    if target is None or not hasattr(target, "volume"):
        # SAME planted-fault class as rib_v1's own missing-target guard, reached
        # here via the exec/DISPATCH path instead of a direct rib_v1() call -- a dangling/absent
        # target_ref (never populated in ctx, or a None solid) is a configuration error, not a
        # silent no-op.
        raise ValueError(
            f"rib: target_ref {data.get('target_ref')!r} resolved to no solid (rib without a "
            f"target face is a configuration error, not a silent no-op)"
        )
    result, meta = _rib_v1(data["id"], target, data["midplane"], data["thickness"],
                            data["run_length"], data["depth"])
    result._op_rib_meta = meta
    return result


DISPATCH = {
    "sketch_2d": _handle_sketch_2d,
    "extrude": _handle_extrude,
    "loft": _handle_loft,
    "boolean_union": _handle_boolean_union,
    "boolean_cut": _handle_boolean_cut,
    "boolean_intersect": _handle_boolean_intersect,
    "hole": _handle_hole,
    "fillet": _handle_fillet,
    "chamfer": _handle_chamfer,
    "revolve": _handle_revolve,
    "sweep": _handle_sweep,
    "shell": _handle_shell,
    "pattern_linear": _handle_pattern_linear,
    "pattern_circular": _handle_pattern_circular,
    "mirror": _handle_mirror,
    "draft": _handle_draft,
    "sheet_bend": _handle_sheet_bend,
    "thread_cosmetic": _handle_thread_cosmetic,
    "rib": _handle_rib,
    "sketch_profile": _handle_sketch_profile,
}


def _base_volume_for(op_type: str, data: dict, ctx: dict):
    ref_key = {"boolean_union": "base_ref", "boolean_cut": "base_ref",
               "boolean_intersect": "base_ref", "hole": "target_ref",
               "fillet": "target_ref", "chamfer": "target_ref", "shell": "target_ref",
               "pattern_linear": "target_ref", "pattern_circular": "target_ref",
               "mirror": "target_ref", "draft": "target_ref", "sheet_bend": "target_ref",
               "thread_cosmetic": "target_ref", "rib": "target_ref"}.get(op_type)
    if ref_key is None:
        return None
    base = ctx.get(data.get(ref_key))
    return getattr(base, "volume", None) if base is not None else None


def _write_log(log_path: str, log: list[dict]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"log": log}, f, indent=2)


def exec_ops(ops: list, params: dict | None = None, log_path: str | None = None) -> dict:
    """Direct op-dict -> build123d dispatch, in memory. Accepts raw dicts OR OpStep-like objects
    (duck-typed via .op_type/.op_id/.data) -- deliberately NOT coupled to cad_op_schema_v1.validate_op,
    so a caller CAN exercise the dispatch layer's own NotImplementedError contract directly (AT2d).

    ALL-POPULATIONSVAKTEN (see module docstring): for op_type in _ALL_CHAIN_ENTITY with
    data.get("selector") is None, check_all_population runs on ctx[target_ref] BEFORE the handler --
    a declared-and-drifted population raises geometri_selektor_v1.RefusalError(reason=POPULATION_DRIFT)
    FROM THIS FUNCTION, with op_id + this op's chain_position (its index in `ops`) folded into
    e.diagnosis, propagated to the caller (never swallowed, never downgraded to a generic FAIL row --
    matches the un-guarded NotImplementedError branch's own re-raise-after-log pattern above)."""
    params = params or {}
    ctx: dict[str, Any] = {}
    log: list[dict] = []
    drain_population_warnings()  # clear any warnings left over from a prior, unrelated exec_ops call
    for chain_position, raw in enumerate(ops):
        if hasattr(raw, "op_type"):
            op_type, op_id, data = raw.op_type, raw.op_id, raw.data
        elif isinstance(raw, dict):
            op_type, op_id, data = raw.get("op"), raw.get("id"), raw
        else:
            raise ValueError(f"unsupported op entry type {type(raw).__name__}")

        # SYMBOLIC EXPRESSIONS -- resolved BEFORE the ALL-ALL-population guard check and BEFORE the
        # handler runs (module docstring RESOLUTION ORDER). BACKWARD COMPATIBILITY early-exit:
        # params falsy => `data` is left as the EXACT SAME dict object (no walk, no copy) -- a
        # legacy caller with no params is byte-for-byte unaffected.
        expr_provenance: list[dict] = []
        if params:
            data, expr_provenance = _resolve_op_exprs(data, params)

        handler = DISPATCH.get(op_type)
        t0 = time.perf_counter()
        if handler is None:
            log.append({"op_id": op_id, "op_type": op_type, "volymdelta_mm3": None,
                        "wall_ms": (time.perf_counter() - t0) * 1000.0, "status": "ESCALATE",
                        "reason": "NotImplementedError -- op-type has no dispatch handler in pilot scope"})
            if log_path:
                _write_log(log_path, log)
            raise NotImplementedError(op_type)

        population_guard: dict | None = None
        entity = _ALL_CHAIN_ENTITY.get(op_type)
        if entity is not None and data.get("selector") is None:
            target = ctx.get(data.get("target_ref"))
            if target is not None:
                try:
                    guard = check_all_population(
                        target, entity, data.get("expected_population"), op_id=op_id,
                        context={"op_id": op_id, "op_type": op_type, "chain_position": chain_position},
                    )
                except RefusalError as e:  # noqa: F841 -- e used below (diagnosis already carries op_id/chain_position)
                    log.append({"op_id": op_id, "op_type": op_type, "volymdelta_mm3": None,
                                "wall_ms": (time.perf_counter() - t0) * 1000.0, "status": "REFUSED",
                                "reason": f"POPULATION_DRIFT:{e}", "population_guard": e.diagnosis})
                    if log_path:
                        _write_log(log_path, log)
                    raise
                population_guard = {"declared": guard["declared"], "flag": guard["flag"],
                                     "population": guard["population"]}
                if not guard["declared"]:
                    # backward-compatible (legacy, undeclared) path: never raises, geometry is
                    # UNCHANGED for this run -- but the measured-now population is folded back into
                    # the log/receipt so the recipe CAN be upgraded to a declared expected_population
                    # next revision (record_expected_population, see module docstring).
                    auto = record_expected_population(target, entity)
                    population_guard["auto_recorded_expected_population"] = auto["expected_population"]

        try:
            vol_before = _base_volume_for(op_type, data, ctx)
            result = handler(data, ctx, params)
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001 -- log the failure row before re-raising
            row = {"op_id": op_id, "op_type": op_type, "volymdelta_mm3": None,
                   "wall_ms": (time.perf_counter() - t0) * 1000.0, "status": "FAIL",
                   "reason": f"exception:{type(e).__name__}:{e}"}
            if population_guard is not None:
                row["population_guard"] = population_guard
            log.append(row)
            if log_path:
                _write_log(log_path, log)
            raise

        wall_ms = (time.perf_counter() - t0) * 1000.0
        vol_after = getattr(result, "volume", None) if result is not None else None
        delta = (vol_after - vol_before) if (vol_after is not None and vol_before is not None) else None
        status, reason = "PASS", None
        if op_type in ("boolean_cut", "hole") and delta is not None and abs(delta) < _MISSED_CUT_TOL_MM3:
            status, reason = "FAIL", "MISSED_CUT"
        if result is not None:
            ctx[op_id] = result
        row = {"op_id": op_id, "op_type": op_type, "volymdelta_mm3": delta,
               "wall_ms": wall_ms, "status": status, "reason": reason}
        if population_guard is not None:
            row["population_guard"] = population_guard
        if expr_provenance:
            row["expr_resolutions"] = expr_provenance
        # radius_frac / rib side-channel metadata: the handler stashes it as a python attribute on
        # the returned shape (same idiom _handle_thread_cosmetic already uses for
        # thread_cosmetic_meta) -- fold it into this op's own log row, then strip the attribute so
        # it never leaks into a later op's volume/repr.
        value_resolution = getattr(result, "_op_value_resolution", None) if result is not None else None
        if value_resolution is not None:
            row["value_resolution"] = value_resolution
            try:
                del result._op_value_resolution
            except Exception:  # noqa: BLE001
                pass
        rib_meta = getattr(result, "_op_rib_meta", None) if result is not None else None
        if rib_meta is not None:
            row["rib_meta"] = rib_meta
            try:
                del result._op_rib_meta
            except Exception:  # noqa: BLE001
                pass
        sketch_profile_meta = getattr(result, "_op_sketch_profile_meta", None) if result is not None else None
        if sketch_profile_meta is not None:
            row["sketch_profile_meta"] = sketch_profile_meta
            try:
                del result._op_sketch_profile_meta
            except Exception:  # noqa: BLE001
                pass
        log.append(row)

    population_warnings = drain_population_warnings()
    if log_path:
        _write_log(log_path, log)
    return {"solids": ctx, "log": log, "population_warnings": population_warnings}


# --------------------------------------------------------------------------------------------------
# SELFTEST -- AT2a (3 synthetic recipes vs analytical reference values, rel. dev < 1e-6), AT2b (a
# constructed empty cut => MISSED_CUT FAIL row), AT2d onwards (the remaining handlers and the
# expression evaluator)
# -------------------------------------------------------------------------------------------------
def _at2a_box_hole() -> dict:
    """box (20x30x10) with ONE through-hole d=4mm along Z -- analytical facit."""
    ops = [
        {"op": "sketch_2d", "id": "sk_box", "shapes": [{"type": "rectangle", "width": 20, "height": 30, "mode": "add"}]},
        {"op": "extrude", "id": "box", "sketch_ref": "sk_box", "amount": 10},
        {"op": "hole", "id": "cut1", "target_ref": "box", "center": [0, 0, 5], "diameter": 4, "axis": "Z", "through": True},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["cut1"].volume
    expected = 20 * 30 * 10 - math.pi * (4 / 2) ** 2 * 10
    rel_dev = abs(measured - expected) / expected
    return {"measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6, "log": out["log"]}


def _at2a_loft_two_profiles() -> dict:
    """loft between 2 identical 8x6 rectangles 10mm apart along Z -- exact prism facit."""
    plane0 = {"origin": [0, 0, 0], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]}
    plane1 = {"origin": [0, 0, 10], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]}
    ops = [
        {"op": "sketch_2d", "id": "s0", "plane": plane0, "shapes": [{"type": "rectangle", "width": 8, "height": 6, "mode": "add"}]},
        {"op": "sketch_2d", "id": "s1", "plane": plane1, "shapes": [{"type": "rectangle", "width": 8, "height": 6, "mode": "add"}]},
        {"op": "loft", "id": "lofted", "sketch_refs": ["s0", "s1"], "ruled": True},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["lofted"].volume
    expected = 8 * 6 * 10
    rel_dev = abs(measured - expected) / expected
    return {"measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6, "log": out["log"]}


def _at2a_cut_series() -> dict:
    """box (40x40x20) minus 3 sequential non-overlapping through-holes -- analytical facit."""
    ops = [
        {"op": "sketch_2d", "id": "sk_box2", "shapes": [{"type": "rectangle", "width": 40, "height": 40, "mode": "add"}]},
        {"op": "extrude", "id": "box2", "sketch_ref": "sk_box2", "amount": 20},
        {"op": "hole", "id": "cut_a", "target_ref": "box2", "center": [-10, 0, 10], "diameter": 3, "axis": "Z", "through": True},
        {"op": "hole", "id": "cut_b", "target_ref": "cut_a", "center": [0, 0, 10], "diameter": 3, "axis": "Z", "through": True},
        {"op": "hole", "id": "cut_c", "target_ref": "cut_b", "center": [10, 0, 10], "diameter": 3, "axis": "Z", "through": True},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["cut_c"].volume
    expected = 40 * 40 * 20 - 3 * math.pi * (3 / 2) ** 2 * 20
    rel_dev = abs(measured - expected) / expected
    return {"measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6, "log": out["log"]}


def _at2b_tomcut_fallbevis() -> dict:
    """box (10x10x10) cut by a tool box placed ENTIRELY outside it -- MUST produce a MISSED_CUT
    FAIL row, never a silent no-op."""
    ops = [
        {"op": "sketch_2d", "id": "sk_base", "shapes": [{"type": "rectangle", "width": 10, "height": 10, "mode": "add"}]},
        {"op": "extrude", "id": "base", "sketch_ref": "sk_base", "amount": 10},
        {"op": "sketch_2d", "id": "sk_tool", "plane": {"origin": [100, 0, 0], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]},
         "shapes": [{"type": "rectangle", "width": 10, "height": 10, "mode": "add"}]},
        {"op": "extrude", "id": "tool", "sketch_ref": "sk_tool", "amount": 10},
        {"op": "boolean_cut", "id": "tomcut", "base_ref": "base", "tool_refs": ["tool"]},
    ]
    out = exec_ops(ops)
    tomcut_rows = [r for r in out["log"] if r["op_id"] == "tomcut"]
    row = tomcut_rows[0] if tomcut_rows else {}
    ok = row.get("status") == "FAIL" and row.get("reason") == "MISSED_CUT"
    return {"pass": ok, "row": row, "log": out["log"]}


def _at2d_reference_not_implemented() -> dict:
    ok = False
    msg = ""
    try:
        exec_ops([{"op": "reference", "id": "rf1", "kind": "plane"}])
    except NotImplementedError as e:
        ok = True
        msg = str(e)
    return {"pass": ok, "raised_msg": msg}


def _at2e_draft_cube() -> dict:
    """draft's own acceptance test: kub L x L x H, 4 vertikala sidoytor draftade vinkel theta, neutral
    plan vid basen (z=0), pull_direction=+Z. Analytiskt reference value = frustum-of-pyramid.
    Fallbevis: theta sa stort att L' <= 0 (ytorna korsar innan H nas) MASTE resa fel, aldrig ett
    silently sjalv-korsande solid.
    """
    L, H, theta = 20.0, 20.0, 8.0
    ops = [
        {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": L, "height": L, "mode": "add"}]},
        {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": H},
        {"op": "draft", "id": "drafted", "target_ref": "box", "angle_deg": theta,
         "pull_direction": [0, 0, 1], "neutral_plane": {"origin": [0, 0, 0], "normal": [0, 0, 1]}},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["drafted"].volume
    Lp = L - 2 * H * math.tan(math.radians(theta))
    expected = H / 3.0 * (L**2 + L * Lp + Lp**2)
    rel_dev = abs(measured - expected) / expected

    # fallbevis: theta=80deg -> L - 2*20*tan(80deg) < 0 (faces cross before reaching H) -- MUST raise
    fallbevis_raised, fallbevis_type = False, None
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk2", "shapes": [{"type": "rectangle", "width": L, "height": L, "mode": "add"}]},
            {"op": "extrude", "id": "box2", "sketch_ref": "sk2", "amount": H},
            {"op": "draft", "id": "bad", "target_ref": "box2", "angle_deg": 80.0,
             "pull_direction": [0, 0, 1], "neutral_plane": {"origin": [0, 0, 0], "normal": [0, 0, 1]}},
        ])
    except Exception as e:
        fallbevis_raised, fallbevis_type = True, type(e).__name__

    return {
        "measured": measured, "expected": expected, "rel_dev": rel_dev, "pass": rel_dev < 1e-6,
        "Lp_top_side_mm": Lp,
        "fallbevis": {"pass": fallbevis_raised, "exception_type": fallbevis_type,
                      "params": "angle_deg=80.0 (top side L'=L-2*H*tan(80deg) < 0, self-intersecting)"},
    }


def _at2f_sheet_bend_l_bracket() -> dict:
    """reference value: flat plate 100(X)x20(Y)x2(Z)mm, bend at x0=60, radius=5. PRIMARY reference
    value (matches the doc's own claimed invariant): at k_factor=0.5, V(bent) == V(flat) EXACTLY,
    rel_dev<1e-6. SECONDARY reference value: the flat-pattern reconstruction identity leg1_len + BA
    + leg2_len == L_total holds exactly BY CONSTRUCTION (zero_min/zero_max are DEFINED from BA),
    stated explicitly here as a hand-checkable cross-check, not just asserted. Fallbevis 1: radius <
    thickness -> ValueError. Fallbevis 2: bend allowance zone wider than available flat
    material -> ValueError.
    """
    L, W, t, x0, radius = 100.0, 20.0, 2.0, 60.0, 5.0
    angle_deg = 90.0

    def _ops(k_factor):
        return [
            {"op": "sketch_2d", "id": "sk", "plane": {"origin": [0, 0, 0], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]},
             "shapes": [{"type": "rectangle", "width": L, "height": W, "center": [L / 2.0, W / 2.0], "mode": "add"}]},
            {"op": "extrude", "id": "flat", "sketch_ref": "sk", "amount": t},
            {"op": "sheet_bend", "id": "bent", "target_ref": "flat",
             "bend_line": [[x0, 0.0, 0.0], [x0, W, 0.0]], "angle_deg": angle_deg, "radius": radius,
             "thickness": t, "k_factor": k_factor},
        ]

    out_k50 = exec_ops(_ops(0.5))
    # recompute flat volume independently (fresh exec, sketch+extrude only) for a clean before/after
    flat_only = exec_ops(_ops(0.5)[:2])
    v_flat = flat_only["solids"]["flat"].volume
    v_bent_k50 = out_k50["solids"]["bent"].volume
    rel_dev_k50 = abs(v_bent_k50 - v_flat) / v_flat

    angle_rad = math.radians(angle_deg)
    BA = angle_rad * (radius + 0.5 * t)
    zone_min, zone_max = x0 - BA / 2.0, x0 + BA / 2.0
    leg1_len, leg2_len = zone_min - 0.0, L - zone_max
    reconstruct_rel_dev = abs((leg1_len + BA + leg2_len) - L) / L

    k_scan = {}
    for k in (0.5, 0.44, 0.33, 0.0):
        o = exec_ops(_ops(k))
        v = o["solids"]["bent"].volume
        k_scan[str(k)] = {"volume": v, "rel_dev_vs_flat": abs(v - v_flat) / v_flat}

    fallbevis1_raised, fallbevis1_msg = False, ""
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk2", "shapes": [{"type": "rectangle", "width": L, "height": W, "center": [L / 2.0, W / 2.0], "mode": "add"}]},
            {"op": "extrude", "id": "flat2", "sketch_ref": "sk2", "amount": t},
            {"op": "sheet_bend", "id": "bad1", "target_ref": "flat2", "bend_line": [[x0, 0.0, 0.0], [x0, W, 0.0]],
             "angle_deg": angle_deg, "radius": 1.0, "thickness": t, "k_factor": 0.44},
        ])
    except ValueError as e:
        fallbevis1_raised, fallbevis1_msg = True, str(e)

    fallbevis2_raised, fallbevis2_msg = False, ""
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "sk3", "shapes": [{"type": "rectangle", "width": L, "height": W, "center": [L / 2.0, W / 2.0], "mode": "add"}]},
            {"op": "extrude", "id": "flat3", "sketch_ref": "sk3", "amount": t},
            {"op": "sheet_bend", "id": "bad2", "target_ref": "flat3", "bend_line": [[2.0, 0.0, 0.0], [2.0, W, 0.0]],
             "angle_deg": angle_deg, "radius": radius, "thickness": t, "k_factor": 0.44},
        ])
    except ValueError as e:
        fallbevis2_raised, fallbevis2_msg = True, str(e)

    return {
        "v_flat_mm3": v_flat, "v_bent_k050_mm3": v_bent_k50, "rel_dev_k050": rel_dev_k50,
        "pass_volume_facit": rel_dev_k50 < 1e-6,
        "reconstruct_rel_dev": reconstruct_rel_dev, "pass_reconstruct": reconstruct_rel_dev < 1e-6,
        "REFRAME_k_factor_scan": k_scan,
        "fallbevis_radius_lt_thickness": {"pass": fallbevis1_raised, "raised_msg": fallbevis1_msg},
        "fallbevis_zone_exceeds_material": {"pass": fallbevis2_raised, "raised_msg": fallbevis2_msg},
        "pass": bool(
            rel_dev_k50 < 1e-6 and reconstruct_rel_dev < 1e-6 and fallbevis1_raised and fallbevis2_raised
        ),
    }


def _at2g_thread_cosmetic() -> dict:
    """reference value: M8 cosmetic thread on a d=8mm cylinder's lateral face -- STRONGEST possible no-
    op invariant, volume bit-identical (rel_dev==0.0 exactly, not <1e-6). Fallbevis: designation
    with a pitch that does not match the ISO 724 coarse table (M8x2.0, real coarse pitch is 1.25) ->
    ValueError before any geometry is touched.
    """
    ops = [
        {"op": "sketch_2d", "id": "skc", "shapes": [{"type": "circle", "radius": 4.0, "mode": "add"}]},
        {"op": "extrude", "id": "rod", "sketch_ref": "skc", "amount": 20},
        {"op": "thread_cosmetic", "id": "threaded", "target_ref": "rod",
         "selector": {"entity": "face", "geometry_type": "CYLINDER", "expected_count": 1}, "designation": "M8"},
    ]
    out = exec_ops(ops)
    v_before = out["solids"]["rod"].volume  # same python object as "threaded" (metadata-only op)
    v_after = out["solids"]["threaded"].volume
    fc_before = len(out["solids"]["rod"].faces())
    fc_after = len(out["solids"]["threaded"].faces())
    meta = getattr(out["solids"]["threaded"], "thread_cosmetic_meta", None)

    fallbevis_raised, fallbevis_msg = False, ""
    try:
        exec_ops([
            {"op": "sketch_2d", "id": "skc2", "shapes": [{"type": "circle", "radius": 4.0, "mode": "add"}]},
            {"op": "extrude", "id": "rod2", "sketch_ref": "skc2", "amount": 20},
            {"op": "thread_cosmetic", "id": "bad", "target_ref": "rod2",
             "selector": {"entity": "face", "geometry_type": "CYLINDER", "expected_count": 1}, "designation": "M8x2.0"},
        ])
    except ValueError as e:
        fallbevis_raised, fallbevis_msg = True, str(e)

    return {
        "volume_rel_dev": (abs(v_after - v_before) / v_before) if v_before else None,
        "volume_bit_identical": v_after == v_before,
        "facecount_before": fc_before, "facecount_after": fc_after,
        "facecount_pass": fc_before == fc_after,
        "meta": meta,
        "meta_pass": bool(meta and meta["diameter_mm"] == 8.0 and meta["pitch_mm"] == 1.25),
        "fallbevis_wrong_pitch": {"pass": fallbevis_raised, "raised_msg": fallbevis_msg},
        "pass": bool(
            v_after == v_before and fc_before == fc_after and meta
            and meta["diameter_mm"] == 8.0 and meta["pitch_mm"] == 1.25 and fallbevis_raised
        ),
    }


# --------------------------------------------------------------------------------------------------
# AT3: resolve_expr planted-fault tests + end-to-end exec_ops(params=...) + backward compatibility
# (no params =>
# bit-identical to a plain exec_ops call).
# -------------------------------------------------------------------------------------------------
def _at3_resolve_expr_fallbevis() -> dict:
    out = {}
    out["arithmetic_ok"] = resolve_expr("BORE_D/2", {"BORE_D": 96.0}) == 48.0
    out["nested_arithmetic_ok"] = resolve_expr("(BORE_D + 4) * 2 - 1", {"BORE_D": 10.0}) == 27.0
    out["unary_minus_ok"] = resolve_expr("-BORE_D", {"BORE_D": 5.0}) == -5.0
    unknown_name_raised, unknown_msg = False, ""
    try:
        resolve_expr("BORE_D/2", {"OTHER": 1.0})
    except ValueError as e:
        unknown_name_raised, unknown_msg = True, str(e)
    out["unknown_name_rejected"] = {"pass": unknown_name_raised, "raised_msg": unknown_msg}
    call_raised, call_msg = False, ""
    try:
        resolve_expr("__import__('os').system('true')", {})
    except ValueError as e:
        call_raised, call_msg = True, str(e)
    except Exception as e:  # noqa: BLE001 -- must be ValueError specifically, any other type is itself a fail
        call_raised, call_msg = False, f"WRONG_EXCEPTION_TYPE:{type(e).__name__}:{e}"
    out["function_call_rejected"] = {"pass": call_raised, "raised_msg": call_msg}
    attr_raised, attr_msg = False, ""
    try:
        resolve_expr("BORE_D.__class__", {"BORE_D": 1.0})
    except ValueError as e:
        attr_raised, attr_msg = True, str(e)
    out["attribute_access_rejected"] = {"pass": attr_raised, "raised_msg": attr_msg}
    string_raised = False
    try:
        resolve_expr("'x'", {})
    except ValueError:
        string_raised = True
    out["string_literal_rejected"] = string_raised
    out["pass"] = bool(
        out["arithmetic_ok"] and out["nested_arithmetic_ok"] and out["unary_minus_ok"]
        and out["unknown_name_rejected"]["pass"] and out["function_call_rejected"]["pass"]
        and out["attribute_access_rejected"]["pass"] and out["string_literal_rejected"]
    )
    return out


def _at3d_arithmetic_error_guard_fallbevis() -> dict:
    """-- the 6 legC raw-exception repros from an earlier stress-test corpus's legE_faklassifikation,
    re-run against the wrapped evaluator: every one must now raise ValueError (never the raw built-
    in exception class), naming the expr AND the params that produced it -- never a silent wrong
    value.
    """
    out = {"cases": {}}

    def _expect_valueerror(label, expr, params, must_contain):
        raised, exc_type, msg = False, None, ""
        try:
            resolve_expr(expr, params)
        except ValueError as e:
            raised, exc_type, msg = True, "ValueError", str(e)
        except Exception as e:  # noqa: BLE001 -- a non-ValueError here IS the failure being tested for
            raised, exc_type, msg = False, type(e).__name__, str(e)
        contains_ok = all(tok in msg for tok in must_contain)
        out["cases"][label] = {
            "raised_valueerror": raised, "exc_type": exc_type, "raised_msg": msg,
            "mentions_expr_and_params": contains_ok,
            "pass": bool(raised and contains_ok),
        }

    _expect_valueerror("div_by_zero", "A/B", {"A": 10.0, "B": 0.0}, ["A/B", "ZeroDivisionError", "params"])
    _expect_valueerror("floordiv_by_zero", "A//B", {"A": 10.0, "B": 0.0}, ["A//B", "ZeroDivisionError", "params"])
    _expect_valueerror("mod_by_zero", "A%B", {"A": 10.0, "B": 0.0}, ["A%B", "ZeroDivisionError", "params"])
    _expect_valueerror("zero_over_zero", "A/B", {"A": 0.0, "B": 0.0}, ["A/B", "ZeroDivisionError", "params"])
    _expect_valueerror(
        "inf_via_huge_pow", "BASE**EXP", {"BASE": 10.0, "EXP": 400.0},
        ["BASE**EXP", "OverflowError", "params"],
    )
    _expect_valueerror(
        "neg_base_fractional_power", "BASE**EXP", {"BASE": -4.0, "EXP": 0.5},
        ["BASE**EXP", "complex", "params"],
    )
    # division by zero in 'BORE_D/(N-4)' with N=4
    _expect_valueerror(
        "bore_d_over_n_minus_4", "BORE_D/(N-4)", {"BORE_D": 96.0, "N": 4.0},
        ["BORE_D/(N-4)", "ZeroDivisionError", "N-4", "params"],
    )
    # valid exprs must be COMPLETELY unaffected by the guard (facit from AT3/AT3b re-checked here too).
    out["valid_exprs_unaffected"] = bool(
        resolve_expr("A/B", {"A": 10.0, "B": 2.0}) == 5.0
        and resolve_expr("BASE**EXP", {"BASE": 2.0, "EXP": 10.0}) == 1024.0
        and resolve_expr("BORE_D/2", {"BORE_D": 96.0}) == 48.0
    )
    out["pass"] = bool(
        all(c["pass"] for c in out["cases"].values()) and out["valid_exprs_unaffected"]
    )
    return out


def _at3b_exec_ops_params_end_to_end() -> dict:
    """box (L x L x 10) with radius=BORE_D/2 fillet on all edges, expressed via params+expr; measured
    against the SAME fixture built with a plain literal radius -- both must give the identical volume
    (rel_dev==0.0), proving the expr path and the literal path resolve to the SAME geometry, not just
    the same declared number."""
    L = 30.0
    params = {"BORE_D": 6.0}  # radius = 3.0 -- well inside the L=30 box's valid fillet range

    def _ops(radius_field):
        return [
            {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": L, "height": L, "mode": "add"}]},
            {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": 10.0},
            {"op": "fillet", "id": "filleted", "target_ref": "box", **radius_field},
        ]

    out_expr = exec_ops(_ops({"radius": {"expr": "BORE_D/2"}}), params=params)
    out_literal = exec_ops(_ops({"radius": 3.0}), params=None)
    v_expr = out_expr["solids"]["filleted"].volume
    v_literal = out_literal["solids"]["filleted"].volume
    rel_dev = abs(v_expr - v_literal) / v_literal

    row = next(r for r in out_expr["log"] if r["op_id"] == "filleted")
    prov = row.get("expr_resolutions", [])
    prov_ok = bool(
        len(prov) == 1 and prov[0]["path"] == "radius" and prov[0]["expr"] == "BORE_D/2"
        and prov[0]["resolved_value"] == 3.0 and prov[0]["params_used"] == ["BORE_D"]
    )
    literal_row = next(r for r in out_literal["log"] if r["op_id"] == "filleted")
    no_prov_on_literal_run = "expr_resolutions" not in literal_row

    return {
        "v_expr_mm3": v_expr, "v_literal_mm3": v_literal, "rel_dev": rel_dev,
        "geometry_pass": rel_dev < 1e-9,
        "provenance": {"row": prov, "pass": prov_ok},
        "no_provenance_leak_on_legacy_run": no_prov_on_literal_run,
        "pass": bool(rel_dev < 1e-9 and prov_ok and no_prov_on_literal_run),
    }


def _at3c_backward_compat_data_identity() -> dict:
    """params falsy (None or {}) => the early-exit path leaves `data` as the EXACT SAME dict object
    (module docstring BACKWARD COMPATIBILITY) -- checked via `is`, not just equality, since equality
    alone would not distinguish 'rebuilt an identical copy' from 'never touched it'."""
    op = {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": 1, "height": 1, "mode": "add"}]}
    seen_data_ids = []
    orig_handler = DISPATCH["sketch_2d"]

    def _spy(data, ctx, params):
        seen_data_ids.append(id(data))
        return orig_handler(data, ctx, params)

    DISPATCH["sketch_2d"] = _spy
    try:
        exec_ops([dict(op)], params=None)
        exec_ops([dict(op)], params={})
    finally:
        DISPATCH["sketch_2d"] = orig_handler
    # each call's data object must be the SAME id as the raw dict handed in (only 1 op per call, so
    # any rebuild would show up as a DIFFERENT id than the dict literal constructed just above it) --
    # proven indirectly: re-run once more capturing the input id directly.
    probe = {"op": "sketch_2d", "id": "sk2", "shapes": [{"type": "rectangle", "width": 1, "height": 1, "mode": "add"}]}
    captured = {}

    def _spy2(data, ctx, params):
        captured["id"] = id(data)
        return orig_handler(data, ctx, params)

    DISPATCH["sketch_2d"] = _spy2
    try:
        exec_ops([probe], params=None)
    finally:
        DISPATCH["sketch_2d"] = orig_handler
    identity_preserved = captured["id"] == id(probe)
    return {"identity_preserved_params_none": identity_preserved, "pass": identity_preserved}


def _at2h_rib_on_bracket() -> dict:
    """rib op type through the DISPATCH path: a stiffener straddling the synthetic bracket plate's
    top face. Reference value: closed-form box arithmetic (plate and strip are both axis-aligned),
    so the expected volume is independent of the execution machinery."""
    L, B, H = 140.0, 90.0, 12.0
    thickness, run_length, depth = 6.0, 60.0, 16.0
    ops = [
        {"op": "sketch_2d", "id": "sk_rib_plate",
         "shapes": [{"type": "rectangle", "width": L, "height": B, "mode": "add"}]},
        {"op": "extrude", "id": "rib_plate", "sketch_ref": "sk_rib_plate", "amount": H},
        {"op": "rib", "id": "rib1", "target_ref": "rib_plate",
         "midplane": {"origin": [0.0, 0.0, H], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]},
         "thickness": thickness, "run_length": run_length, "depth": depth},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["rib1"].volume
    # the strip spans Z in [H-depth/2, H+depth/2]; only the part above Z=H is new material
    expected = L * B * H + thickness * run_length * (depth / 2.0)
    rel_dev = abs(measured - expected) / expected
    return {"measured": measured, "expected": expected, "rel_dev": rel_dev,
            "pass": rel_dev < 1e-6, "log": out["log"]}


def _at2i_sketch_profile_extrude() -> dict:
    """sketch_profile op type through the DISPATCH path: the bracket plate outline declared as a
    constraint graph (two free points solved by the GCS, two pinned) and extruded. Reference value:
    the analytic prism volume of the dimensioned rectangle."""
    L, B, H = 140.0, 90.0, 12.0
    ops = [
        {"op": "sketch_profile", "id": "skp_plate",
         "points": {
             "p0": {"x": -L / 2.0, "y": -B / 2.0, "fixed": True},
             # seeded off the solution on purpose: the solver must move it onto the dimension
             "p1": {"x": L / 2.0 - 7.0, "y": -B / 2.0 + 3.0, "fixed": False},
             "p2": {"x": L / 2.0, "y": B / 2.0, "fixed": True},
             "p3": {"x": -L / 2.0, "y": B / 2.0, "fixed": True},
         },
         "lines": {"L01": ["p0", "p1"]},
         "constraints": [
             {"type": "horizontal", "line": "L01", "name": "bottom_h"},
             {"type": "distance_p2p", "p1": "p0", "p2": "p1", "value": L, "name": "plate_width"},
         ],
         "profile": {"type": "polygon", "point_order": ["p0", "p1", "p2", "p3"]}},
        {"op": "extrude", "id": "skp_plate_solid", "sketch_ref": "skp_plate", "amount": H},
    ]
    out = exec_ops(ops)
    measured = out["solids"]["skp_plate_solid"].volume
    expected = L * B * H
    rel_dev = abs(measured - expected) / expected
    meta = next((r.get("sketch_profile_meta") for r in out["log"] if r["op_id"] == "skp_plate"), None)
    status = (meta or {}).get("status")
    return {"measured": measured, "expected": expected, "rel_dev": rel_dev,
            "solver_status": status,
            "pass": bool(rel_dev < 1e-6 and status == "FULLT_BESTAMD"), "log": out["log"]}


def _at2j_all_op_types_dispatch() -> dict:
    """Every op type declared in the schema has a handler, except `reference`, which the schema
    validates and the executor deliberately refuses with NotImplementedError (AT2d). Both lazily
    imported handler modules (rib, sketch_profile) must import in this repository."""
    from cad_op_schema_v1 import OP_SPECS

    declared = set(OP_SPECS)
    handled = set(DISPATCH)
    missing = sorted(declared - handled - {"reference"})
    lazy = {}
    for mod, sym in (("formrib_v1", "rib_v1"), ("sketch_gcs_v1", "SketchGCS")):
        try:
            lazy[mod] = hasattr(__import__(mod), sym)
        except Exception as exc:  # noqa: BLE001
            lazy[mod] = f"{type(exc).__name__}: {exc}"
    return {"n_declared": len(declared), "n_handlers": len(handled),
            "declared_not_implemented": ["reference"], "missing_handlers": missing,
            "lazy_imports": lazy,
            "pass": bool(not missing and all(v is True for v in lazy.values()))}


def _selftest() -> dict:
    results = {}
    results["AT2a_box_hole"] = _at2a_box_hole()
    results["AT2a_loft_two_profiles"] = _at2a_loft_two_profiles()
    results["AT2a_cut_series"] = _at2a_cut_series()
    results["AT2a_pass"] = (results["AT2a_box_hole"]["pass"] and results["AT2a_loft_two_profiles"]["pass"]
                             and results["AT2a_cut_series"]["pass"])
    results["AT2b_tomcut_fallbevis"] = _at2b_tomcut_fallbevis()
    results["AT2d_reference_not_implemented"] = _at2d_reference_not_implemented()
    results["AT2e_draft_cube"] = _at2e_draft_cube()
    results["AT2f_sheet_bend_l_bracket"] = _at2f_sheet_bend_l_bracket()
    results["AT2g_thread_cosmetic"] = _at2g_thread_cosmetic()
    results["AT2h_rib_on_bracket"] = _at2h_rib_on_bracket()
    results["AT2i_sketch_profile_extrude"] = _at2i_sketch_profile_extrude()
    results["AT2j_all_op_types_dispatch"] = _at2j_all_op_types_dispatch()
    results["AT3_resolve_expr_fallbevis"] = _at3_resolve_expr_fallbevis()
    results["AT3b_exec_ops_params_end_to_end"] = _at3b_exec_ops_params_end_to_end()
    results["AT3c_backward_compat_data_identity"] = _at3c_backward_compat_data_identity()
    results["AT3d_arithmetic_error_guard_fallbevis"] = _at3d_arithmetic_error_guard_fallbevis()
    results["all_pass"] = bool(
        results["AT2a_pass"] and results["AT2b_tomcut_fallbevis"]["pass"]
        and results["AT2d_reference_not_implemented"]["pass"]
        and results["AT2e_draft_cube"]["pass"] and results["AT2e_draft_cube"]["fallbevis"]["pass"]
        and results["AT2f_sheet_bend_l_bracket"]["pass"] and results["AT2g_thread_cosmetic"]["pass"]
        and results["AT2h_rib_on_bracket"]["pass"] and results["AT2i_sketch_profile_extrude"]["pass"]
        and results["AT2j_all_op_types_dispatch"]["pass"]
        and results["AT3_resolve_expr_fallbevis"]["pass"] and results["AT3b_exec_ops_params_end_to_end"]["pass"]
        and results["AT3c_backward_compat_data_identity"]["pass"]
        and results["AT3d_arithmetic_error_guard_fallbevis"]["pass"]
    )
    return results


if __name__ == "__main__":
    out = _selftest()
    # trim verbose per-op logs from the printed/stdout summary (kept in the written log files instead)
    printable = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "log"})
                 for k, v in out.items()}
    print(json.dumps(printable, indent=2))
    sys.exit(0 if out["all_pass"] else 1)
