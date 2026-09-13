#!/usr/bin/env python3
"""cad_op_schema_v1.py -- closed-keyset schema for the CAD operation steps a recipe is made of.

A recipe is a dict {"ops": [op,...], "params": {name: number}?}. Each op is a plain dict with an
"op" type, a globally unique "id" and a fixed set of keys. Twenty-one op types are enumerated here
(sketch_2d, sketch_profile, extrude, revolve, loft, sweep, the three booleans, fillet, chamfer,
shell, mirror, pattern_linear, pattern_circular, hole, draft, rib, sheet_bend, thread_cosmetic,
reference). An unknown op type, an unknown key inside a known op, or a missing required key raises
ValueError; nothing is ever skipped or coerced silently.

Values are not inspected, only key names, so any numeric field may be given as {"expr": "BORE_D/2"}
against the recipe's top-level "params"; resolution of those expressions belongs to cad_op_exec_v1.
If "params" is present, every value in it must be a real number.

API:
validate_op(op: dict) -> OpStep              # ValueError(unknown_key|unknown_type|missing_key)
validate_recipe(recipe: dict) -> list[OpStep]  # also enforces globally unique op ids

Ops that carry a sketch plane get the XY plane written into the op dict when "plane" is missing, so
downstream readers never have to guess a default. An explicitly given plane is left untouched.

Run the selftest with `python cad_op_schema_v1.py`; it prints a JSON report and exits non-zero if a
check fails.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------------------------------
# - THE 17-TYPE CLOSED SET. Nothing outside this dict is a valid op-type -- an op-type not present
# here (e.g. a typo'd "drafft") raises ValueError at validate_op() time (AT1b planted-fault test).
# Each entry: {"required": frozenset(...), "optional": frozenset(...)}.
# -------------------------------------------------------------------------------------------------
COMMON_KEYS = frozenset({"op", "id"})

OP_SPECS: dict[str, dict[str, frozenset]] = {
    "sketch_2d": {
        "required": frozenset({"shapes"}),
        "optional": frozenset({"plane"}),
    },
    "extrude": {
        "required": frozenset({"sketch_ref", "amount"}),
        "optional": frozenset({"mode", "taper"}),
    },
    "revolve": {
        "required": frozenset({"sketch_ref", "axis"}),
        "optional": frozenset({"angle_deg", "mode"}),
    },
    "loft": {
        "required": frozenset({"sketch_refs"}),
        "optional": frozenset({"ruled", "mode"}),
    },
    "sweep": {
        "required": frozenset({"sketch_ref", "path_ref"}),
        "optional": frozenset({"mode"}),
    },
    "fillet": {
        # radius XOR radius_frac: "radius" is a fixed absolute mm value (original contract,
        # unchanged); "radius_frac" is an ALTERNATIVE value form -- a fraction of the MEASURED min-
        # edge-length of the selector's GEOMETRIC population (geometry_type/radius/
        # normal/direction/min_area/max_area/wall/near_point -- what the selector geometrically
        # MEANS), measured BEFORE the safety filters (min_length/max_length/vertex_safe_k -- which
        # edges are safe to actually TOUCH) subtract short/vertex-adjacent edges. MEASURED: scaling
        # off the safety- FILTERED population's own min length instead inflates the resolved radius
        # past what the surviving edges can individually tolerate, because a
        # min_length/vertex_safe_k-filtered population's min length is ALWAYS >= the shape's true
        # local minimum by construction of the filter itself (see
        # geometri_selektor_v1.geometric_edge_population docstring for the full decision + rejected
        # cap/clamp alternative). Measured live by cad_op_exec_v1 immediately BEFORE the destructive
        # call. Exactly one of radius/radius_frac must be present -- enforced below in validate_op
        # (_VALUE_XOR_SPECS), not by the required/optional keysets alone (neither key is
        # individually required; the pair is jointly required).
        "required": frozenset({"target_ref"}),
        # expected_population (exec_populationsvakt_v1, ALL-POPULATIONSVAKTEN): the edge population
        # of target_ref the recipe author MEASURED (geometri_selektor_v1.record_expected_population)
        # when selector is omitted (a selector=None/'ALL' chain position) -- cad_op_exec_v1.exec_ops
        # checks it before the destructive fillet call and refuses (RefusalError POPULATION_DRIFT) on
        # drift. Optional + ignored when "selector" is present (selector_match already scopes the
        # population explicitly in that case). Omitting it is backward-compatible (undeclared/legacy
        # path -- warns + flags in the log, never raises); see cad_op_exec_v1.py module docstring.
        "optional": frozenset({"radius", "radius_frac", "selector", "expected_population"}),
    },
    "chamfer": {
        # distance XOR radius_frac -- same adaptive-value contract as fillet above (radius_frac's
        # measured fraction is applied to `distance` here, the chamfer's own scalar).
        "required": frozenset({"target_ref"}),
        "optional": frozenset({"distance", "radius_frac", "selector", "expected_population"}),
    },
    "hole": {
        "required": frozenset({"target_ref", "center", "diameter"}),
        "optional": frozenset({"depth", "through", "axis"}),
    },
    "boolean_union": {
        "required": frozenset({"base_ref", "tool_refs"}),
        "optional": frozenset(),
    },
    "boolean_cut": {
        "required": frozenset({"base_ref", "tool_refs"}),
        "optional": frozenset(),
    },
    "boolean_intersect": {
        "required": frozenset({"base_ref", "tool_refs"}),
        "optional": frozenset(),
    },
    "pattern_linear": {
        "required": frozenset({"target_ref", "direction", "count", "spacing"}),
        "optional": frozenset(),
    },
    "pattern_circular": {
        "required": frozenset({"target_ref", "axis", "count"}),
        "optional": frozenset({"angle_deg"}),
    },
    "mirror": {
        "required": frozenset({"target_ref", "plane"}),
        "optional": frozenset(),
    },
    "shell": {
        "required": frozenset({"target_ref", "thickness"}),
        "optional": frozenset({"faces_to_remove"}),
    },
    "reference": {
        "required": frozenset({"kind"}),
        "optional": frozenset({"origin", "direction"}),
    },
    "draft": {
        # post-hoc release-angle: applied to an ALREADY-BUILT solid, geometrically-selected faces
        # (selector, same geometri_selektor_v1 contract as fillet/chamfer) or -- if selector is
        # omitted -- every planar face parallel to pull_direction. neutral_plane is the parting-line
        # reference the selected faces are held fixed against (does NOT need to be an extrude end).
        "required": frozenset({"target_ref", "angle_deg", "pull_direction", "neutral_plane"}),
        # expected_population: same ALL-POPULATIONSVAKTEN contract as fillet/chamfer above, guarding
        # the FACE population of target_ref when selector is omitted (draft's own auto-detect path).
        "optional": frozenset({"selector", "expected_population"}),
    },
    "sheet_bend": {
        # isometric plate bend -- a straight bend_line splits target_ref's flat sheet into a FIXED
        # side (unchanged) and a ROTATING side, joined by a real curved bend-zone solid (swept
        # cylindrical wall) whose flat-pattern width is the bend allowance BA=angle_rad*(radius+
        # k_factor*thickness), DIN 6935's standard formula. PILOT SCOPE: target_ref's thickness axis
        # MUST be world Z, bend_line MUST be parallel to world Y -- other orientations raise
        # NotImplementedError. "side" is reserved for a future extension (only "positive" -- the
        # higher-X side rotates -- is implemented; any other value raises NotImplementedError, never
        # silently wrong).
        "required": frozenset({"target_ref", "bend_line", "angle_deg", "radius", "thickness"}),
        "optional": frozenset({"k_factor", "side"}),
    },
    "thread_cosmetic": {
        # . A cosmetic/annotation-only thread: NO geometry is cut -- target_ref's volume is bit-
        # identical before/after (rel_dev==0.0, the strongest possible no-op invariant). selector
        # must resolve to EXACTLY one CYLINDER face; designation ("M8" or "M8x1.25") is validated
        # against iso_thread_table_v1's ISO 724 coarse-series (coarse series) table -- an unknown
        # diameter or a mismatched pitch raises ValueError before anything is touched. Chosen over a
        # modelled helix because the ONLY declared consumer, mate_check_v1.json's own pin/socket
        # mating chain, reads nominal diameter+pitch METADATA -- never a spiral BRep surface.
        "required": frozenset({"target_ref", "selector", "designation"}),
        "optional": frozenset(),
    },
    "rib": {
        # DISPATCH-wiring of the rib recipe-idiom that an earlier stress-test corpus's prior art
        # measured as "callable but not wired into DISPATCH". This is a REAL 20th op-type --
        # promoted here so a recipe author can address a rib step directly by id/target_ref inside
        # one exec_ops() chain, same as fillet/chamfer/draft, instead of having to drop out of the
        # ops-list into a separate Python call. midplane/ thickness/run_length/depth map 1:1 onto
        # formrib_v1.rib_v1's own positional args.
        "required": frozenset({"target_ref", "midplane", "thickness", "run_length", "depth"}),
        "optional": frozenset(),
    },
    "sketch_profile": {
        # name-keyed sketch_gcs_v1.SketchGCS graph + a "profile" spec naming the solved boundary --
        # see module docstring SKETCH PROFILE. "points" (required): {name:
        # {"x":float,"y":float,"fixed":bool?}} -- every unknown/fixed point in the sketch, INCLUDING
        # construction points not on the final profile boundary (e.g. a chamfer's un-chamfered
        # corner, kept as a solver unknown but excluded from "profile.point_order"). "constraints"
        # (required): list of {"type": <sketch_gcs_v1 constraint type>, "name": str?,...type-
        # specific keys, string point/line/circle/param NAMES not integer ids}. "profile"
        # (required): {"type": "polygon", "point_order": [name,...]} -- ordered solved point names
        # forming the closed boundary build123d consumes (bd.Polygon). Other profile["type"] values
        # are a valid FUTURE extension (e.g. spline-bounded) but raise NotImplementedError at exec
        # time today. "plane" (optional): identical shape/semantics to sketch_2d's own "plane" key.
        # "lines"/"circles" (optional): {name: [p1_name, p2_name]} / {name: {"center": p_name,
        # "radius_param": param_name}} -- named entities the constraints list can address.
        # "sketch_params" (optional): {name: {"value": float, "fixed": bool?}} -- scalar unknowns
        # (e.g. a circle's radius) distinct from "points"; SketchGCS.add_param, not add_point.
        # "positive_params" (optional): list of sketch_params names the post-solve geometric
        # validity gate (sketch_gcs_v1 §5.3 anchor) must find > 0 and finite. Defaults (when
        # omitted) to every sketch_param referenced as a circle's "radius_param" -- a radius that
        # solved negative/degenerate is a GEOMETRISKT_OGILTIG refusal even when residual-convergence
        # alone said "solved", never a silent pass (same principle sketch_gcs_v1.SketchGCS.solve
        # already enforces; this key only lets a recipe author name additional positive unknowns
        # beyond that auto-detected default, e.g. a distance that must stay positive). "arcs":
        # {name: {"circle": circle_name, "p_start": p_name, "p_end": p_name, "ccw": bool?}} -- a
        # bounded piece of an already declared circle. Adds NO solver unknowns (the endpoints are
        # ordinary points the constraints pin onto the circle), so every existing rank/dof/blame
        # diagnostic is unchanged. Enables profile {"type": "arc_chain", "arc_order": [name,...]},
        # which exec turns into REAL circular build123d edges -> the extruded BREP face is an exact
        # CYLINDER. MEASURED reason this exists: CADGenBench part 231's outer boundary is a 12-arc
        # tangent chain (3 lobe R20 + 6 concave blend R70 + 3 hub R80); the polygon-only profile
        # could not express the CONCAVE blends, which is where the boolean/hull path booked its
        # +18.52 % volume error. "splines": {name: {"point_order": [p_name,...], "periodic": bool?}}
        # -- an interpolating B-spline through ALREADY DECLARED solved points. Adds NO solver
        # unknowns (identical stance to "arcs"): the control/interpolation points are ordinary
        # sketch points the constraints may or may not pin, so every existing rank/dof/blame
        # diagnostic is unchanged. profile {"type": "mixed_chain", "segment_order": [{"kind":
        # "line"|"arc"|"spline", "line"/"arc"/"spline": <name>},...], "continuity": [{"at": int,
        # "type": "G1", "tol_deg": float?}]?} -- a closed boundary of ARBITRARILY INTERLEAVED
        # line/arc/spline segments. MEASURED reason this exists: on the 32 real CADGenBench
        # industrial parts the prismatic cross-section boundary is a general closed contour;
        # "polygon" is LOSSY (it chords every curved edge) and "arc_chain" cannot carry a straight
        # segment at all, so no existing profile primitive could express "line-arc-line-arc". Exact
        # geometry: each segment becomes a REAL build123d Edge (Line/CIRCLE/BSPLINE), never a chord.
        # The closure gate is the SAME one arc_chain already enforces (Wire.combine -> exactly ONE
        # wire and is_closed), reused verbatim; "continuity" is an ADDITIONAL two-sided tangent-
        # angle gate measured on the built edges (default 1.0 deg), never a narrated claim.
        "required": frozenset({"points", "constraints", "profile"}),
        "optional": frozenset({"plane", "lines", "circles", "arcs", "splines", "sketch_params",
                               "positive_params", "_arc_ids"}),
    },
}

assert len(OP_SPECS) == 21, f"schema drift: expected 21 op-types, got {len(OP_SPECS)}"

# _VALUE_XOR_SPECS: op-types where two keys are JOINTLY required (exactly one of the pair, never
# both, never neither) -- the required/optional keyset check above cannot express this (both keys
# are individually optional there), so validate_op enforces it as a second pass, keyed by op_type.
_VALUE_XOR_SPECS: dict[str, tuple[str, str]] = {
    "fillet": ("radius", "radius_frac"),
    "chamfer": ("distance", "radius_frac"),
}

# PLANE NORMALIZATION: "plane" is optional on sketch_2d/sketch_profile, and
# cad_op_exec_v1._sketch_plane already defaults a missing/None plane_spec to bd.Plane.XY (origin
# (0,0,0), x_dir (1,0,0), z_dir (0,0,1)) -- but recept_till_ikarus_v1.py's extrude-handler reads
# sk['plane']['origin'|'x_dir'|'z_dir'] with NO default, raising an uncaught KeyError on any recipe
# that omits 'plane' (measured: 17/18 v2+v3 torture recipes; the SDF sweep's own harness
# (_preview_paritet_svep_v1.py _normalize_ops) had to inject this same default as a TEST-ONLY
# workaround to see past op #1). MEASURED before choosing the fix: every sketch_2d op across every
# JSON that omits 'plane' (66 ops / 60 files) is a stress-test fixture -- ZERO production recipes
# omit it. Making 'plane' REQUIRED (option a) would therefore only break 60
# fixture files for no parity benefit; making it schema-NORMALIZED (option b) closes the two-
# readers-one-field seam for every current AND future caller that runs validate_op/validate_recipe
# as a gate. Chosen: (b).
_DEFAULT_PLANE_XY: dict = {"origin": [0.0, 0.0, 0.0], "x_dir": [1.0, 0.0, 0.0], "z_dir": [0.0, 0.0, 1.0]}
_PLANE_BEARING_OP_TYPES = frozenset({"sketch_2d", "sketch_profile"})


@dataclass(frozen=True)
class OpStep:
    op_type: str
    op_id: str
    data: dict = field(default_factory=dict)  # the full validated raw dict (incl. op/id)


def validate_op(op: Any) -> OpStep:
    """Validate ONE op dict against the closed keyset. Raises ValueError on:
    - not a dict
    - missing 'op' or 'id'
    - unknown op-type (not one of the 17)
    - unknown key for that op-type (the typo trap, AT1c)
    - missing required key for that op-type
    Never silently skips or coerces -- matches this component's api contract verbatim.
    """
    if not isinstance(op, dict):
        raise ValueError(f"op must be a dict, got {type(op).__name__}")
    if "op" not in op:
        raise ValueError("op dict missing required key 'op' (the op-type)")
    if "id" not in op:
        raise ValueError("op dict missing required key 'id'")
    op_type = op["op"]
    op_id = op["id"]
    if op_type not in OP_SPECS:
        raise ValueError(
            f"unknown op-type {op_type!r} (id={op_id!r}) -- not in the {len(OP_SPECS)}-member closed "
            f"set {sorted(OP_SPECS)}. A silently-skipped op-type is exactly the catch-all failure "
            f"this schema exists to prevent."
        )
    spec = OP_SPECS[op_type]
    allowed = COMMON_KEYS | spec["required"] | spec["optional"]
    unknown = set(op.keys()) - allowed
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} for op-type {op_type!r} (id={op_id!r}); "
            f"allowed keys = {sorted(allowed)}"
        )
    missing = spec["required"] - set(op.keys())
    if missing:
        raise ValueError(f"missing required key(s) {sorted(missing)} for op-type {op_type!r} (id={op_id!r})")
    xor_pair = _VALUE_XOR_SPECS.get(op_type)
    if xor_pair is not None:
        a, b = xor_pair
        has_a, has_b = a in op, b in op
        if has_a == has_b:  # both present or both absent -- neither is a valid recipe
            raise ValueError(
                f"op-type {op_type!r} (id={op_id!r}) requires EXACTLY ONE of {a!r}/{b!r} "
                f"(got {'both' if has_a else 'neither'})"
            )
    if op_type in _PLANE_BEARING_OP_TYPES and not op.get("plane"):
        # NORMALIZE, don't leave implicit: write cad_op_exec_v1._sketch_plane's own XY default IN
        # PLACE on the caller's op dict (mutates `op`, not just the returned OpStep.data) so every
        # consumer downstream of this gate -- BRep (cad_op_exec_v1) and ikarus (recept_till_ikarus_v1)
        # alike -- observes the identical explicit {"origin","x_dir","z_dir"} value post-validation.
        # See _DEFAULT_PLANE_XY / PLANE NORMALIZATION note above for the measured rationale.
        op["plane"] = dict(_DEFAULT_PLANE_XY)
    return OpStep(op_type=op_type, op_id=op_id, data=dict(op))


def validate_params_block(params: Any) -> dict:
    """validate a recipe's optional top-level "params" value: must be a dict, every key a str, every
    value a real number (bool excluded -- isinstance(True, int) is True in Python, which would
    silently accept a typo'd boolean as a usable arithmetic parameter). Raises ValueError naming the
    offending key on any violation. Called by validate_recipe ONLY when "params" is present -- a
    recipe without the key never reaches this function (backward-compat: absence is a no-op, never
    validated as if it were an empty dict with different semantics).
    """
    if not isinstance(params, dict):
        raise ValueError(f"recipe['params'] must be a dict, got {type(params).__name__}")
    for k, v in params.items():
        if not isinstance(k, str):
            raise ValueError(f"recipe['params'] key {k!r} is not a string")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"recipe['params'][{k!r}] = {v!r} is not a number (got {type(v).__name__})")
    return dict(params)


def validate_recipe(recipe: dict) -> list[OpStep]:
    """Validate a whole recipe {"ops": [...], "params": {...}?}. Enforces globally-unique 'id's across
    the recipe (downstream refs -- sketch_ref/target_ref/base_ref/tool_refs/path_ref -- address
    these ids). "params" is OPTIONAL and, when present, validated by validate_params_block -- a
    recipe without it is completely unaffected (no new required key, no behaviour change for every
    pre-existing recipe in the repo).
    """
    if not isinstance(recipe, dict) or "ops" not in recipe:
        raise ValueError("recipe must be a dict with an 'ops' key (list of op dicts)")
    if "params" in recipe:
        validate_params_block(recipe["params"])
    ops_raw = recipe["ops"]
    if not isinstance(ops_raw, list) or not ops_raw:
        raise ValueError("recipe['ops'] must be a non-empty list")
    steps: list[OpStep] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(ops_raw):
        step = validate_op(raw)
        if step.op_id in seen_ids:
            raise ValueError(f"duplicate op id {step.op_id!r} at ops[{i}] -- ids must be globally unique")
        seen_ids.add(step.op_id)
        steps.append(step)
    return steps


# --------------------------------------------------------------------------------------------------
# - SELFTEST -- AT1a (17/17 minimal valid examples), AT1b, AT1c (unknown key in a valid op =>
# ValueError, typo trap)
# -------------------------------------------------------------------------------------------------
def _minimal_examples() -> dict[str, dict]:
    return {
        "sketch_2d": {"op": "sketch_2d", "id": "sk1", "shapes": [{"type": "rectangle", "width": 10, "height": 10, "mode": "add"}]},
        "extrude": {"op": "extrude", "id": "ex1", "sketch_ref": "sk1", "amount": 5},
        "revolve": {"op": "revolve", "id": "rv1", "sketch_ref": "sk1", "axis": "Z"},
        "loft": {"op": "loft", "id": "lf1", "sketch_refs": ["sk1", "sk2"]},
        "sweep": {"op": "sweep", "id": "sw1", "sketch_ref": "sk1", "path_ref": "path1"},
        "fillet": {"op": "fillet", "id": "fl1", "target_ref": "ex1", "radius": 1.0},
        "chamfer": {"op": "chamfer", "id": "ch1", "target_ref": "ex1", "distance": 1.0},
        "hole": {"op": "hole", "id": "ho1", "target_ref": "ex1", "center": [0, 0, 0], "diameter": 3.0},
        "boolean_union": {"op": "boolean_union", "id": "bu1", "base_ref": "ex1", "tool_refs": ["ex2"]},
        "boolean_cut": {"op": "boolean_cut", "id": "bc1", "base_ref": "ex1", "tool_refs": ["ex2"]},
        "boolean_intersect": {"op": "boolean_intersect", "id": "bi1", "base_ref": "ex1", "tool_refs": ["ex2"]},
        "pattern_linear": {"op": "pattern_linear", "id": "pl1", "target_ref": "ex1", "direction": [1, 0, 0], "count": 3, "spacing": 10},
        "pattern_circular": {"op": "pattern_circular", "id": "pc1", "target_ref": "ex1", "axis": "Z", "count": 4},
        "mirror": {"op": "mirror", "id": "mr1", "target_ref": "ex1", "plane": "XZ"},
        "shell": {"op": "shell", "id": "sh1", "target_ref": "ex1", "thickness": 1.5},
        "reference": {"op": "reference", "id": "rf1", "kind": "plane"},
        "draft": {"op": "draft", "id": "dr1", "target_ref": "ex1", "angle_deg": 3.0,
                  "pull_direction": [0, 0, 1], "neutral_plane": {"origin": [0, 0, 0], "normal": [0, 0, 1]}},
        "sheet_bend": {"op": "sheet_bend", "id": "sb1", "target_ref": "ex1",
                       "bend_line": [[10, 0, 0], [10, 5, 0]], "angle_deg": 90.0, "radius": 3.0,
                       "thickness": 2.0},
        "thread_cosmetic": {"op": "thread_cosmetic", "id": "tc1", "target_ref": "ex1",
                             "selector": {"entity": "face", "geometry_type": "CYLINDER", "expected_count": 1},
                             "designation": "M8"},
        "rib": {"op": "rib", "id": "rb1", "target_ref": "ex1",
                "midplane": {"origin": [0, 0, 0], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]},
                "thickness": 2.0, "run_length": 10.0, "depth": 5.0},
        "sketch_profile": {
            "op": "sketch_profile", "id": "skp1",
            "points": {
                "p0": {"x": 0.0, "y": 0.0, "fixed": True},
                "p1": {"x": 10.0, "y": 0.0, "fixed": False},
                "p2": {"x": 10.0, "y": 10.0, "fixed": True},
                "p3": {"x": 0.0, "y": 10.0, "fixed": True},
            },
            "lines": {"L01": ["p0", "p1"]},
            "constraints": [
                {"type": "horizontal", "line": "L01", "name": "bottom_h"},
                {"type": "distance_p2p", "p1": "p0", "p2": "p1", "value": 10.0, "name": "width"},
            ],
            "profile": {"type": "polygon", "point_order": ["p0", "p1", "p2", "p3"]},
        },
    }


def _selftest() -> dict:
    results = {}

    # AT1a: all 17 op-types validate a minimal valid example each
    examples = _minimal_examples()
    at1a_pass = 0
    at1a_detail = {}
    for op_type, ex in examples.items():
        try:
            step = validate_op(ex)
            ok = step.op_type == op_type
            at1a_pass += int(ok)
            at1a_detail[op_type] = "PASS" if ok else "FAIL-mismatch"
        except Exception as e:  # noqa: BLE001 -- selftest wants to record, not crash
            at1a_detail[op_type] = f"FAIL-raised:{e}"
    results["AT1a"] = {"pass_count": at1a_pass, "total": len(OP_SPECS), "pass": at1a_pass == len(OP_SPECS),
                        "detail": at1a_detail}

    # AT1b: unknown op-type => ValueError, never silent skip. REFRAME DECLARED: this fixture used to
    # be literally "draft" (draft was outside the 16-type set at the time). draft is now the 17th
    # real op-type --testing "draft" here would silently start exercising the MISSING-REQUIRED-KEY
    # path instead of the UNKNOWN-OP-TYPE path (both raise ValueError, but they are different
    # assertions; conflating them would let a real unknown-op-type regression hide behind a still-
    # green AT1b). Swapped to an op-type that is genuinely absent from OP_SPECS.
    at1b_ok = False
    at1b_msg = ""
    try:
        validate_op({"op": "extrude_helical_thread", "id": "d1", "target_ref": "ex1"})
    except ValueError as e:
        at1b_ok = True
        at1b_msg = str(e)
    results["AT1b"] = {"pass": at1b_ok, "raised_msg": at1b_msg}

    # AT1d: draft's OWN missing-required-key path -- 'draft' is now a KNOWN op-type, so a minimal
    # call missing angle_deg/pull_direction/neutral_plane must still raise ValueError, but via the
    # missing-required-key branch, not unknown-op-type.
    at1d_ok = False
    at1d_msg = ""
    try:
        validate_op({"op": "draft", "id": "d2", "target_ref": "ex1"})
    except ValueError as e:
        at1d_ok = True
        at1d_msg = str(e)
    results["AT1d"] = {"pass": at1d_ok, "raised_msg": at1d_msg,
                        "desc": "draft is a KNOWN op-type now; missing required keys still raises ValueError"}

    # AT1c: unknown key in a valid op => ValueError (the typo trap, e.g. a "radiuss" typo)
    at1c_ok = False
    at1c_msg = ""
    try:
        validate_op({"op": "fillet", "id": "fl2", "target_ref": "ex1", "radiuss": 1.0})
    except ValueError as e:
        at1c_ok = True
        at1c_msg = str(e)
    results["AT1c"] = {"pass": at1c_ok, "raised_msg": at1c_msg}

    # bonus: validate_recipe end-to-end + duplicate-id fall-bevis
    recipe = {"ops": [examples["sketch_2d"], examples["extrude"]]}
    steps = validate_recipe(recipe)
    dup_ok = False
    try:
        validate_recipe({"ops": [examples["sketch_2d"], examples["sketch_2d"]]})
    except ValueError:
        dup_ok = True
    results["recipe_roundtrip"] = {"n_steps": len(steps), "pass": len(steps) == 2}
    results["duplicate_id_rejected"] = {"pass": dup_ok}

    # AT1e: fillet/chamfer radius XOR radius_frac -- neither present AND both present must BOTH
    # raise ValueError; radius_frac ALONE (the new alternative form) must PASS.
    at1e = {}
    try:
        validate_op({"op": "fillet", "id": "fl3", "target_ref": "ex1"})
        at1e["neither_rejected"] = False
    except ValueError:
        at1e["neither_rejected"] = True
    try:
        validate_op({"op": "fillet", "id": "fl4", "target_ref": "ex1", "radius": 1.0, "radius_frac": 0.1})
        at1e["both_rejected"] = False
    except ValueError:
        at1e["both_rejected"] = True
    try:
        step = validate_op({"op": "fillet", "id": "fl5", "target_ref": "ex1", "radius_frac": 0.1})
        at1e["radius_frac_alone_accepted"] = step.op_type == "fillet"
    except ValueError:
        at1e["radius_frac_alone_accepted"] = False
    try:
        step = validate_op({"op": "chamfer", "id": "ch2", "target_ref": "ex1", "radius_frac": 0.1})
        at1e["chamfer_radius_frac_alone_accepted"] = step.op_type == "chamfer"
    except ValueError:
        at1e["chamfer_radius_frac_alone_accepted"] = False
    at1e["pass"] = bool(
        at1e["neither_rejected"] and at1e["both_rejected"] and at1e["radius_frac_alone_accepted"]
        and at1e["chamfer_radius_frac_alone_accepted"]
    )
    results["AT1e_radius_frac_xor"] = at1e

    # AT1f: recipe-level "params" block -- absent is a no-op (backward compat), a well-formed
    # numeric dict passes, a malformed one (non-numeric value / bool value / non-dict) is rejected
    # with a named-key ValueError. An "expr" dict standing in for a plain scalar value (e.g. fillet
    # "radius": {"expr": "BORE_D/2"}) must ALSO validate cleanly -- this module's key-name-only
    # check never inspects value shape (see module docstring "SYMBOLIC EXPRESSIONS").
    at1f = {}
    try:
        validate_recipe({"ops": [examples["sketch_2d"], examples["extrude"]]})  # no params key at all
        at1f["absent_params_is_noop"] = True
    except ValueError:
        at1f["absent_params_is_noop"] = False
    try:
        validate_recipe({"ops": [examples["sketch_2d"], examples["extrude"]], "params": {"BORE_D": 96.0}})
        at1f["well_formed_params_accepted"] = True
    except ValueError:
        at1f["well_formed_params_accepted"] = False
    try:
        validate_recipe({"ops": [examples["sketch_2d"], examples["extrude"]], "params": {"BORE_D": "96"}})
        at1f["non_numeric_param_rejected"] = False
    except ValueError:
        at1f["non_numeric_param_rejected"] = True
    try:
        validate_recipe({"ops": [examples["sketch_2d"], examples["extrude"]], "params": {"FLAG": True}})
        at1f["bool_param_rejected"] = False
    except ValueError:
        at1f["bool_param_rejected"] = True
    try:
        expr_fillet = {"op": "fillet", "id": "fl6", "target_ref": "ex1", "radius": {"expr": "BORE_D/2"}}
        step = validate_op(expr_fillet)
        at1f["expr_value_shape_accepted"] = step.op_type == "fillet"
    except ValueError:
        at1f["expr_value_shape_accepted"] = False
    at1f["pass"] = bool(
        at1f["absent_params_is_noop"] and at1f["well_formed_params_accepted"]
        and at1f["non_numeric_param_rejected"] and at1f["bool_param_rejected"]
        and at1f["expr_value_shape_accepted"]
    )
    results["AT1f_params_block"] = at1f

    # AT1g: sketch_2d/sketch_profile ops that OMIT 'plane' get the default written IN PLACE by
    # validate_op -- both the mutated input dict and the returned OpStep.data must carry the
    # explicit XY triple; an op that already carries an explicit (non-XY) plane must be left
    # untouched (normalization never overwrites an authored value).
    at1g = {}
    op_no_plane = {"op": "sketch_2d", "id": "skn1", "shapes": [{"type": "rectangle", "width": 5, "height": 5, "mode": "add"}]}
    step_n = validate_op(op_no_plane)
    at1g["mutates_input_dict_in_place"] = op_no_plane.get("plane") == _DEFAULT_PLANE_XY
    at1g["opstep_data_carries_default"] = step_n.data.get("plane") == _DEFAULT_PLANE_XY
    at1g["default_matches_brep_xy_convention"] = _DEFAULT_PLANE_XY == {
        "origin": [0.0, 0.0, 0.0], "x_dir": [1.0, 0.0, 0.0], "z_dir": [0.0, 0.0, 1.0]}
    custom_plane = {"origin": [10.0, 0.0, 0.0], "x_dir": [0.0, 1.0, 0.0], "z_dir": [1.0, 0.0, 0.0]}
    op_explicit = {"op": "sketch_2d", "id": "skn2", "plane": dict(custom_plane),
                   "shapes": [{"type": "rectangle", "width": 5, "height": 5, "mode": "add"}]}
    validate_op(op_explicit)
    at1g["explicit_plane_untouched"] = op_explicit["plane"] == custom_plane
    ex_skp = dict(examples["sketch_profile"])
    ex_skp.pop("plane", None)
    step_skp = validate_op(ex_skp)
    at1g["sketch_profile_also_normalized"] = step_skp.data.get("plane") == _DEFAULT_PLANE_XY
    # non-plane-bearing op-types (e.g. extrude) never gain a spurious 'plane' key.
    op_extrude = dict(examples["extrude"])
    validate_op(op_extrude)
    at1g["non_plane_op_untouched"] = "plane" not in op_extrude
    # end-to-end through validate_recipe: the SAME list object callers feed into exec_ops afterward
    # (formrib_v1.py/formfeature_v1.py idiom: `validate_recipe({"ops": ops}); exec_ops(ops, ...)`) must
    # observe the normalized plane post-gate.
    ops_list = [dict(op_no_plane_raw) for op_no_plane_raw in [
        {"op": "sketch_2d", "id": "skr1", "shapes": [{"type": "rectangle", "width": 5, "height": 5, "mode": "add"}]},
    ]]
    validate_recipe({"ops": ops_list})
    at1g["validate_recipe_normalizes_same_list_object"] = ops_list[0].get("plane") == _DEFAULT_PLANE_XY
    at1g["pass"] = bool(
        at1g["mutates_input_dict_in_place"] and at1g["opstep_data_carries_default"]
        and at1g["default_matches_brep_xy_convention"] and at1g["explicit_plane_untouched"]
        and at1g["sketch_profile_also_normalized"] and at1g["non_plane_op_untouched"]
        and at1g["validate_recipe_normalizes_same_list_object"]
    )
    results["AT1g_plane_normalization"] = at1g

    results["all_pass"] = bool(
        results["AT1a"]["pass"] and results["AT1b"]["pass"] and results["AT1c"]["pass"] and results["AT1d"]["pass"]
        and results["recipe_roundtrip"]["pass"] and results["duplicate_id_rejected"]["pass"]
        and results["AT1e_radius_frac_xor"]["pass"] and results["AT1f_params_block"]["pass"]
        and results["AT1g_plane_normalization"]["pass"]
    )
    return results


if __name__ == "__main__":
    out = _selftest()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["all_pass"] else 1)
