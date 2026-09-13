"""brep_recept_synth_v1.py -- turn a recognised BREP into a recipe that rebuilds it.

Reads a solid, runs the attributed adjacency graph (brep_aag_v1) and the feature recogniser
(brep_feature_igenkann_v1) over it, and emits a recipe of ops from cad_op_schema_v1.OP_SPECS: a stock
block plus one op per recognised hole, pocket, fillet and chamfer. The emitted recipe is validated
with cad_op_schema_v1.validate_recipe and executed with cad_op_exec_v1.exec_ops, and the rebuilt
solid's volume and face count are compared with the input's, so the synthesis is measured rather than
asserted.

API:
    synth(shape, rec=None, aag=None) -> {"ops": [...], "params": {...}, ...}
    roundtrip(step_path, param_overrides=None) -> {"PASS": bool, "vol_dev_pct": float,
        "orig": {...}, "rebuilt": {...}}  -- rebuild the STEP file from its own synthesised recipe
        and compare volume, face count and face mix

Base profiles are synthesised for a circle, an annulus and a right-angled rectangle; any other outer
contour raises UnsupportedProfile rather than being approximated. Holes, chamfers and fillets are
emitted for every recognised instance regardless of profile type.

Run it as a script with one or more STEP file paths, or with `--selftest` to build a synthetic
bracket, round-trip it and exit non-zero if the comparison fails.
"""
from __future__ import annotations

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))  # the recognizer and the adjacency graph live one level up

import build123d as bd  # noqa: E402
import cad_op_exec_v1 as ex  # noqa: E402
from brep_aag_v1 import _add, _cross, _dot, _norm, _scale, _sub, build_aag  # noqa: E402
from brep_feature_igenkann_v1 import recognize  # noqa: E402
from cad_op_schema_v1 import validate_recipe  # noqa: E402

AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


class UnsupportedProfile(NotImplementedError):
    pass


def _axis_name(d):
    for k, v in AXES.items():
        if abs(abs(_dot(_norm(d), v)) - 1.0) < 1e-6:
            return k
    return None


def _cap_faces(aag, d):
    """Planar faces perpendicular to d, grouped by their position along d."""
    out = []
    for f in aag.faces:
        if f["type"] != "PLANE":
            continue
        if abs(abs(_dot(_norm(f["normal"]), _norm(d))) - 1.0) < 1e-6:
            out.append((round(_dot(f["sample_point"], _norm(d)), 6), f))
    return sorted(out, key=lambda t: t[0])


def _profile_of(aag, face, d, blend_idx: set):
    """The base profile is taken from the body's LATERAL faces, never from an end face.

    Two measured errors share the same cause and this function is written to avoid both: (1) a chamfer
    on a cylinder's rim edge shrinks the end plane's radius from 48 to 45, so the cap face's circle is
    not the base body's (12.88 % volume error); (2) a plate's cap face measured 14 mm corner arcs and a
    256 mm bounding box, because an r=2 fillet on its line edges had eroded both the corner radius
    (16 -> 14) and the outer contour (260 -> 256). Post-processing (fillets and chamfers) always eats
    into the end face, while the lateral skin (side planes whose normal is perpendicular to the pull
    direction, and corner cylinders parallel to it) carries the profile exactly.
    """
    lat_planes, lat_cyls = [], []
    for g in aag.faces:
        if g["type"] == "PLANE" and g["idx"] not in blend_idx \
                and abs(_dot(_norm(g["normal"]), _norm(d))) < 1e-6:
            lat_planes.append(g)
        elif g["type"] == "CYLINDER" and g["concavity"] == "OUTWARD" \
                and abs(abs(_dot(_norm(g["axis_dir"]), _norm(d))) - 1.0) < 1e-6:
            # Corner cylinders are TANGENT to the side planes and are therefore classified as
            # "blend" by the recogniser -- they must NOT be filtered out here. A corner radius
            # PARALLEL to the pull direction belongs to the PROFILE: it is the same geometry whether
            # it came from a rounded-rectangle sketch or from a corner fillet applied afterwards (the
            # declared equivalence class). Blends that are NOT parallel to d (edge fillets on the end
            # faces) are removed by the parallelism test above.
            lat_cyls.append(g)
    an = _axis_name(d)
    keep = [i for i, k in enumerate("XYZ") if k != an]

    # (a) PLAIN CIRCLE: a single lateral cylinder sweeping the whole revolution
    if not lat_planes and len(lat_cyls) == 1 and (lat_cyls[0].get("u_span") or 0) > 6.2:
        g = lat_cyls[0]
        return ([{"type": "circle", "radius": {"expr": "r_outer"}, "center": [0.0, 0.0]}],
                {"r_outer": g["radius"]}, list(g["axis_point"]))

    if len(lat_planes) < 4:
        raise UnsupportedProfile(
            f"lateral skin: {len(lat_planes)} side planes + {len(lat_cyls)} corner cylinders "
            f"(supported: 1 full cylinder, 4 side planes, or 4 side planes + 4 equal corner cylinders)")
    # (b/c) RECTANGLE / ROUNDED RECTANGLE. The frame comes from the side planes' OWN normals, not
    # from the world axes: a part need not be axis-aligned in the STEP file (measured: 3 of 32
    # benchmark parts failed on that requirement alone). sketch_2d's plane already takes a free
    # x_dir, so the schema always allowed this; only the profile classifier was world-bound.
    e1 = _norm(aag.faces[lat_planes[0]["idx"]]["normal"])
    e1 = _norm(_sub(e1, _scale(_norm(d), _dot(e1, _norm(d)))))
    e2 = _norm(_cross(_norm(d), e1))
    offs = {0: [], 1: []}
    for g in lat_planes:
        n = _norm(g["normal"])
        c1, c2 = _dot(n, e1), _dot(n, e2)
        if abs(abs(c1) - 1.0) < 1e-6:
            offs[0].append(_dot(g["sample_point"], n) * (1.0 if c1 > 0 else -1.0))
        elif abs(abs(c2) - 1.0) < 1e-6:
            offs[1].append(_dot(g["sample_point"], n) * (1.0 if c2 > 0 else -1.0))
        else:
            raise UnsupportedProfile(
                "a side plane that is neither parallel nor perpendicular to the profile's first side "
                "plane (supported: rectangular and rounded-rectangular contours)")
    if len(offs[0]) < 2 or len(offs[1]) < 2:
        raise UnsupportedProfile(f"the side planes do not cover both transverse directions: "
                                 f"{len(offs[0])}/{len(offs[1])}")
    w = max(offs[0]) - min(offs[0])
    h = max(offs[1]) - min(offs[1])
    m1 = 0.5 * (max(offs[0]) + min(offs[0]))
    m2 = 0.5 * (max(offs[1]) + min(offs[1]))
    cen = _add(_scale(e1, m1), _scale(e2, m2))
    params = {"w": w, "h": h}
    if lat_cyls:
        rr = {round(g["radius"], 6) for g in lat_cyls}
        if len(rr) != 1 or len(lat_cyls) != 4:
            raise UnsupportedProfile(
                f"{len(lat_cyls)} corner cylinders with radii {sorted(rr)} (supported: 4 equal)")
        params["r_corner"] = rr.pop()
    params["_x_dir"] = list(e1)
    return ([{"type": "rectangle", "width": {"expr": "w"}, "height": {"expr": "h"},
              "center": [0.0, 0.0]}], params, cen)


def synth(shape, rec: dict | None = None, aag=None) -> dict:
    aag = aag or build_aag(shape)
    rec = rec or recognize(shape, aag)
    feats = rec["features"]
    base = [f for f in feats if f["type"] == "extrude_body"]
    if not base:
        raise UnsupportedProfile("no extrude_body direction -- the base verb cannot be chosen")
    d = _norm(base[0]["direction"])
    an = _axis_name(d)
    if an is None:
        raise UnsupportedProfile(f"extrusion direction {d} is not a world axis (supported: X|Y|Z)")
    # The sign must be canonicalised (measured: a part was built in z=30..90 instead of -30..30 and
    # its hole cut only half the height, a 2.96 % volume error). The axis candidate may as well come
    # back as -Z as +Z, while the sketch plane's z_dir was set to the POSITIVE world axis, so the
    # extrusion ran opposite to the t0 measured along the negative direction.
    d = AXES[an]
    blend_idx = {i for f in feats if f["type"] in ("fillet", "chamfer") for i in f["faces"]}
    caps = _cap_faces(aag, d)
    if len(caps) < 2:
        raise UnsupportedProfile("did not find two end faces perpendicular to the extrusion direction")
    t0, f0 = caps[0]
    t1 = caps[-1][0]
    shapes, params, cen = _profile_of(aag, f0, d, blend_idx)

    # inner contour (annulus): an INWARD cylinder coaxial with the profile, running through both ends
    holes = [f for f in feats if f["type"] in ("hole_through", "hole_blind")]
    inner = None
    for h in holes:
        if h["type"] == "hole_through" and abs(abs(_dot(_norm(h["axis_dir"]), d)) - 1.0) < 1e-6:
            off = _sub(h["axis_point"], cen)
            if math.dist(_scale(_norm(d), _dot(off, _norm(d))), off) < 1e-6 and \
                    shapes[0]["type"] == "circle":
                inner = h
                break
    ops = []
    if inner is not None:
        params["r_inner"] = inner["radius"]
        shapes.append({"type": "circle", "radius": {"expr": "r_inner"},
                       "center": [0.0, 0.0], "mode": "subtract"})
        holes = [h for h in holes if h is not inner]

    params["height"] = t1 - t0
    origin = list(_scale(_norm(d), t0))
    for i, k in enumerate("XYZ"):
        if k != an:
            origin[i] = cen[i]
    del i, k
    x_dir = params.pop("_x_dir", None) or list(AXES["X"] if an != "X" else AXES["Y"])
    plane = {"origin": [round(x, 9) for x in origin],
             "x_dir": [round(x, 12) for x in x_dir],
             "z_dir": list(AXES[an])}
    ops.append({"op": "sketch_2d", "id": "sk_base", "plane": plane, "shapes": shapes})
    ops.append({"op": "extrude", "id": "base", "sketch_ref": "sk_base",
                "amount": {"expr": "height"}})
    prev = "base"

    if "r_corner" in params:
        ops.append({"op": "fillet", "id": "corner", "target_ref": prev,
                    "radius": {"expr": "r_corner"},
                    "selector": {"entity": "edge", "geometry_type": "LINE",
                                 "direction": list(AXES[an]), "expected_count": 4}})
        prev = "corner"

    # chamfer BEFORE holes (the build order in the emitted recipe: chamfer on the base body's
    # circular edges)
    chamfers = [f for f in feats if f["type"] == "chamfer" and f.get("surface") == "CONE"]
    if chamfers:
        dist = sum(f["axial_extent"] for f in chamfers) / len(chamfers)
        params["chamfer_d"] = dist
        # pin the radius: an annulus base has 4 circular edges (outer and inner), but the chamfer
        # sits only on the outer ones, and the recogniser knows the radius of the cone's large end.
        r_big = max(max(aag.faces[i]["ref_radius"] for i in f["faces"]) for f in chamfers)
        # The selector's radius must be SYMBOLIC, not a frozen number (measured: with
        # chamfer_edge_r as the literal 48.0 the selector refused with REFERENCE_LOST as soon as
        # r_outer changed to 60 -- the recipe was parametric in its geometry but not in its own
        # reference). `r_outer` is referenced directly, so the chamfer follows the diameter.
        r_expr = "r_outer" if "r_outer" in params else "chamfer_edge_r"
        if r_expr == "chamfer_edge_r":
            params["chamfer_edge_r"] = round(r_big + dist, 6)
        ops.append({"op": "chamfer", "id": "cham", "target_ref": prev,
                    "distance": {"expr": "chamfer_d"},
                    "selector": {"entity": "edge", "geometry_type": "CIRCLE",
                                 "radius": {"expr": r_expr}, "radius_tol_mm": 0.01,
                                 "expected_count": len(chamfers)}})
        prev = "cham"

    # planar chamfers (a band breaking the edge between two planes). One op per band, addressed by
    # the MEASURED midpoint of the edge it broke (brep_feature_igenkann_v1._planar_chamfer_geometry),
    # with expected_count 1 -- the same "nearest, with a decisive margin" contract the selector
    # already enforces. A band without a measured distance (asymmetric legs) is not emitted; it
    # would have to be guessed.
    plane_chamfers = [f for f in feats if f["type"] == "chamfer" and f.get("surface") == "PLANE"
                      and f.get("distance") is not None]
    for n, f in enumerate(plane_chamfers):
        params[f"cham_pl{n}_d"] = f["distance"]
        ops.append({"op": "chamfer", "id": f"cham_pl{n}", "target_ref": prev,
                    "distance": {"expr": f"cham_pl{n}_d"},
                    "selector": {"entity": "edge", "geometry_type": "LINE",
                                 "near_point": [round(c, 9) for c in f["edge_point"]],
                                 "expected_count": 1}})
        prev = f"cham_pl{n}"

    # fillet: post-processing. A band the recogniser LOCATED (it rounded one edge between two
    # planes, so brep_feature_igenkann_v1._edge_fillet_geometry measured that edge's midpoint) gets
    # its own op addressed by that midpoint, exactly as a planar chamfer does -- `near_point` with
    # expected_count 1, the selector's "nearest, with a decisive margin" contract, which refuses
    # instead of guessing on a tie. Bands without a location keep the type-and-count selector over
    # all LINE edges, grouped by radius.
    fillets = [f for f in feats if f["type"] == "fillet" and f.get("radius") is not None
               and abs(f["radius"] - params.get("r_corner", -1)) > 1e-9]
    located = [f for f in fillets if f.get("edge_point") is not None]
    for n, f in enumerate(located):
        params[f"fil_pt{n}_r"] = f["radius"]
        ops.append({"op": "fillet", "id": f"fil_pt{n}", "target_ref": prev,
                    "radius": {"expr": f"fil_pt{n}_r"},
                    "selector": {"entity": "edge", "geometry_type": "LINE",
                                 "near_point": [round(c, 9) for c in f["edge_point"]],
                                 "expected_count": 1}})
        prev = f"fil_pt{n}"
    frad = sorted({round(f["radius"], 6) for f in fillets if f.get("edge_point") is None})
    for n, r in enumerate(frad):
        cnt = sum(1 for f in fillets if f.get("edge_point") is None and abs(f["radius"] - r) < 1e-9)
        params[f"fillet{n}_r"] = r
        ops.append({"op": "fillet", "id": f"fil{n}", "target_ref": prev,
                    "radius": {"expr": f"fillet{n}_r"},
                    "selector": {"entity": "edge", "geometry_type": "LINE",
                                 "expected_count": cnt}})
        prev = f"fil{n}"

    for n, h in enumerate(holes):
        han = _axis_name(h["axis_dir"])
        if han is None:
            raise UnsupportedProfile(f"hole axis {h['axis_dir']} is not a world axis")
        params[f"hole{n}_d"] = h["diameter"]
        op = {"op": "hole", "id": f"hole{n}", "target_ref": prev,
              "center": [round(c, 9) for c in h["center"]],
              "diameter": {"expr": f"hole{n}_d"}, "axis": han}
        if h["type"] == "hole_through":
            op["through"] = True
        else:
            params[f"hole{n}_depth"] = h["depth"]
            op["depth"] = {"expr": f"hole{n}_depth"}
        ops.append(op)
        prev = f"hole{n}"

    _measure_expected_counts(ops, params)
    return {"params": params, "ops": ops, "result_ref": prev}


def _measure_expected_counts(ops: list, params: dict) -> None:
    """Fills each selector's expected_count with the population actually MEASURED at that chain position.
    Measured reason: counting the recogniser's FACES (16 blend faces) while the selector counts EDGES on
    the partially built body (12) is the wrong quantity, and the selector correctly refused with
    POPULATION_DRIFT. A recipe gets its expected_count from a measurement, never from a guess.
    """
    import geometri_selektor_v1 as gs
    for k, op in enumerate(ops):
        sel = op.get("selector")
        if not sel:
            continue
        probe = dict(sel)
        # A near_point selector RANKS by distance and only truncates when expected_count is set, so
        # probing it with 0 would measure the whole unranked population instead of the nearest ones.
        probe["expected_count"] = sel.get("expected_count", 0) if "near_point" in sel else 0
        try:
            res = ex.exec_ops(ops[:k], params=params)
            target = res["solids"][op["target_ref"]]
        except Exception:
            continue
        # resolve any {"expr": ...} in the selector before measuring the population
        rs = {}
        for kk, vv in probe.items():
            rs[kk] = ex.resolve_expr(vv["expr"], params) if isinstance(vv, dict) and set(vv) == {"expr"} else vv
        try:
            pop = gs.geometric_edge_population(target, rs)
        except Exception:
            continue
        sel["expected_count"] = len(pop)


# ---------------------------------------------------------------- gate
def _face_mix(shape) -> dict:
    from collections import Counter
    a = build_aag(shape)
    return dict(Counter(f["type"] for f in a.faces))


def roundtrip(step_path: str, param_overrides: dict | None = None) -> dict:
    orig = bd.import_step(step_path)
    sols = orig.solids()
    if len(sols) > 1:
        orig = sols[0]
    aag = build_aag(orig)
    rec = recognize(orig, aag)
    recipe = synth(orig, rec, aag)
    validate_recipe({"ops": recipe["ops"]})
    params = dict(recipe["params"])
    params.update(param_overrides or {})
    res = ex.exec_ops(recipe["ops"], params=params)
    built = res["solids"][recipe["result_ref"]]
    out = {
        "step": step_path, "recipe": recipe,
        "params_used": params,
        "orig": {"volume": float(orig.volume), "n_faces": len(aag.faces),
                 "face_mix": _face_mix(orig)},
        "rebuilt": {"volume": float(built.volume), "n_faces": len(build_aag(built).faces),
                    "face_mix": _face_mix(built)},
    }
    o, b = out["orig"], out["rebuilt"]
    out["vol_dev_pct"] = round(100.0 * abs(b["volume"] - o["volume"]) / max(o["volume"], 1e-9), 6)
    out["face_count_match"] = (o["n_faces"] == b["n_faces"])
    out["face_mix_match"] = (o["face_mix"] == b["face_mix"])
    out["PASS"] = bool(out["vol_dev_pct"] < 0.01 and out["face_count_match"] and out["face_mix_match"])
    return out


def _synthetic_bracket():
    """A declared synthetic part for the selftest: a rectangular plate with four through holes,
    the profile + through-hole classes this module synthesises."""
    with bd.BuildPart() as bp:
        bd.Box(140.0, 90.0, 12.0)
        with bd.Locations(*[(x, y, 0.0) for x in (-55.0, 55.0) for y in (-32.0, 32.0)]):
            bd.Hole(radius=4.0)
    return bp.part


def _chamfered_bracket():
    """The same synthetic plate with a 2 mm chamfer on the four straight top edges: the planar
    chamfer class, on top of the profile + through-hole classes."""
    part = _synthetic_bracket()
    top = part.faces().sort_by(bd.Axis.Z)[-1]
    return bd.chamfer(top.edges().filter_by(bd.GeomType.LINE), CHAMFER_MM)


def _filleted_bracket():
    """The same synthetic plate with a 3 mm fillet on ONE straight top edge: the located-fillet
    class (a cylindrical band rounding a single edge between two planes)."""
    part = _synthetic_bracket()
    top = part.faces().sort_by(bd.Axis.Z)[-1]
    edge = top.edges().filter_by(bd.GeomType.LINE).sort_by(bd.Axis.X)[-1]
    return bd.fillet([edge], FILLET_MM)


def _asym_chamfered_bracket():
    """The same synthetic plate with an UNEQUAL-leg (2 mm x 4 mm) chamfer on one straight top edge.
    The recogniser measures both legs; the schema's chamfer op carries one scalar, so the band is
    not emitted and the rebuild is short by exactly the wedge that was never cut."""
    part = _synthetic_bracket()
    top = part.faces().sort_by(bd.Axis.Z)[-1]
    edge = top.edges().filter_by(bd.GeomType.LINE).sort_by(bd.Axis.X)[-1]
    return bd.chamfer([edge], length=ASYM_LEG_A, length2=ASYM_LEG_B)


def _edge_at(part, x: float, z: float):
    """The single straight edge whose midpoint sits at (x, *, z) -- the fixtures address their edges
    by measured position, so each operation is applied to a known edge and never to a list index."""
    hits = [e for e in part.edges().filter_by(bd.GeomType.LINE)
            if abs(e.center().X - x) < 1e-6 and abs(e.center().Z - z) < 1e-6]
    if len(hits) != 1:
        raise AssertionError(f"fixture edge at x={x}, z={z}: {len(hits)} candidates, expected 1")
    return hits[0]


def _combined_bracket():
    """A plate with two through holes and three treated edges that share no corner: a 2 mm symmetric
    chamfer on the +X top edge, a 2 x 4 asymmetric chamfer on the -X top edge and a 3 mm fillet on
    the +X bottom edge. Non-adjacent edges, so the rebuild order cannot change the geometry at a
    shared corner and the only difference from the input is the band that was not emitted."""
    with bd.BuildPart() as bp:
        bd.Box(140.0, 90.0, 12.0)
        with bd.Locations((-55.0, 0.0, 0.0), (55.0, 0.0, 0.0)):
            bd.Hole(radius=4.0)
    part = bp.part
    part = bd.chamfer([_edge_at(part, 70.0, 6.0)], CHAMFER_MM)
    part = bd.chamfer([_edge_at(part, -70.0, 6.0)], length=ASYM_LEG_A, length2=ASYM_LEG_B)
    return bd.fillet([_edge_at(part, 70.0, -6.0)], FILLET_MM)


CHAMFER_MM = 2.0
FILLET_MM = 3.0
ASYM_LEG_A, ASYM_LEG_B = 2.0, 4.0
ASYM_EDGE_LEN = 90.0
# the material the un-emitted asymmetric band would have removed: a triangular prism, closed form
ASYM_WEDGE_MM3 = 0.5 * ASYM_LEG_A * ASYM_LEG_B * ASYM_EDGE_LEN


def _roundtrip_part(part, name: str) -> dict:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        step_path = os.path.join(tmp, f"{name}.step")
        bd.export_step(part, step_path)
        return roundtrip(step_path)


def _selftest() -> dict:
    plain = _roundtrip_part(_synthetic_bracket(), "synthetic_bracket_v1")
    chamfered = _roundtrip_part(_chamfered_bracket(), "synthetic_bracket_chamfered_v1")
    n_cham = sum(1 for o in chamfered["recipe"]["ops"] if o["op"] == "chamfer")
    chamfered["n_chamfer_ops"] = n_cham
    chamfered["chamfer_distances"] = [v for k, v in chamfered["params_used"].items()
                                      if k.startswith("cham_pl")]
    # the chamfer must be REBUILT, not merely tolerated: four ops, each at the declared distance
    chamfered["PASS"] = bool(chamfered["PASS"] and n_cham == 4
                             and all(abs(d - CHAMFER_MM) < 1e-9
                                     for d in chamfered["chamfer_distances"]))

    # one located fillet: the band must be REBUILT from its own op at the declared radius, addressed
    # by the measured midpoint of the edge it rounded
    filleted = _roundtrip_part(_filleted_bracket(), "synthetic_bracket_filleted_v1")
    fil_ops = [o for o in filleted["recipe"]["ops"] if o["op"] == "fillet"]
    filleted["n_fillet_ops"] = len(fil_ops)
    filleted["fillet_radii"] = [filleted["params_used"][o["radius"]["expr"]] for o in fil_ops]
    filleted["fillet_near_points"] = [o["selector"].get("near_point") for o in fil_ops]
    filleted["PASS"] = bool(filleted["PASS"] and len(fil_ops) == 1
                            and abs(filleted["fillet_radii"][0] - FILLET_MM) < 1e-9
                            and filleted["fillet_near_points"][0] is not None)

    # an asymmetric band: MEASURED (both legs) but NOT emitted -- cad_op_schema_v1's chamfer op
    # carries one scalar (distance XOR radius_frac) and the schema is frozen, so the rebuild is
    # short by exactly the wedge that was never cut. The gate is that closed-form wedge, not a
    # loosened deviation bound: anything else differing would break it.
    asym_part = _asym_chamfered_bracket()
    asym = _roundtrip_part(asym_part, "synthetic_bracket_asym_chamfer_v1")
    asym_feats = [f for f in recognize(asym_part)["features"]
                  if f["type"] == "chamfer" and f.get("symmetric") is False]
    asym["asym_features"] = [{k: f.get(k) for k in
                              ("leg_a", "leg_b", "symmetric", "distance", "bisector_dev_deg")}
                             for f in asym_feats]
    asym["n_chamfer_ops"] = sum(1 for o in asym["recipe"]["ops"] if o["op"] == "chamfer")
    asym["vol_excess_mm3"] = asym["rebuilt"]["volume"] - asym["orig"]["volume"]
    asym["vol_excess_vs_wedge_mm3"] = asym["vol_excess_mm3"] - ASYM_WEDGE_MM3
    asym["PASS"] = bool(
        len(asym_feats) == 1
        and sorted([asym_feats[0]["leg_a"], asym_feats[0]["leg_b"]]) == [ASYM_LEG_A, ASYM_LEG_B]
        and asym_feats[0].get("distance") is None
        and asym["n_chamfer_ops"] == 0
        and abs(asym["vol_excess_vs_wedge_mm3"]) < 1e-6 * ASYM_WEDGE_MM3
        and asym["orig"]["n_faces"] - asym["rebuilt"]["n_faces"] == 1)

    # all three classes on one part
    comb_part = _combined_bracket()
    comb = _roundtrip_part(comb_part, "synthetic_bracket_combined_v1")
    comb_ops = [o["op"] for o in comb["recipe"]["ops"]]
    comb_feats = [f for f in recognize(comb_part)["features"] if f["type"] == "chamfer"]
    comb["op_types"] = comb_ops
    comb["chamfer_legs"] = sorted(
        [sorted([f.get("leg_a"), f.get("leg_b")]) for f in comb_feats])
    comb["vol_excess_mm3"] = comb["rebuilt"]["volume"] - comb["orig"]["volume"]
    comb["vol_excess_vs_wedge_mm3"] = comb["vol_excess_mm3"] - ASYM_WEDGE_MM3
    comb_cham = [o for o in comb["recipe"]["ops"] if o["op"] == "chamfer"]
    comb_fil = [o for o in comb["recipe"]["ops"] if o["op"] == "fillet"]
    comb["PASS"] = bool(
        len(comb_cham) == 1 and len(comb_fil) == 1
        and sum(1 for o in comb_ops if o == "hole") == 2
        and abs(comb["params_used"][comb_cham[0]["distance"]["expr"]] - CHAMFER_MM) < 1e-9
        and abs(comb["params_used"][comb_fil[0]["radius"]["expr"]] - FILLET_MM) < 1e-9
        and comb["chamfer_legs"] == sorted([[CHAMFER_MM, CHAMFER_MM], [ASYM_LEG_A, ASYM_LEG_B]])
        and abs(comb["vol_excess_vs_wedge_mm3"]) < 1e-6 * ASYM_WEDGE_MM3
        and comb["orig"]["n_faces"] - comb["rebuilt"]["n_faces"] == 1)

    return {"plate_with_holes": plain, "plate_with_chamfer": chamfered,
            "plate_with_fillet": filleted, "plate_with_asym_chamfer": asym,
            "plate_combined": comb,
            "PASS": bool(plain["PASS"] and chamfered["PASS"] and filleted["PASS"]
                         and asym["PASS"] and comb["PASS"])}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        r = _selftest()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["PASS"] else 1)
    for p in sys.argv[1:]:
        try:
            r = roundtrip(p)
            print(os.path.basename(p), "PASS" if r["PASS"] else "FAIL",
                  f"dv={r['vol_dev_pct']}%", r["orig"]["face_mix"], "->", r["rebuilt"]["face_mix"])
        except Exception as e:
            print(os.path.basename(p), "ESCALATE", repr(e)[:160])
