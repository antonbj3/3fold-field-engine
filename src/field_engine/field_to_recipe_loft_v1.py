"""Section-loft recipe for a measured field, beside the frozen primitive baseline.

A piecewise constant section reconstruction handles splits/merges between slabs.
Every solid is built by the existing op executor. It is not a smooth loft fit.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import Polygon
from skimage.measure import find_contours
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "recipe"))
from cad_op_schema_v1 import validate_recipe
from cad_op_exec_v1 import exec_ops


def recipe_from_field(f, pitch, origin):
    ops, pieces, area_volume, simplification = [], [], 0., []
    origin = np.asarray(origin)
    for z in range(1, f.shape[2]-1):
        rings = [Polygon(c[:-1] * pitch + origin[:2]) for c in find_contours(f[:, :, z], 0)]
        if not all(p.is_valid and not p.is_empty for p in rings):
            raise ValueError("Invalid section polygon")
        rings.sort(key=lambda p: (-p.area, p.centroid.x, p.centroid.y))
        parents = [next((i for i in range(q-1, -1, -1) if rings[i].contains(rings[q])), None)
                   for q in range(len(rings))]
        depth = [0]*len(rings)
        for q, parent in enumerate(parents):
            if parent is not None:
                depth[q] = depth[parent]+1
        simple = [p.simplify(pitch/8, preserve_topology=True) for p in rings]
        for q, poly in enumerate(simple):
            distance = poly.hausdorff_distance(rings[q])
            simplification.append(distance)
            if not poly.is_valid or distance > pitch/8:
                raise ValueError("Contour simplification exceeded the registered bound")
            for i in range(q):
                if poly.within(simple[i]) != rings[q].within(rings[i]):
                    raise ValueError("Contour containment hierarchy changed")
        refs = []
        for q, poly in enumerate(simple):
            coords = np.asarray(poly.exterior.coords[:-1])
            prefix = f"section_{z}_loop_{q}"
            sketches = []
            for side in (0, 1):
                name = prefix + f"_face_{side}"
                sketches.append(name)
                points = {f"p{i}": {"x": float(x), "y": float(y), "fixed": True}
                          for i, (x,y) in enumerate(coords)}
                ops.append({"op": "sketch_profile", "id": name, "points": points, "constraints": [],
                            "profile": {"type": "polygon", "point_order": list(points)},
                            "plane": {"origin": [0,0,float(origin[2]+(z-.5+side)*pitch)],
                                      "x_dir": [1,0,0], "z_dir": [0,0,1]}})
            ops.append({"op": "loft", "id": prefix, "sketch_refs": sketches, "ruled": True})
            refs.append(prefix)
            area_volume += (-1 if depth[q]%2 else 1)*poly.area*pitch
        for q, ref in enumerate(refs):
            if depth[q]%2:
                continue
            holes = [refs[i] for i in range(len(refs)) if parents[i] == q]
            if holes:
                cut = ref+"_cut"
                ops.append({"op": "boolean_cut", "id": cut, "base_ref": ref, "tool_refs": holes})
                pieces.append(cut)
            else:
                pieces.append(ref)
    if not pieces:
        raise ValueError("Field has no material sections")
    ops.append({"op": "boolean_union", "id": "rebuilt", "base_ref": pieces[0], "tool_refs": pieces[1:]})
    return {"ops": ops}, area_volume, max(simplification, default=0.)


def main():
    dest = HERE / "artifacts"
    reference = json.loads((dest / "field_recipe_sections_probe.json").read_text())
    f = np.load(dest / "field_recipe_sections_probe.npy")
    if hashlib.sha256(f.tobytes()).hexdigest() != reference["field_sha256"]:
        raise ValueError("Field digest disagrees with measured reference")
    recipe, expected, deviation = recipe_from_field(f, reference["pitch"], reference["origin"])
    validate_recipe(copy.deepcopy(recipe))
    encoded = json.dumps(recipe, sort_keys=True, indent=2)+"\n"
    (dest / "field_to_recipe_loft_v1_recipe.json").write_text(encoded)
    result = exec_ops(copy.deepcopy(recipe["ops"]))
    solid = result["solids"]["rebuilt"]
    volume = float(solid.volume)
    error = abs(volume-reference["volume"])/reference["volume"]
    integral_error = abs(volume-expected)/expected
    gates = {"executor": all(r["status"] == "PASS" for r in result["log"]),
             "valid": bool(solid.is_valid), "one_solid": len(solid.solids()) == 1,
             "rasterisation_volume": error < reference["rasterisation_bound"],
             "polygon_integral": integral_error < 1e-8,
             "simplification": deviation <= reference["pitch"]/8}
    report = {"status": "SYNTHETIC-ONLY", "volume": volume, "reference_volume": reference["volume"],
              "relative_volume_error": error, "rasterisation_bound": reference["rasterisation_bound"],
              "polygon_integral_volume": expected, "polygon_integral_relative_error": integral_error,
              "max_simplification_distance": deviation, "ops": len(recipe["ops"]),
              "lofts": sum(op["op"] == "loft" for op in recipe["ops"]),
              "recipe_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "gates": gates,
              "pass": all(gates.values())}
    (dest / "field_to_recipe_loft_v1.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
