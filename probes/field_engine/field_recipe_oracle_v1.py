"""Sampled field-to-recipe-to-B-rep regression oracle with planted shape errors.

Equality is on the declared lattice, not continuous surface identity. Negative
controls replace a chamfer by a square and a twelve-sided profile by a circle
approximation. No CAD baseline or recipe executor is modified.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, Point
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_IN, TopAbs_ON
from OCP.gp import gp_Pnt
from field_to_recipe_loft_v1 import recipe_from_field, validate_recipe, exec_ops

ROOT = Path(__file__).resolve().parents[2]


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def distance(mask, pitch):
    """Frozen half-pitch corrected grid EDT convention."""
    sd = (ndimage.distance_transform_edt(~mask, sampling=(pitch,) * 3).astype(np.float32)
          - ndimage.distance_transform_edt(mask, sampling=(pitch,) * 3).astype(np.float32))
    return (sd - np.sign(sd) * np.float32(.5 * pitch)).astype(np.float32)


def sample_recipe(recipe, points, shape):
    validate_recipe(copy.deepcopy(recipe))
    result = exec_ops(copy.deepcopy(recipe["ops"]))
    body = result["solids"]["rebuilt"]
    if not all(row["status"] == "PASS" for row in result["log"]):
        raise ValueError("Recipe execution failed")
    if not body.is_valid or len(body.solids()) != 1:
        raise ValueError("Oracle requires one valid solid")
    classifier = BRepClass3d_SolidClassifier(body.solids()[0].wrapped)
    values = np.empty(len(points), dtype=bool)
    for i, point in enumerate(points):
        classifier.Perform(gp_Pnt(*map(float, point)), 1e-8)
        values[i] = classifier.State() in (TopAbs_IN, TopAbs_ON)
    return values.reshape(shape)


def compare_fields(reference, rebuilt, pitch):
    if reference.shape != rebuilt.shape or reference.dtype != bool or rebuilt.dtype != bool:
        raise ValueError("Aligned boolean fields required")
    changed = reference != rebuilt
    return dict(occupancy_differences=int(changed.sum()),
                distance_differences=int(np.count_nonzero(distance(reference, pitch) !=
                                                         distance(rebuilt, pitch))),
                per_section_differences=changed.sum(axis=(0, 1)).tolist(),
                reference_sha256=digest(reference), rebuilt_sha256=digest(rebuilt))


def fixtures():
    angles = np.arange(12) * (2 * np.pi / 12)
    return dict(square=Polygon([(-4, -4), (4, -4), (4, 4), (-4, 4)]),
                chamfer=Polygon([(-4, -4), (4, -4), (4, 2), (2, 4), (-4, 4)]),
                twelve_gon=Polygon(np.column_stack((4 * np.cos(angles), 4 * np.sin(angles)))),
                through_hole=Polygon([(-4, -4), (4, -4), (4, 4), (-4, 4)],
                                     holes=[[(-1, -1), (-1, 1), (1, 1), (1, -1)]]))


def replace_profiles(recipe, coords):
    mutated = copy.deepcopy(recipe)
    for op in mutated["ops"]:
        if op["op"] == "sketch_profile":
            op["points"] = {f"p{i}": dict(x=float(x), y=float(y), fixed=True)
                            for i, (x, y) in enumerate(coords)}
            op["profile"]["point_order"] = list(op["points"])
    return mutated


def main():
    pitch = .2
    origin = np.array([-4.713, -4.729, -.2])
    shape = (48, 48, 5)
    indices = np.indices(shape).reshape(3, -1).T
    points = origin + indices * pitch
    xy = origin[:2] + np.indices(shape[:2]).reshape(2, -1).T * pitch
    arrays = dict(points=points)
    rows = []
    for name, polygon in fixtures().items():
        section = np.array([polygon.contains(Point(p)) for p in xy]).reshape(shape[:2])
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, 1:-1] = section[:, :, None]
        implicit = np.where(mask, 1., -1.)
        legs = []
        for leg in range(2):
            recipe, _, _ = recipe_from_field(implicit, pitch, origin)
            encoded = json.dumps(recipe, sort_keys=True).encode()
            rebuilt = sample_recipe(recipe, points, shape)
            arrays[f"{name}_{leg}_rebuilt"] = rebuilt
            row = compare_fields(mask, rebuilt, pitch)
            row["recipe_sha256"] = hashlib.sha256(encoded).hexdigest()
            legs.append(row)
        arrays[name + "_reference"] = mask
        negative = None
        if name in ("chamfer", "twelve_gon"):
            if name == "chamfer":
                coords = [(-4, -4), (4, -4), (4, 4), (-4, 4)]
            else:
                theta = np.arange(96) * (2 * np.pi / 96)
                coords = np.column_stack((4 * np.cos(theta), 4 * np.sin(theta)))
            bad_recipe = replace_profiles(recipe, coords)
            bad = [sample_recipe(bad_recipe, points, shape) for _ in range(2)]
            negative = compare_fields(mask, bad[0], pitch)
            negative["repeat"] = bad[0].tobytes() == bad[1].tobytes()
            arrays[name + "_mutated"] = bad[0]
        rows.append(dict(feature=name, legs=legs, negative=negative))
        print(json.dumps(rows[-1]), flush=True)
    gates = dict(zero_round_trip=all(r["legs"][0]["occupancy_differences"] == 0 and
                                    r["legs"][0]["distance_differences"] == 0 for r in rows),
                 full_repeat=all(r["legs"][0] == r["legs"][1] for r in rows),
                 planted_errors_detected=all(r["negative"]["occupancy_differences"] > 0 and
                                             r["negative"]["distance_differences"] > 0 and
                                             r["negative"]["repeat"] for r in rows
                                             if r["negative"] is not None))
    report = dict(status="VERIFIED-FRESH" if all(gates.values()) else "OWN-GATE-FAIL",
                  scope="Four synthetic extruded sections, occupancy and derived grid EDT only.",
                  pitch=pitch, origin=origin.tolist(), shape=list(shape), rows=rows, gates=gates)
    destination = ROOT / "reports"
    destination.mkdir(exist_ok=True)
    (destination / "field_recipe_oracle.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(destination / "field_recipe_oracle_arrays.npz", **arrays)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
