"""Circular copies must rotate about the declared axis, including its origin."""
import copy
import math
from pathlib import Path
import sys
import unittest

import build123d as bd
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "field_engine" / "recipe"))
from cad_op_exec_v1 import _handle_pattern_circular, exec_ops  # noqa: E402

CHECK = unittest.TestCase()


def properties(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    center = props.CentreOfMass()
    return props.Mass(), (center.X(), center.Y(), center.Z())


def assert_same_solid(actual, expected):
    volume, center = properties(expected)
    actual_volume, actual_center = properties(actual)
    CHECK.assertAlmostEqual(actual_volume, volume, delta=1e-8)
    for actual_coordinate, expected_coordinate in zip(actual_center, center):
        CHECK.assertAlmostEqual(actual_coordinate, expected_coordinate, delta=1e-8)
    # Matching mass and centroid alone do not establish matching geometry.
    common = BRepAlgoAPI_Common(actual, expected)
    common.SetRunParallel(False)
    common.Build()
    assert common.IsDone()
    CHECK.assertAlmostEqual(properties(common.Shape())[0], volume, delta=1e-7)


def assert_rotated_copies(result, seed, origin, direction, count, angle):
    step = angle / count if angle == 360 else angle / (count - 1) if count > 1 else 0
    expected = []
    for index in range(count):
        transform = gp_Trsf()
        transform.SetRotation(
            gp_Ax1(gp_Pnt(*origin), gp_Dir(*direction)), math.radians(step * index)
        )
        for solid in seed.solids():
            expected.append(BRepBuilderAPI_Transform(solid.wrapped, transform, True).Shape())
    actual = list(result.solids())
    assert len(actual) == len(expected)
    for solid, reference in zip(actual, expected):
        assert_same_solid(solid.wrapped, reference)


def check_axis_origin_is_a_pivot(origin, direction, count, angle, multiple_solids):
    seed = bd.Pos(12, 0, 5) * bd.Box(1, 2, 3)
    if multiple_solids:
        seed = bd.Compound(children=[seed, bd.Pos(16, 1, 7) * bd.Box(2, 3, 1)])
    before = [properties(solid.wrapped) for solid in seed.solids()]
    context = {"seed": seed}
    result = _handle_pattern_circular({
        "target_ref": "seed", "axis": {"origin": origin, "direction": direction},
        "count": count, "angle_deg": angle,
    }, context, {})
    assert_rotated_copies(result, seed, origin, direction, count, angle)
    assert list(context) == ["seed"] and context["seed"] is seed
    assert [properties(solid.wrapped) for solid in seed.solids()] == before


def public_recipe(axis, count=3, angle=180):
    return [
        {"op": "sketch_2d", "id": "profile", "plane": {"origin": [12, 0, 5]},
         "shapes": [{"type": "rectangle", "width": 1, "height": 2}]},
        {"op": "extrude", "id": "seed", "sketch_ref": "profile", "amount": 3},
        {"op": "pattern_circular", "id": "copies", "target_ref": "seed",
         "axis": axis, "count": count, "angle_deg": angle},
    ]


def check_public_parameter_edit_and_restoration():
    operations = public_recipe(
        {"origin": [{"expr": "pivot_x"}, 0, 5], "direction": [0, 0, 1]},
        angle={"expr": "sweep"},
    )
    original = copy.deepcopy(operations)
    snapshots = []
    for pivot_x, sweep in [(10, 180), (11, 90), (10, 180)]:
        result = exec_ops(operations, params={"pivot_x": pivot_x, "sweep": sweep})
        seed, copies = result["solids"]["seed"], result["solids"]["copies"]
        assert_rotated_copies(copies, seed, (pivot_x, 0, 5), (0, 0, 1), 3, sweep)
        snapshots.append([properties(solid.wrapped) for solid in copies.solids()])
        assert operations == original
    assert snapshots[0] == snapshots[2]
    assert snapshots[0] != snapshots[1]


def check_world_axis_and_dictionary_defaults(axis, direction):
    operations = public_recipe(axis, angle=360)
    del operations[-1]["angle_deg"]
    result = exec_ops(operations)
    assert_rotated_copies(
        result["solids"]["copies"], result["solids"]["seed"], (0, 0, 0), direction, 3, 360
    )


class CircularPatternTests(unittest.TestCase):
    def test_axis_origin_is_a_pivot(self):
        for origin in [(0, 0, 0), (10, -2, 5)]:
            for direction, count, angle in [
                ((0, 0, 1), 1, 180), ((0, 0, 1), 3, 180),
                ((0, 1, 0), 3, 360), ((1, 2, 3), 2, 73),
            ]:
                for multiple_solids in [False, True]:
                    with self.subTest(origin=origin, direction=direction, count=count,
                                      angle=angle, multiple_solids=multiple_solids):
                        check_axis_origin_is_a_pivot(origin, direction, count, angle, multiple_solids)

    def test_public_parameter_edit_and_restoration(self):
        check_public_parameter_edit_and_restoration()

    def test_world_axis_and_dictionary_defaults(self):
        for axis, direction in [
            ("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1)), ({}, (0, 0, 1)),
        ]:
            with self.subTest(axis=axis):
                check_world_axis_and_dictionary_defaults(axis, direction)


if __name__ == "__main__":
    unittest.main()
