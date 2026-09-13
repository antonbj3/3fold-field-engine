"""Exact bracket queries and storage gate for adaptive sample blocks, CPU only."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import weakref
import numpy as np
from scipy import ndimage
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks
import faltkarna_v1_multires as baseline

ROOT = Path(__file__).resolve().parents[2]


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def differences(a, b):
    return int(np.count_nonzero(np.ascontiguousarray(a).view(np.uint32) !=
                                np.ascontiguousarray(b).view(np.uint32)))


def observe(samples, origin, pitch, points):
    before = [digest(a) for a in (samples, origin, points)]
    field = AdaptiveSampleBlocks(samples, origin, pitch)
    indices = np.indices(samples.shape).reshape(3, -1).T
    rebuilt = field.samples_at(indices).reshape(samples.shape)
    expected = ndimage.map_coordinates(samples, ((points - origin) / pitch).T,
                                      order=1, mode="nearest")
    actual = field.query(points)
    # Internal split planes lie halfway between their adjacent fine samples.
    # Check both sides using the same fine-grid query as the reference.
    face_points = []
    for node in field.nodes:
        left_id, right_id = node["children"]
        if left_id < 0:
            continue
        left, right = field.nodes[left_id], field.nodes[right_id]
        axis = int(np.flatnonzero(left["lo"] != right["lo"])[0])
        center = node["lo"].astype(float) + (node["shape"] - 1) * .5
        center[axis] = right["lo"][axis] - .5
        for side in (-1, 1):
            q = center.copy()
            q[axis] += side * 1e-6
            face_points.append(origin + q * pitch)
    face_points = np.asarray(face_points, dtype=float).reshape(-1, 3)
    face_actual = field.query(face_points)
    face_expected = ndimage.map_coordinates(samples, ((face_points - origin) / pitch).T,
                                           order=1, mode="nearest")
    after = [digest(a) for a in (samples, origin, points)]
    row = dict(grid_differences=differences(rebuilt, samples),
               query_differences=differences(actual, expected),
               face_differences=differences(face_actual, face_expected),
               face_points=len(face_points), stored_samples=int(field.payload.size),
               nodes=int(field.nodes.size), storage_bytes=field.storage_bytes,
               inputs_unchanged=before == after,
               hashes={"nodes": digest(field.nodes), "payload": digest(field.payload),
                       "grid": digest(rebuilt), "queries": digest(actual),
                       "face_queries": digest(face_actual)})
    return row, dict(grid=rebuilt, queries=actual, reference_queries=expected,
                     face_queries=face_actual, reference_face_queries=face_expected)


def controls():
    rng = np.random.default_rng(431)
    signed_zero = np.zeros((7, 5, 3), dtype=np.float32)
    signed_zero[::2] = np.float32(-0.)
    controls = dict(random=rng.normal(size=(13, 11, 7)).astype(np.float32),
                    thin_x=rng.normal(size=(1, 7, 9)).astype(np.float32),
                    thin_y=rng.normal(size=(9, 1, 7)).astype(np.float32),
                    thin_z=rng.normal(size=(7, 9, 1)).astype(np.float32),
                    constant=np.full((31, 23, 17), 2.5, dtype=np.float32),
                    signed_zero=signed_zero)
    rows = []
    for name, samples in controls.items():
        points = rng.uniform(-1, np.array(samples.shape) + 1, (513, 3))
        origin = np.zeros(3)
        a, _ = observe(samples, origin, 1., points)
        b, _ = observe(samples, origin, 1., points)
        rows.append(dict(case=name, result=a, repeat=a == b))
    rejected = 0
    good = np.ones((3, 4, 5), dtype=np.float32)
    for samples, origin, pitch in (
        (good.astype(np.float64), [0, 0, 0], 1), (good[0], [0, 0, 0], 1),
        (good[:0], [0, 0, 0], 1), (good * np.nan, [0, 0, 0], 1),
        (good, [0, 0], 1), (good, [0, np.nan, 0], 1),
        (good, [0, 0, 0], 0), (good, [0, 0, 0], -1),
        (good, [0, 0, 0], True), (good, [0, 0, 0], "1"),
        (good, [0, 0, 0], np.inf)):
        try:
            AdaptiveSampleBlocks(samples, origin, pitch)
        except ValueError:
            rejected += 1
    field = AdaptiveSampleBlocks(good, [0, 0, 0], 1.)
    for call, arg in ((field.query, [[np.nan, 0, 0]]), (field.query, [0, 0, 0]),
                      (field.samples_at, [[-1, 0, 0]]), (field.samples_at, [[3, 0, 0]]),
                      (field.samples_at, [[0., 0., 0.]])):
        try:
            call(arg)
        except ValueError:
            rejected += 1
    for size in (True, 1, 4097, 8.5):
        try:
            AdaptiveSampleBlocks(good, [0, 0, 0], 1., leaf_samples=size)
        except ValueError:
            rejected += 1
    return rows, rejected


def main():
    wp, _ = baseline.M2S._import_warp()
    rows, arrays = [], {}
    for leg in range(2):
        vertices, faces = baseline._bracket_mesh()
        pitch, _ = baseline.M2S.valj_pitch_for_feature(vertices, faces, 2., "auto")
        origin = vertices.min(0) - 12 * pitch
        fine = baseline._global_fall(wp, "cpu", vertices, faces, pitch, origin, "fine")
        samples = baseline.SMF.tat_sd_fran_gles(fine["sf"], fine["shape_l"], baseline.BLOCK)
        points = vertices[faces].mean(axis=1)
        rng = np.random.default_rng(20260912)
        points = points[rng.choice(len(points), size=min(4000, len(points)), replace=False)]
        interior = fine["origin"] + rng.uniform(-1, np.array(samples.shape) + 1, (8193, 3)) * pitch
        combined = np.concatenate((points, interior))
        row, outputs = observe(samples, fine["origin"], pitch, combined)
        surface = outputs["queries"][:len(points)]
        row.update(fine_active_samples=fine["aktiva_voxlar"],
                   sample_ratio=row["stored_samples"] / fine["aktiva_voxlar"],
                   byte_ratio=row["storage_bytes"] / (4 * fine["aktiva_voxlar"]),
                   surface_median=float(np.median(np.abs(surface))),
                   surface_p95=float(np.percentile(np.abs(surface), 95)))
        rows.append(row)
        arrays.update({f"leg_{leg}_{key}": value for key, value in outputs.items()})
        print(json.dumps(row), flush=True)
    control_rows, rejected = controls()
    temporary = np.arange(210, dtype=np.float32).reshape(7, 5, 6).copy()
    pointer = weakref.ref(temporary)
    detached = AdaptiveSampleBlocks(temporary, [0, 0, 0], 1.)
    del temporary
    released = pointer() is None and detached.samples_at(np.array([[6, 4, 5]]))[0] == 209
    all_rows = rows + [r["result"] for r in control_rows]
    gates = dict(grid_exact=all(r["grid_differences"] == 0 for r in all_rows),
                 query_exact=all(r["query_differences"] == 0 for r in all_rows),
                 interface_exact=all(r["face_differences"] == 0 for r in all_rows),
                 sample_budget=all(r["sample_ratio"] <= .5 for r in rows),
                 byte_budget=all(r["byte_ratio"] <= .5 for r in rows),
                 repeat=rows[0] == rows[1] and all(r["repeat"] for r in control_rows),
                 inputs_unchanged=all(r["inputs_unchanged"] for r in all_rows),
                 invalid_rejected=rejected == 20,
                 dense_source_released=bool(released))
    report = dict(status="VERIFIED-FRESH" if all(gates.values()) else "OWN-GATE-FAIL",
                  scope="CPU adaptive retained storage; dense fine reference required during build.",
                  rows=rows, controls=control_rows, rejected=rejected, gates=gates)
    dest = ROOT / "reports"
    dest.mkdir(exist_ok=True)
    (dest / "adaptive_blocks.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(dest / "adaptive_blocks_arrays.npz", **arrays)
    print(json.dumps(dict(gates=gates, status=report["status"])))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
