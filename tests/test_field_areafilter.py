import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PITCH = 0.001
OFFSETS = (0.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6)
MESHES = {
    "obj_0.04": (0.04, 0.04, 0.01),
    "bin_2^-4": (0.0625, 0.0625, 0.015625),
}
FACES = np.asarray(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 7, 4], [0, 3, 7], [1, 2, 6], [1, 6, 5],
    ],
    dtype=np.int64,
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ENGINE = _load(ROOT / "src/field_engine/faltkarna_v1_mesh_to_sdf.py", "field_area_filter_f1")
def _baseline_path():
    """Pre-fix engine (a09ed16) materialised from git for the regression comparison."""
    import subprocess
    import tempfile
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show", "a09ed16:src/field_engine/faltkarna_v1_mesh_to_sdf.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    path = Path(tempfile.mkdtemp()) / "faltkarna_v1_mesh_to_sdf_baseline.py"
    path.write_text(src)
    return path


BASELINE = _load(_baseline_path(), "field_area_filter_baseline")


def _mesh(extents, offset):
    x = float(offset)
    dx, dy, dz = map(float, extents)
    V = np.asarray(
        [
            [x, x, x], [x + dx, x, x], [x + dx, x + dy, x], [x, x + dy, x],
            [x, x, x + dz], [x + dx, x, x + dz],
            [x + dx, x + dy, x + dz], [x, x + dy, x + dz],
        ],
        dtype=np.float64,
    )
    return V, FACES.copy()


def _lo(V, mode):
    if mode == "zero":
        return np.zeros(3, dtype=np.float64)
    return np.asarray(V, dtype=np.float64).min(axis=0) - 3.0 * PITCH


def _surface(engine, V, T, mode):
    diag = {}
    lo = _lo(V, mode)
    gmin, shape, _, solid, sd = engine.surface_raster_and_flood(
        V,
        T,
        PITCH,
        lo,
        global_shape=None,
        metod="raypar_vindning",
        wp=None,
        device="cpu",
        diag_ut=diag,
        ytkorrektion=True,
    )
    origin = np.asarray(lo, dtype=np.float64) + np.asarray(gmin, dtype=np.float64) * PITCH
    return origin, tuple(int(x) for x in shape), solid, sd, diag


def _reference_engine_count(V, origin, shape):
    Vq = [[Fraction.from_float(float(v)) for v in row] for row in V]
    lo = [min(row[a] for row in Vq) for a in range(3)]
    hi = [max(row[a] for row in Vq) for a in range(3)]
    jx = PITCH * ENGINE.JITTER_FRAC * ENGINE._GYLLENE[0]
    jy = PITCH * ENGINE.JITTER_FRAC * ENGINE._GYLLENE[1]
    nx, ny, nz = shape
    nxin = sum(
        1
        for i in range(nx)
        if lo[0] <= Fraction.from_float(float(origin[0] + i * PITCH + jx)) <= hi[0]
    )
    nyin = sum(
        1
        for j in range(ny)
        if lo[1] <= Fraction.from_float(float(origin[1] + j * PITCH + jy)) <= hi[1]
    )
    nzin = sum(
        1
        for k in range(nz)
        if lo[2] <= Fraction.from_float(float(origin[2] + k * PITCH)) < hi[2]
    )
    return nxin * nyin * nzin


@pytest.mark.parametrize("mesh_name", tuple(MESHES))
@pytest.mark.parametrize("offset", OFFSETS)
def test_f1_keeps_projected_faces_and_rejects_z_parallel(mesh_name, offset):
    V, T = _mesh(MESHES[mesh_name], offset)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1 = B[:, :2] - A[:, :2]
    e2 = C[:, :2] - A[:, :2]
    a2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    scale = np.maximum(np.abs(e1).max(axis=1), np.abs(e2).max(axis=1))
    keep = np.abs(a2) > 1.0e-12 * scale * scale
    degenerate = a2 == 0.0
    assert int(np.count_nonzero(keep)) == 4
    assert int(np.count_nonzero(degenerate)) == 8
    assert int(np.count_nonzero(keep & degenerate)) == 0
    origin, shape, _, _, _ = _surface(ENGINE, V, T, "zero")
    hits, _, _ = ENGINE._kolumntraffar(V, T, PITCH, origin, shape[0], shape[1])
    n_xy = 0
    Vq = [[Fraction.from_float(float(v)) for v in row] for row in V]
    jx = PITCH * ENGINE.JITTER_FRAC * ENGINE._GYLLENE[0]
    jy = PITCH * ENGINE.JITTER_FRAC * ENGINE._GYLLENE[1]
    n_xy += sum(
        1
        for i in range(shape[0])
        if min(row[0] for row in Vq) <= Fraction.from_float(float(origin[0] + i * PITCH + jx))
        <= max(row[0] for row in Vq)
    )
    n_xy *= sum(
        1
        for j in range(shape[1])
        if min(row[1] for row in Vq) <= Fraction.from_float(float(origin[1] + j * PITCH + jy))
        <= max(row[1] for row in Vq)
    )
    assert len(hits) == 2 * n_xy


@pytest.mark.parametrize("mesh_name", tuple(MESHES))
@pytest.mark.parametrize("mode", ("zero", "follow"))
def test_f1_is_byte_identical_at_zero(mesh_name, mode):
    V, T = _mesh(MESHES[mesh_name], 0.0)
    _, _, solid_base, sd_base, _ = _surface(BASELINE, V, T, mode)
    origin, shape, solid_fix, sd_fix, _ = _surface(ENGINE, V, T, mode)
    assert solid_fix.tobytes() == solid_base.tobytes()
    assert sd_fix.tobytes() == sd_base.tobytes()
    old_hits = BASELINE._kolumntraffar(V, T, PITCH, origin, shape[0], shape[1])
    new_hits = ENGINE._kolumntraffar(V, T, PITCH, origin, shape[0], shape[1])
    assert all(a.tobytes() == b.tobytes() for a, b in zip(old_hits, new_hits))


def test_n_solid_reference():
    mismatches = []
    for mesh_name, extents in MESHES.items():
        for offset in OFFSETS:
            V, T = _mesh(extents, offset)
            for mode in ("zero", "follow"):
                origin, shape, solid, _, _ = _surface(ENGINE, V, T, mode)
                expected = _reference_engine_count(V, origin, shape)
                actual = int(np.count_nonzero(solid))
                if actual != expected:
                    mismatches.append((mesh_name, offset, mode, actual, expected))
    assert not mismatches, repr(mismatches)
