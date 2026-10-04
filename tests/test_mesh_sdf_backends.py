"""Native mesh->SDF backend (raypar_vindning_native) against the NumPy path.

1. Bit identity: solid, surface layer, sd bits, window and parity diagnostic equal the NumPy path on
   analytic bodies, a non-dyadic pitch, and the area-filter meshes at offsets up to 1e6 mm, for 1 and
   3 threads; the full op (block classification, tiles, final occupancy) is equal too.
2. Non-finite input raises ValueError on every path (before: empty window, gate OK).
3. A backend that cannot run (no library, input outside the kernel contract, including vertex or face
   arrays that are not (n, 3)) falls back to the NumPy path, declares it and warns; the kernel itself
   refuses bad face indices.
4. The C++ kernel under AddressSanitizer/UndefinedBehaviorSanitizer (tests/native_mesh_sdf_sanitizer.cpp).
The library is built here from source with the documented command; tests that need a compiler, OpenMP
or the sanitizer runtime are skipped when it is missing.
"""
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "field_engine"))
import faltkarna_v1_mesh_to_sdf as M2S  # noqa: E402

SOURCE = ROOT / "src" / "field_engine" / "mesh_sdf_native_v1" / "field.cpp"
CXX = shutil.which(os.environ.get("CXX", "c++"))
BUILD = ["-std=c++17", "-O3", "-fopenmp", "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC"]
SANITIZE = ["-std=c++17", "-O1", "-g", "-fopenmp", "-fno-fast-math", "-ffp-contract=off", "-fno-omit-frame-pointer",
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
METOD = "raypar_vindning_native"


@pytest.fixture(scope="session")
def native_library(tmp_path_factory):
    if CXX is None:
        pytest.skip("no C++ compiler")
    out = tmp_path_factory.mktemp("native") / "mesh_sdf_native.so"
    subprocess.run([CXX, *BUILD, str(SOURCE), "-o", str(out)], check=True, capture_output=True, timeout=300)
    return out


@pytest.fixture
def native(native_library, monkeypatch):
    monkeypatch.setenv("FIELD_MESH_SDF_LIBRARY", str(native_library))
    return native_library


BOX_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]], np.int64)


def box(lo, hi, inverted=False):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)
    return V, (BOX_F[:, ::-1].copy() if inverted else BOX_F.copy())


def join(*parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def rotated_box():
    V, F = box((-12.0, -8.0, -6.0), (12.0, 8.0, 6.0))
    a = math.radians(30.0)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    b = math.radians(20.0)
    R = R @ np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return V @ R.T, F


def sphere(n_lat=16, n_lon=32, r=9.0):
    V = [[0.0, 0.0, r]]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            V.append([r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)])
    V.append([0.0, 0.0, -r])
    F = []
    for j in range(n_lon):
        F.append([0, 1 + j, 1 + (j + 1) % n_lon])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b = 1 + i * n_lon + j, 1 + i * n_lon + (j + 1) % n_lon
            c, d = a + n_lon, b + n_lon
            F.append([a, c, b])
            F.append([b, c, d])
    bot = len(V) - 1
    for j in range(n_lon):
        a, b = 1 + (n_lat - 2) * n_lon + j, 1 + (n_lat - 2) * n_lon + (j + 1) % n_lon
        F.append([a, bot, b])
    return np.array(V, np.float64), np.array(F, np.int64)


def area_filter_box(extents, offset):
    x = float(offset)
    dx, dy, dz = extents
    return box((x, x, x), (x + dx, x + dy, x + dz))


CASES = [
    ("box_axis", *box((0.0, 0.0, 0.0), (24.0, 16.0, 12.0)), 1.0),
    ("box_rotated", *rotated_box(), 0.7),
    ("sphere", *sphere(), 0.5),
    ("sphere_pitch_0.3", *sphere(), 0.3),
] + [(f"area_filter_{off:g}", *area_filter_box((0.04, 0.04, 0.01), off), 0.001) for off in (0.0, 1.0e3, 1.0e6)]


def _field(metod, V, F, pitch, lo):
    d = {}
    r = M2S.surface_raster_and_flood(V, F, pitch, lo, metod=metod, device="cpu", diag_ut=d)
    return r, d


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_backend_bit_identical(native, case):
    name, V, F, pitch = case
    lo = V.min(0) - 3.0 * pitch
    (g0, s0, y0, solid0, sd0), d0 = _field("raypar_vindning", V, F, pitch, lo)
    (g1, s1, y1, solid1, sd1), d1 = _field(METOD, V, F, pitch, lo)
    assert "fallback" not in d1, d1.get("fallback")
    assert d1["backend"] == "native_cpu"
    assert np.array_equal(g0, g1) and tuple(s0) == tuple(s1)
    assert np.array_equal(solid0, solid1)
    assert np.array_equal(y0, y1)
    assert np.array_equal(sd0.view(np.uint32), sd1.view(np.uint32))
    assert d0["n_vindning_vs_paritet_diff"] == d1["n_vindning_vs_paritet_diff"]


@pytest.mark.parametrize("nthreads", [1, 3])
def test_thread_count_does_not_change_bits(native, nthreads):
    import mesh_sdf_native_v1 as NAT
    V, F = sphere()
    pitch = 0.3
    lo = V.min(0) - 3.0 * pitch
    g0, s0, y0, solid0, sd0 = M2S.surface_raster_and_flood(V, F, pitch, lo, metod="raypar_vindning")
    g1, s1, y1, solid1, sd1, d = NAT.field_stage(V, F, pitch, lo, nthreads=nthreads)
    assert np.array_equal(solid0, solid1) and np.array_equal(y0, y1)
    assert np.array_equal(sd0.view(np.uint32), sd1.view(np.uint32))


def test_full_op_identical(native):
    import warp as wp
    V, F = sphere()
    pitch = 0.5
    lo = V.min(0) - 3.0 * pitch
    r0 = M2S.mesh_to_sdf_del(wp, V, F, pitch, 1.0, lo, "cpu", metod="raypar_vindning", returnera_falt=True)
    r1 = M2S.mesh_to_sdf_del(wp, V, F, pitch, 1.0, lo, "cpu", metod=METOD, returnera_falt=True)
    assert r1["signering"]["backend"] == "native_cpu"
    for k in ("kind", "active_ids", "tiles"):
        assert np.array_equal(r0["sf"][k], r1["sf"][k]), k
    assert np.array_equal(r0["solid_final"], r1["solid_final"])
    assert np.array_equal(r0["sd"].view(np.uint32), r1["sd"].view(np.uint32))
    assert r0["n_diff_pre_margin_gpu_vs_cpu"] == r1["n_diff_pre_margin_gpu_vs_cpu"] == 0
    assert r0["vattentathet"]["status"] == r1["vattentathet"]["status"]


PITCH_A = 0.5
FRAME = box((4.0, 4.0, 4.0), (16.0, 16.0, 16.0))


# ---------------------------------------------------------------- non-finite input and fallback
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("metod", ["raypar_vindning", METOD, "raypar_paritet"])
def test_nonfinite_vertex_raises(bad, metod):
    V, F = FRAME
    V = V.copy()
    V[6, 2] = bad
    with pytest.raises(ValueError):
        M2S.surface_raster_and_flood(V, F, PITCH_A, np.zeros(3), metod=metod, device="cpu")


def test_nonfinite_op_raises():
    import warp as wp
    V, F = FRAME
    V = V.copy()
    V[0, 0] = float("nan")
    with pytest.raises(ValueError):
        M2S.mesh_to_sdf_del(wp, V, F, PITCH_A, 0.0, np.zeros(3), "cpu")
    with pytest.raises(ValueError):
        M2S.mesh_to_sdf_del(wp, FRAME[0], F, float("nan"), 0.0, np.zeros(3), "cpu")


def test_missing_library_falls_back_warned(monkeypatch):
    monkeypatch.delenv("FIELD_MESH_SDF_LIBRARY", raising=False)
    V, F = FRAME
    (g0, s0, y0, solid0, sd0), d0 = _field("raypar_vindning", V, F, PITCH_A, np.zeros(3))
    with pytest.warns(RuntimeWarning, match="NumPy path"):
        (g1, s1, y1, solid1, sd1), d1 = _field(METOD, V, F, PITCH_A, np.zeros(3))
    assert "FIELD_MESH_SDF_LIBRARY" in d1["fallback"] and d1["metod_begard"] == METOD
    assert np.array_equal(solid0, solid1) and np.array_equal(sd0.view(np.uint32), sd1.view(np.uint32))


def test_unsupported_input_falls_back_declared(native):
    V, F = FRAME
    F = F[:0]
    (g0, s0, y0, solid0, sd0), d0 = _field("raypar_vindning", V, F, PITCH_A, np.zeros(3))
    with pytest.warns(RuntimeWarning, match="NumPy path"):
        (g1, s1, y1, solid1, sd1), d1 = _field(METOD, V, F, PITCH_A, np.zeros(3))
    assert "fallback" in d1 and d1["metod_begard"] == METOD
    assert np.array_equal(solid0, solid1) and np.array_equal(sd0.view(np.uint32), sd1.view(np.uint32))


def test_bad_face_index_falls_back_not_crash(native):
    """Bad indices reach the NumPy path (same result or same IndexError as the reference), never the
    kernel (an unchecked index read out of bounds and crashed the process)."""
    V, F = FRAME
    neg = F.copy()
    neg[3, 1] = -1
    (g0, s0, y0, solid0, sd0), d0 = _field("raypar_vindning", V, neg, PITCH_A, np.zeros(3))
    with pytest.warns(RuntimeWarning):
        (g1, s1, y1, solid1, sd1), d1 = _field(METOD, V, neg, PITCH_A, np.zeros(3))
    assert "face index" in d1["fallback"]
    assert np.array_equal(solid0, solid1) and np.array_equal(sd0.view(np.uint32), sd1.view(np.uint32))
    oob = F.copy()
    oob[5, 2] = len(V) + 100000
    with pytest.warns(RuntimeWarning), pytest.raises(IndexError):
        _field(METOD, V, oob, PITCH_A, np.zeros(3))


def test_flat_vertices_fall_back_not_read_past_array(native):
    """A flat (3n,) vertex array has len(V) = 3n, so face indices in [n, 3n) passed the face-range
    check and the kernel read past the array (ASan heap-buffer-overflow; an empty field or SIGSEGV).
    The reference raises IndexError, and so must the native method, through the declared fallback."""
    import mesh_sdf_native_v1 as NAT
    V, F = FRAME
    flat = V.reshape(-1).copy()
    T = F + len(V)
    with pytest.raises(IndexError):
        _field("raypar_vindning", flat, T, PITCH_A, np.zeros(3))
    with pytest.warns(RuntimeWarning, match="NumPy path"), pytest.raises(IndexError):
        _field(METOD, flat, T, PITCH_A, np.zeros(3))
    for v, t in ((flat, T), (V, F.reshape(-1)), (V, F[:, :2].copy())):
        with pytest.raises(NAT.NativeUnavailable, match=r"\(n, 3\) arrays"):
            NAT.field_stage(v, t, PITCH_A, np.zeros(3))


def test_kernel_refuses_bad_face_index(native):
    import mesh_sdf_native_v1 as NAT
    V, F = FRAME
    for bad in (-1, len(V)):
        T = F.copy()
        T[5, 2] = bad
        with pytest.raises(NAT.NativeUnavailable, match="returned 2"):
            NAT.field_stage(V, T, PITCH_A, np.zeros(3))


def test_replaced_library_is_refused(native, tmp_path):
    import mesh_sdf_native_v1 as NAT
    lib = tmp_path / "copy.so"
    lib.write_bytes(native.read_bytes())
    NAT.lib(str(lib))
    # Atomic replacement keeps the loaded inode mapped; its identity must not change silently.
    replacement = tmp_path / "replacement.so"
    replacement.write_bytes(native.read_bytes() + b"different build")
    replacement.replace(lib)
    with pytest.raises(RuntimeError, match="Restart the process"):
        NAT.lib(str(lib))


def test_native_sanitizer(tmp_path):
    """tests/native_mesh_sdf_sanitizer.cpp under ASan/UBSan: exact fields, canaries, refusals."""
    if CXX is None:
        pytest.skip("no C++ compiler")
    probe = tmp_path / "probe.cpp"
    probe.write_text("#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }\n")
    p = subprocess.run([CXX, *SANITIZE, str(probe), "-o", str(tmp_path / "probe")], capture_output=True, text=True)
    if p.returncode != 0:
        pytest.skip("sanitizer runtime or OpenMP unavailable: " + p.stderr.strip()[-200:])
    exe = tmp_path / "mesh_sdf_sanitizer"
    subprocess.run([CXX, *SANITIZE, str(ROOT / "tests" / "native_mesh_sdf_sanitizer.cpp"), "-o", str(exe)],
                   check=True, capture_output=True, timeout=300)
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    assert '"exact":true' in r.stdout and '"sanitizers":"address,undefined"' in r.stdout
