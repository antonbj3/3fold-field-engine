"""Native CPU backend for faltkarna_v1_mesh_to_sdf (metod="raypar_vindning_native").

The C++ source in mesh_sdf_native_v1/field.cpp repeats the NumPy/SciPy reference's binary64 operation
sequence (ray winding per voxel, two exact Euclidean distance transforms with SciPy's distance
formula, half-pitch correction, surface layer) and the block classifier, threaded with OpenMP. Its
outputs are bit-identical to the reference (tests/test_mesh_sdf_backends.py).

Build explicitly from the repository root, then set FIELD_MESH_SDF_LIBRARY (or pass library=):

    c++ -std=c++17 -O3 -fopenmp -fno-fast-math -ffp-contract=off -shared -fPIC
        src/field_engine/mesh_sdf_native_v1/field.cpp -o mesh_sdf_native.so

(one command). -march=native is left out: it gave no consistent gain and ties the binary to the build host's CPU.

Import never compiles and nothing writes files. The loaded library is bound to its sha256; replacing
it in-process is refused. A missing or unloadable library raises NativeUnavailable, which
surface_raster_and_flood turns into the NumPy path with the fallback declared and warned.
ASan/UBSan control: tests/native_mesh_sdf_sanitizer.cpp.
"""
import ctypes
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import faltkarna_v1_mesh_to_sdf as M2S  # noqa: E402

THREADS = int(os.environ.get("FIELD_ENGINE_NATIVE_THREADS", "4"))
_LOADED = {}


class NativeUnavailable(RuntimeError):
    """No library configured, a library that does not load, or input outside the kernel contract."""


def lib(library=None):
    """The configured library (argument or FIELD_MESH_SDF_LIBRARY), loaded once per path and bound to
    its sha256. Returns (handle, library_sha256)."""
    name = library or os.environ.get("FIELD_MESH_SDF_LIBRARY")
    if not name:
        raise NativeUnavailable("FIELD_MESH_SDF_LIBRARY is not set; build mesh_sdf_native_v1/field.cpp explicitly")
    path = Path(name).resolve()
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        raise NativeUnavailable(f"cannot read native library: {e}") from e
    key = str(path)
    if key not in _LOADED:
        try:
            L = ctypes.CDLL(key)
        except OSError as e:
            raise NativeUnavailable(str(e)) from e
        P = ctypes.c_void_p
        L.meshsdf_field_cpu.argtypes = [P, ctypes.c_int64, P, ctypes.c_int64] + [ctypes.c_double] * 6 + \
            [ctypes.c_int64] * 3 + [ctypes.c_int, ctypes.c_int] + [P] * 5
        L.meshsdf_field_cpu.restype = ctypes.c_int
        L.meshsdf_classify_cpu.argtypes = [P] + [ctypes.c_int64] * 4 + [ctypes.c_float, P, ctypes.c_int]
        L.meshsdf_classify_cpu.restype = ctypes.c_int
        L.meshsdf_tiles_cpu.argtypes = [P] + [ctypes.c_int64] * 4 + [P, ctypes.c_int64, P, ctypes.c_int]
        L.meshsdf_tiles_cpu.restype = ctypes.c_int
        _LOADED[key] = (digest, L)
    loaded_digest, L = _LOADED[key]
    if digest != loaded_digest:
        raise RuntimeError("Restart the process after replacing the native mesh-SDF library")
    return L, loaded_digest


def field_stage(V, T, pitch, lo, global_shape=None, margin_vox_pad=2, ytkorrektion=None, nthreads=None,
                library=None):
    """Same return tuple as surface_raster_and_flood(metod='raypar_vindning') plus the diagnostics."""
    Vf = np.ascontiguousarray(np.asarray(V, dtype=np.float64))
    Ti = np.ascontiguousarray(np.asarray(T, dtype=np.int64))
    if Vf.ndim != 2 or Vf.shape[1] != 3 or Ti.ndim != 2 or Ti.shape[1] != 3:
        # the kernel reads 3*nv doubles and 3*nt indices from the row counts passed below
        raise NativeUnavailable(f"vertices and faces must be (n, 3) arrays, got {Vf.shape} and {Ti.shape}")
    if not (np.isfinite(Vf).all() and math.isfinite(float(pitch)) and np.isfinite(np.asarray(lo, np.float64)).all()):
        raise ValueError("non-finite vertex, pitch or origin: no field and no certified side")
    ytkorr = M2S.YTKORREKTION_STANDARD if ytkorrektion is None else bool(ytkorrektion)
    gmin, shape_l = M2S._fonster(Vf, pitch, lo, global_shape, margin_vox_pad)
    origin = np.asarray(lo, dtype=np.float64) + gmin.astype(np.float64) * pitch
    nx, ny, nz = shape_l
    N = nx * ny * nz
    if N == 0 or len(Ti) == 0 or N >= 2**31 - 1:
        raise NativeUnavailable(f"empty or oversized input (cells {N}, triangles {len(Ti)})")
    jx = pitch * M2S.JITTER_FRAC * M2S._GYLLENE[0]
    jy = pitch * M2S.JITTER_FRAC * M2S._GYLLENE[1]
    solid = np.empty(shape_l, np.uint8)
    sd = np.empty(shape_l, np.float32)
    yta = np.empty(shape_l, np.uint8)
    stats = np.zeros(4, np.int64)
    ts = np.zeros(5, np.float64)
    L, digest = lib(library)
    rc = L.meshsdf_field_cpu(Vf.ctypes.data, Vf.shape[0], Ti.ctypes.data, Ti.shape[0], float(origin[0]),
                             float(origin[1]), float(origin[2]), float(pitch), float(jx), float(jy), nx, ny, nz,
                             int(ytkorr), int(nthreads or THREADS), solid.ctypes.data, sd.ctypes.data, yta.ctypes.data,
                             stats.ctypes.data, ts.ctypes.data)
    if rc != 0:
        raise NativeUnavailable(f"native field returned {rc}")
    solid = solid.view(bool)
    diag = dict(metod="raypar", regel="vindning", backend="native_cpu", native_library_sha256=digest,
                jitter_frac=M2S.JITTER_FRAC,
                n_par=int(stats[0]), n_traffar=int(stats[1]), n_vindning_vs_paritet_diff=int(stats[2]),
                andel_vindning_vs_paritet=int(stats[2]) / float(N), ytkorrektion=bool(ytkorr))
    if stats[3] == 0 or stats[3] == N:
        from scipy import ndimage
        d_out = ndimage.distance_transform_edt(~solid, sampling=(pitch,) * 3).astype(np.float32)
        d_in = ndimage.distance_transform_edt(solid, sampling=(pitch,) * 3).astype(np.float32)
        sd = (d_out - d_in).astype(np.float32)
        if ytkorr:
            sd = (sd - np.sign(sd) * np.float32(0.5 * pitch)).astype(np.float32)
        diag["edt_scipy"] = True
    return gmin, shape_l, yta.view(bool), solid, sd, diag


def classify(dense, pitch, block, nthreads=None, library=None):
    """Same dict as klassificera_och_evaluera_fran_tatt_falt, on the CPU."""
    L, _digest = lib(library)
    d = np.ascontiguousarray(dense, dtype=np.float32)
    nx, ny, nz = d.shape
    n_bx, n_by, n_bz = -(-nx // block), -(-ny // block), -(-nz // block)
    margin = math.sqrt(3.0) * block * pitch
    kind = np.empty(n_bx * n_by * n_bz, np.int32)
    rc = L.meshsdf_classify_cpu(d.ctypes.data, nx, ny, nz, block, float(np.float32(margin)), kind.ctypes.data,
                                int(nthreads or THREADS))
    if rc != 0:
        raise RuntimeError(f"native classify refused inputs: {rc}")
    active_ids = np.where(kind == 0)[0].astype(np.int32)
    tiles = np.empty((len(active_ids), block, block, block), np.float32)
    if len(active_ids):
        rc = L.meshsdf_tiles_cpu(d.ctypes.data, nx, ny, nz, block, active_ids.ctypes.data, len(active_ids),
                                 tiles.ctypes.data, int(nthreads or THREADS))
        if rc != 0:
            raise RuntimeError(f"native tiles refused inputs: {rc}")
    return dict(kind=kind, active_ids=active_ids, tiles=tiles, n_bx=n_bx, n_by=n_by, n_bz=n_bz,
                n_blocks=int(len(kind)), n_active=int(len(active_ids)), margin_mm=margin)
