#!/usr/bin/env python3
"""Shared measurement helpers for the benchmark family.

No results are produced here: only timing, memory, device and hash helpers, so that every component
script (`_komp_*.py`) measures the same quantities the same way -- p50/p95 over n repetitions, peak
RSS via getrusage, VRAM via the Warp device when there is one, determinism via sha256 over raw bytes.

A component collects its sub-measurements in a `Part` and writes one JSON to
`bench/artifacts/_parts/<name>.json`. Every sub-measurement runs in its own try/except: a crash in
one kernel must never hide the others. Sub-measurements declared `gpu=True` need a CUDA device; with
no device present they are recorded as CUDA-ONLY and skipped, not failed.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
PARTS_DIR = os.path.join(ARTIFACTS, "_parts")
GPU_GATE_MB = 6000  # use the GPU only while other processes hold less than this much VRAM


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def have_cuda():
    try:
        import warp as wp
        wp.init()
        return bool(wp.get_cuda_device_count() > 0)
    except Exception:  # noqa: BLE001
        return False


def device():
    return "cuda:0" if have_cuda() else "cpu"


def timeit(fn, n_reps=5, warm=1):
    """Runs fn() warm times (not measured) and n_reps times (measured).

    Returns (dict with p50/p95/min/all in seconds and n_reps, last return value).
    """
    ret = None
    for _ in range(warm):
        ret = fn()
    ts = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        ret = fn()
        ts.append(time.perf_counter() - t0)
    return dict(p50_s=float(np.percentile(ts, 50)), p95_s=float(np.percentile(ts, 95)),
                min_s=float(min(ts)), all_s=[float(t) for t in ts], n_reps=n_reps), ret


def rss_peak_mb():
    """Peak RSS of this process so far (ru_maxrss is kB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rss_children_peak_mb():
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return float(out)
    except Exception:  # noqa: BLE001
        return None


def gpu_proc_used_mb(pid=None):
    """VRAM held by THIS process according to nvidia-smi's compute-apps list."""
    pid = pid or os.getpid()
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout
        for line in out.strip().splitlines():
            p, m = [x.strip() for x in line.split(",")[:2]]
            if int(p) == pid:
                return float(m)
    except Exception:  # noqa: BLE001
        pass
    return None


def gpu_gate(limit_mb=GPU_GATE_MB, wait_s=None):
    """True when a CUDA device exists and other processes hold less than limit_mb of VRAM.

    Our own process is subtracted: it is other load the gate protects against. Polls up to wait_s
    (default env BENCH_GPU_WAIT_S=300). With no CUDA device at all the gate returns immediately.
    """
    if not have_cuda():
        return False, None
    if wait_s is None:
        wait_s = float(os.environ.get("BENCH_GPU_WAIT_S", "300"))
    t0 = time.time()
    while True:
        used = gpu_used_mb()
        egen = gpu_proc_used_mb() or 0.0
        andra = (used - egen) if used is not None else None
        if andra is None or andra < limit_mb:
            return True, andra
        if time.time() - t0 > wait_s:
            return False, andra
        print(f"[gpu_gate] other processes {andra:.0f} MB >= {limit_mb} -- waiting 15 s",
              file=sys.stderr, flush=True)
        time.sleep(15)


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]


def pct(a, q):
    return float(np.percentile(a, q))


class Part:
    """One component report: collects a status per sub-measurement and writes _parts/<name>.json."""

    def __init__(self, namn, venv="."):
        self.namn = namn
        self.d = dict(komponent=namn, venv=venv, start=now_iso(), host="", pid=os.getpid(),
                      device=device(), delar={}, fel={}, status="OK")
        self.t0 = time.time()

    def kor(self, del_namn, fn, gpu=False):
        """Runs one sub-measurement. With gpu=True the VRAM gate is checked first; with no CUDA
        device the sub-measurement is recorded as CUDA-ONLY and skipped."""
        if gpu:
            ok, used = gpu_gate()
            if not ok:
                status = "CUDA-ONLY" if not have_cuda() else "GPU-GATE"
                self.d["delar"][del_namn] = dict(status=status, gpu_used_mb=used)
                self.d["status"] = "DELVIS"
                print(f"[{self.namn}] {del_namn}: {status}", file=sys.stderr)
                return None
        t0 = time.time()
        print(f"[{self.namn}] {del_namn} ...", file=sys.stderr, flush=True)
        try:
            r = fn()
            if r is None:
                r = {}
            r["status"] = r.get("status", "OK")
            r["wall_s"] = time.time() - t0
            self.d["delar"][del_namn] = r
            print(f"[{self.namn}] {del_namn}: {r['status']} ({r['wall_s']:.1f}s)", file=sys.stderr, flush=True)
            return r
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            self.d["delar"][del_namn] = dict(status="BROKEN", fel=str(e)[:500], wall_s=time.time() - t0)
            self.d["fel"][del_namn] = tb[-3000:]
            self.d["status"] = "DELVIS"
            print(f"[{self.namn}] {del_namn}: BROKEN {e}", file=sys.stderr, flush=True)
            return None

    def skriv(self):
        self.d["slut"] = now_iso()
        self.d["wall_total_s"] = time.time() - self.t0
        self.d["rss_peak_mb"] = rss_peak_mb()
        self.d["rss_children_peak_mb"] = rss_children_peak_mb()
        self.d["gpu_used_mb_slut"] = gpu_used_mb()
        os.makedirs(PARTS_DIR, exist_ok=True)
        p = os.path.join(PARTS_DIR, f"{self.namn}.json")
        with open(p, "w") as fh:
            json.dump(self.d, fh, indent=1, default=_json_default)
        print(f"[{self.namn}] wrote {p} status={self.d['status']}", file=sys.stderr)
        return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def snabb():
    """--snabb: reduced sizes to smoke-test the script itself. Never for reported numbers."""
    return "--snabb" in sys.argv


def repo_paths():
    """Puts the package root and the examples directory on sys.path and returns them."""
    src = os.path.dirname(HERE)                       # src/field_engine
    root = os.path.dirname(os.path.dirname(src))      # repository root
    for p in (src, os.path.join(src, "ikarus_v1"), os.path.join(root, "examples", "parts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return src, root
