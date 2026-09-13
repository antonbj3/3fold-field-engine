"""Pytest wrappers around the benchmark family.

Every component runs as a script in a fresh subprocess in its reduced (--snabb) sizes, and the test
asserts exit code 0 and that no sub-measurement came back BROKEN. Sizes and sub-measurements that
need a CUDA device are skipped when no device is present; the numbers in docs/BENCHMARKS.md for
those rows come from a device run, not from this suite.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "src", "field_engine", "bench")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from field_paths import field_path
PARTS = os.path.join(BENCH, "artifacts", "_parts")


def _cuda_available():
    try:
        import warp as wp
        wp.init()
        return wp.is_cuda_available()
    except Exception:
        return False


CUDA = _cuda_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device (CUDA-ONLY row)")


def run(script, args=(), timeout=600):
    path = str(field_path(ROOT, "bench/"+script))
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    proc = subprocess.run([sys.executable, path, *args], capture_output=True, text=True,
                          timeout=timeout, env=env, cwd=BENCH)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return proc.stdout


def part(namn):
    with open(os.path.join(PARTS, f"{namn}.json")) as fh:
        return json.load(fh)


def assert_no_broken(namn, tillatna=("OK", "CUDA-ONLY", "GPU-GATE")):
    d = part(namn)
    bad = {k: v.get("status") for k, v in d["delar"].items() if v.get("status") not in tillatna}
    assert not bad, f"{namn}: {bad} :: {d.get('fel')}"
    return d


# --- components on the CPU backend ---------------------------------------------------------------
def test_bench_faltkarna():
    run("_komp_faltkarna.py", ["--snabb"])
    d = assert_no_broken("faltkarna")
    assert d["delar"]["korrekthet_konservering"]["GRON"] is True
    assert d["delar"]["determinism"]["bit_identisk"] is True


def test_bench_mesh_sdf():
    run("_komp_mesh_sdf.py", ["--snabb"])
    d = assert_no_broken("mesh_sdf")
    s = d["delar"]["solidfyllning_analytiska_kroppar"]
    assert s["GRON_alla"] is True, s["varsta_rel_fel"]
    assert d["delar"]["fail_open_probe_oppen_mesh"]["flaggar_openen_sjalv"] is True


def test_bench_ikarus():
    run("_komp_ikarus.py", ["--snabb"])
    d = assert_no_broken("ikarus")
    assert all(r["GRON"] for r in d["delar"]["storlekar_SML"]["rader"])
    assert d["delar"]["determinism"]["bit_identisk"] is True


def test_bench_occ():
    run("_komp_occ.py", ["--snabb"])
    d = assert_no_broken("occ_cad")
    assert d["delar"]["korrekthet_boolean_sfar"]["GRON"] is True
    assert d["delar"]["determinism_tessellering"]["bit_identisk"] is True


def test_bench_e2e():
    run("_komp_e2e.py", ["--snabb"])
    d = assert_no_broken("e2e")
    steg = d["delar"]["kedja_brep_mesh_sdf_falt_mesh"]["tidslinje"]
    mc = [s for s in steg if s["steg"].startswith("3_")][0]
    assert mc["rel_mc_vs_sluten_form"] < 0.05, mc


def test_bench_importprobe():
    run("_komp_importprobe.py", timeout=900)
    d = assert_no_broken("importprobe")
    r = d["delar"]["import_alla_moduler"]
    assert r["n_broken"] == 0, [x for x in r["rader"] if not x["kor"]]


def test_bench_driver_aggregates_and_gates():
    """The driver aggregates the existing part files and applies its pre-registered gate."""
    out = run("faltbench_v1.py", ["--ingen-korning", "--snabb"], timeout=300)
    assert '"grind"' in out
    assert '"fail": []' in out, out[-2000:]


# --- rows that need a CUDA device -----------------------------------------------------------------
@needs_cuda
def test_bench_faltkarna_gpu_vs_cpu():
    run("_komp_faltkarna.py", ["--snabb"])
    r = part("faltkarna")["delar"]["gpu_vs_cpu"]
    assert r["status"] == "OK" and r["rader"]


@needs_cuda
def test_bench_ikarus_gpu_vs_cpu():
    run("_komp_ikarus.py", ["--snabb"])
    r = part("ikarus")["delar"]["gpu_vs_cpu"]
    assert r["status"] == "OK" and r["rader"]


@needs_cuda
def test_bench_large_sizes_present():
    """Size L of the field kernel and of the mesh-to-SDF op is only measured on a device."""
    run("_komp_faltkarna.py")
    labels = [r["label"] for r in part("faltkarna")["delar"]["analytiskt_testfalt_SML"]["storlekar"]]
    assert "L" in labels, labels
