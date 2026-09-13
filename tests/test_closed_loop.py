"""End-to-end test of the closed loop with stub search, build and gate commands.

The stubs keep the test hermetic: the loop's own behaviour is what is under test, namely that a failed
gate is written back as a constraint and that a passing candidate stops the loop.
"""
import json
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOP = os.path.join(ROOT, "src", "field_engine", "loop", "ratata_loop_v1.py")

SEARCH = textwrap.dedent("""
    import json, sys
    json.dump({"front": [{"w": 10.0}, {"w": 20.0}]}, open(sys.argv[1], "w"))
""")
BUILD = textwrap.dedent("""
    import json, sys
    w = json.load(open(sys.argv[1]))["w"]
    open(sys.argv[2], "w").write(f"STEP {w}")
""")
GATE_FAIL = "import sys; sys.exit(1)"
GATE_PASS = "import sys; sys.exit(0)"


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def _run(tmp_path, gate_body, max_rounds):
    search = _write(tmp_path, "search.py", SEARCH)
    build = _write(tmp_path, "build.py", BUILD)
    gate = _write(tmp_path, "gate.py", gate_body)
    env = {**os.environ, "RATATA_STATE_DIR": str(tmp_path / "state"),
           "RATATA_LOG_DIR": str(tmp_path / "logs")}
    return subprocess.run(
        [sys.executable, LOOP,
         "--search-cmd", f"{sys.executable} {search} {{out}}",
         "--build-cmd", f"{sys.executable} {build} {{winner}} {{step}}",
         "--gate-cmd", f"{sys.executable} {gate} {{step}}",
         "--top-k", "2", "--max-rounds", str(max_rounds), "--stall-rounds", "2"],
        capture_output=True, text=True, timeout=600, env=env)


def test_gate_failure_becomes_a_constraint_and_escalates(tmp_path):
    proc = _run(tmp_path, GATE_FAIL, 4)
    assert proc.returncode == 1, proc.stdout
    assert "ESCALATION" in proc.stdout
    rows = json.load(open(tmp_path / "state" / "ratata_constraints.json"))["rows"]
    assert [r for r in rows if r["type"] == "gate_fail"]


def test_passing_candidate_stops_the_loop(tmp_path):
    proc = _run(tmp_path, GATE_PASS, 4)
    assert proc.returncode == 0, proc.stdout
    assert "GREEN, round 1" in proc.stdout
