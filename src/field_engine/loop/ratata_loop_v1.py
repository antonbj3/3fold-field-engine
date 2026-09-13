#!/usr/bin/env python3
"""The closed loop: search -> build -> gate -> a failed gate becomes a constraint -> search again.

The round:
  1. run the search (the constraint file is read back in as extra constraints when it exists);
  2. build the top K winners in parallel, one process per candidate;
  3. gate each build in parallel;
  4. every gate failure is written to the constraint file with its measured value, and the next search
     round reads it and penalises or excludes that region;
  5. if a candidate passes everything, write the STEP and stop: a green result is a result;
  6. after N rounds without progress, stop and print an escalation, which is where the search space
     needs to be widened or a constraint reformulated.

The steps are pluggable commands (--search-cmd/--build-cmd/--gate-cmd), so build glue can be attached
without touching the loop. Without a build command the loop still runs search plus constraint feedback
and skips the gates.

Run `python ratata_loop_v1.py --search-cmd "... --out {out}" [--build-cmd "... {winner} {step}"]
[--gate-cmd "... {step}"] [--max-rounds 6] [--top-k 3]`. Paths default to ./ratata_state and
./ratata_logs and can be overridden with RATATA_STATE_DIR / RATATA_LOG_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.environ.get("RATATA_ROOT", os.getcwd())
STATE_DIR = os.environ.get("RATATA_STATE_DIR", os.path.join(ROOT, "ratata_state"))
CONSTRAINTS = os.path.join(STATE_DIR, "ratata_constraints.json")
LOG_DIR = os.environ.get("RATATA_LOG_DIR", os.path.join(ROOT, "ratata_logs"))


def run(cmd: str, log: str, timeout: int = 1800) -> int:
    with open(log, "w") as fh:
        try:
            p = subprocess.run(["bash", "-c", cmd], stdout=fh,
                               stderr=subprocess.STDOUT, timeout=timeout,
                               preexec_fn=lambda: os.nice(19))
            return p.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\nTIMEOUT {timeout}s\n")
            return 124


def load_constraints() -> list:
    if os.path.exists(CONSTRAINTS):
        try:
            return json.load(open(CONSTRAINTS)).get("rows", [])
        except Exception:
            return []
    return []


def add_constraint(row: dict):
    rows = load_constraints()
    rows.append(row)
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump({"schema": "ratata_constraints_v1", "rows": rows}, open(CONSTRAINTS, "w"),
              indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-cmd", required=True,
                    help="{out} is replaced by the path of the search result json")
    ap.add_argument("--build-cmd", default=None,
                    help="{winner}=winner json, {step}=target STEP; without it build and gates are skipped")
    ap.add_argument("--gate-cmd", action="append", default=[],
                    help="{step} is replaced; may be given several times, run in order per candidate")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--stall-rounds", type=int, default=3,
                    help="rounds without a feasible candidate before escalating")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    stall = 0
    for rnd in range(1, args.max_rounds + 1):
        t0 = time.time()
        print(f"\n===== ROUND {rnd}/{args.max_rounds} =====")
        n_cons = len(load_constraints())
        print(f"[{time.strftime('%H:%M:%S')}] search (with {n_cons} fed-back constraints) ...")
        out = os.path.join(LOG_DIR, f"r{rnd}_search.json")
        rc = run(args.search_cmd.format(out=out), os.path.join(LOG_DIR, f"r{rnd}_search.log"))
        if rc != 0 or not os.path.exists(out):
            print(f"  the search failed (exit {rc}); see the log, escalating")
            break
        try:
            front = json.load(open(out)).get("front", [])
        except Exception:
            front = []
        print(f"  front: {len(front)} points ({time.time()-t0:.1f} s)")
        if not front:
            stall += 1
            add_constraint({"round": rnd, "type": "empty_front",
                            "note": "the search found nothing feasible; the space needs widening"})
            if stall >= args.stall_rounds:
                print("ESCALATION: empty result over several rounds; the search space needs widening")
                break
            continue

        if not args.build_cmd:
            print("  (no build command attached; the winners are in "
                  f"{out}, attach --build-cmd when the build glue exists)")
            return 0

        # build the top K in parallel
        winners = front[: args.top_k]
        procs = []
        for i, w in enumerate(winners):
            wj = os.path.join(LOG_DIR, f"r{rnd}_winner{i}.json")
            json.dump(w, open(wj, "w"))
            step = os.path.join(LOG_DIR, f"ratata_r{rnd}_c{i}.step")
            log = os.path.join(LOG_DIR, f"r{rnd}_build{i}.log")
            cmd = args.build_cmd.format(winner=wj, step=step)
            fh = open(log, "w")
            p = subprocess.Popen(["bash", "-c", cmd], stdout=fh,
                                 stderr=subprocess.STDOUT, preexec_fn=lambda: os.nice(19))
            procs.append((p, step, i, fh))
        built = []
        for p, step, i, fh in procs:
            p.wait()
            fh.close()
            if p.returncode == 0 and os.path.exists(step):
                built.append((step, i))
            else:
                add_constraint({"round": rnd, "type": "build_fail", "candidate": i,
                                "exit": p.returncode})
        print(f"  built: {len(built)}/{len(winners)}")

        # gate in parallel: one candidate's gates in sequence, candidates in parallel
        gate_procs = []
        for step, i in built:
            chain = " && ".join(g.format(step=step) for g in args.gate_cmd)
            log = os.path.join(LOG_DIR, f"r{rnd}_gates{i}.log")
            fh = open(log, "w")
            p = subprocess.Popen(["bash", "-c", chain], stdout=fh, stderr=subprocess.STDOUT,
                                 preexec_fn=lambda: os.nice(19))
            gate_procs.append((p, step, i, fh, log))
        survivors = []
        for p, step, i, fh, log in gate_procs:
            p.wait()
            fh.close()
            if p.returncode == 0:
                survivors.append(step)
            else:
                add_constraint({"round": rnd, "type": "gate_fail", "candidate": i,
                                "exit": p.returncode, "log": os.path.relpath(log, ROOT)})
        print(f"  passed every gate: {len(survivors)}  (round time {time.time()-t0:.0f} s)")
        if survivors:
            print(f"\nGREEN, round {rnd}: {survivors[0]}")
            return 0
        stall += 1
        if stall >= args.stall_rounds:
            print("ESCALATION: gate failures over several rounds without progress "
                  f"(read {CONSTRAINTS} for the fed-back constraints)")
            break
    return 1


if __name__ == "__main__":
    sys.exit(main())
