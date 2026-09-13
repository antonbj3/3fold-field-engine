#!/usr/bin/env python3
"""part_harness_v1.py -- generic build(theta)->shape / evaluate(shape)->objectives contract, so a
numerical optimizer can call ANY part thousands of times without anyone writing a new script per question.

A part stops being "a script that builds one geometry" and becomes `build(theta) -> shape` plus
`objectives(shape) -> dict` (and optionally `constraints(shape) -> dict`), which an optimiser can call
thousands of times with no new code.

Contract a part module implements:
    BOUNDS: Dict[str, Tuple[float, float]]              -- theta parameter bounds, ordered
    OBJECTIVE_NAMES: List[str]                            -- keys objectives() must return, MINIMIZE convention
    CONSTRAINT_NAMES: List[str]                           -- keys constraints() must return (can be [])
    def build(theta: Dict[str, float]) -> shape: ...      -- may raise on an invalid/degenerate theta
    def objectives(shape, theta) -> Dict[str, float]: ...
    def constraints(shape, theta) -> Dict[str, float]: ... -- convention: value <= 0 is FEASIBLE (pymoo's own
                                                               convention, reused verbatim so nsga2_runner_v1
                                                               needs no sign flip)
    def make_part() -> ParametricPart: ...                -- wires the above into one ParametricPart instance

Measured facts this file bakes in (20-thread desktop CPU):
  * multiprocessing start method MUST be "spawn", never "fork": forking a process that already has
    build123d/OCP imported HANGS (measured: a fork-context pool of build123d workers deadlocked for >2min
    with zero completions, killed by timeout; the same workload under spawn completed in seconds). OCCT
    likely holds internal locks/threads at fork time that don't survive the fork. This harness hard-codes
    spawn and refuses fork.
  * a cold worker pool's first batch is markedly slower than steady state (measured: 2722 evals/min
    for the first batch against 6174-6831 evals/min for later batches at n_workers=8) -- spin-up and
    first-import cost is what a persistent pool (kept alive across the whole run, not one per
    generation) amortizes away.
  * more workers is not monotonically better for this workload: nw=8 and nw=16 measured ~2.7-2.9x serial
    on a short run while nw=20 measured WORSE (~2.5x) -- likely OCCT-internal threading + spawn/pickle
    overhead competing for the same cores. Default n_workers=8 on a >=8-thread machine unless overridden.

Cache key = sha256 of the theta dict rounded to `cache_round` decimals (default 6), so two
floating-point-jittered but practically identical theta never re-run. Checkpointing is per evaluation:
every result (hit or miss) is appended to `cache_path` as one JSON line, flushed and fsync'd before the
call returns, so a killed process loses at most the evaluation in flight.

Error isolation has two layers:
  1. an ordinary Python exception in build/objectives/constraints (e.g. a degenerate theta) is caught in
     _evaluate_theta and turned into a PENALIZED result (objectives/constraints = part.penalty, feasible=
     False, error=<message>) -- the batch continues.
  2. a worker PROCESS dying outright (BrokenProcessPool) is caught per-future in PartHarness.evaluate_batch;
     the dead pool is torn down and rebuilt on the NEXT call, and every future that can no longer be trusted
     is recorded as a penalized "WORKER_CRASH" result rather than silently dropped or crashing the run.

Callable: PartHarness(module_path, factory_attr="make_part", cache_path=..., n_workers=8).evaluate_batch(thetas)
CLI:      python part_harness_v1.py <module_path> [factory_attr] [--n N] [--workers K]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Flat sys.path with bare module names rather than dotted packages. This insertion runs at module
# level, so it also fires inside every spawned worker process (spawn re-imports this module to resolve
# the _worker_init/_worker_eval references), which is what makes a bare part module name resolvable in
# both the main process and every worker.
_OPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARTS_DIR = os.path.join(_OPT_DIR, "parts")
for _p in (_OPT_DIR, _PARTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------------------------- contract
@dataclass
class ParametricPart:
    """The generic contract. `build_fn`/`objectives_fn`/`constraints_fn` MUST be plain module-level
    functions (not lambdas/closures/bound methods) -- workers reconstruct the part by re-importing the
    module (see `load_part`), never by pickling these callables directly (build123d/OCP shapes are not
    picklable, so the shape itself never crosses a process boundary; only the objectives/constraints
    dict of floats does)."""
    name: str
    bounds: Dict[str, Tuple[float, float]]
    build_fn: Callable[[Dict[str, float]], Any]
    objectives_fn: Callable[[Any, Dict[str, float]], Dict[str, float]]
    objective_names: List[str]
    constraints_fn: Optional[Callable[[Any, Dict[str, float]], Dict[str, float]]] = None
    constraint_names: List[str] = field(default_factory=list)
    penalty: float = 1.0e6   # objective/constraint value assigned to a build/eval failure

    def __post_init__(self):
        if not self.bounds:
            raise ValueError("ParametricPart.bounds must be non-empty")
        for k, (lo, hi) in self.bounds.items():
            if not (lo < hi):
                raise ValueError(f"bounds[{k!r}] = ({lo}, {hi}) is not lo < hi")
        if not self.objective_names:
            raise ValueError("ParametricPart.objective_names must be non-empty (declared upfront, not "
                              "inferred, so a failure on the FIRST eval can still be penalized correctly)")

    @property
    def param_names(self) -> List[str]:
        return list(self.bounds.keys())

    def sample_theta(self, rng: random.Random) -> Dict[str, float]:
        return {k: rng.uniform(lo, hi) for k, (lo, hi) in self.bounds.items()}


_PART_REQUIRED_ATTRS = ("bounds", "build_fn", "objectives_fn", "objective_names", "constraint_names",
                        "param_names", "penalty")


def load_part(module_path: str, factory_attr: str = "make_part") -> ParametricPart:
    mod = importlib.import_module(module_path)
    factory = getattr(mod, factory_attr)
    part = factory()
    # DUCK-TYPE check, not isinstance: this module can legitimately be imported under two different
    # identities in the same run -- as `__main__` (CLI invocation: `python3 part_harness_v1.py ...`) and
    # as the bare module `part_harness_v1` (every part module does `from part_harness_v1 import
    # ParametricPart`) -- Python treats those as two DISTINCT module objects with two DISTINCT
    # ParametricPart classes, so isinstance() spuriously fails across that boundary even for a
    # correctly-built part (measured live: hit this exact TypeError on the first CLI smoke test before
    # switching to a structural check).
    missing = [a for a in _PART_REQUIRED_ATTRS if not hasattr(part, a)]
    if missing or type(part).__name__ != "ParametricPart":
        raise TypeError(f"{module_path}.{factory_attr}() returned {type(part)!r} missing {missing} -- "
                         f"expected a ParametricPart-shaped object")
    return part


# ---------------------------------------------------------------------------------------------- eval core
def _evaluate_theta(part: ParametricPart, theta: Dict[str, float]) -> Dict[str, Any]:
    """The ONE evaluation path -- used verbatim by the serial harness path AND by every worker process,
    so there is no logic drift between 'how a serial run evaluates theta' and 'how a parallel run does'."""
    t0 = time.time()
    try:
        shape = part.build_fn(theta)
        raw_obj = part.objectives_fn(shape, theta)
        raw_con = part.constraints_fn(shape, theta) if part.constraints_fn else {}
        objectives = {k: float(raw_obj[k]) for k in part.objective_names}
        constraints = {k: float(raw_con.get(k, 0.0)) for k in part.constraint_names}
        feasible = all(v <= 0.0 for v in constraints.values())
        return {"theta": theta, "objectives": objectives, "constraints": constraints,
                "feasible": feasible, "error": None, "wall_s": time.time() - t0}
    except KeyboardInterrupt:
        raise
    except BaseException as e:   # noqa: BLE001 -- deliberate: isolate a bad theta, never crash the batch
        objectives = {k: part.penalty for k in part.objective_names}
        constraints = {k: part.penalty for k in part.constraint_names}
        return {"theta": theta, "objectives": objectives, "constraints": constraints,
                "feasible": False, "error": f"{type(e).__name__}: {e}", "wall_s": time.time() - t0}


_worker_part: Optional[ParametricPart] = None
_worker_spec: Optional[Tuple[str, str]] = None


def _worker_init(module_path: str, factory_attr: str) -> None:
    global _worker_part, _worker_spec
    _worker_part = load_part(module_path, factory_attr)
    _worker_spec = (module_path, factory_attr)


def _worker_eval(theta: Dict[str, float]) -> Dict[str, Any]:
    if _worker_part is None:
        raise RuntimeError("worker not initialized (missing initializer=_worker_init)")
    return _evaluate_theta(_worker_part, theta)


def _cache_key(part_name: str, theta: Dict[str, float], round_decimals: int) -> str:
    rounded = {k: round(float(v), round_decimals) for k, v in theta.items()}
    blob = json.dumps({"part": part_name, "theta": rounded}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _penalized(part: ParametricPart, theta: Dict[str, float], error: str) -> Dict[str, Any]:
    return {"theta": theta, "objectives": {k: part.penalty for k in part.objective_names},
            "constraints": {k: part.penalty for k in part.constraint_names},
            "feasible": False, "error": error, "wall_s": None}


# ---------------------------------------------------------------------------------------------- harness
class PartHarness:
    """The seat any ParametricPart plugs into. Owns: theta-hash cache, per-evaluation disk checkpoint,
    a persistent (spawn-context) worker pool, and BrokenProcessPool recovery.

    `module_path`/`factory_attr` (not a live ParametricPart) are what get handed to worker processes --
    a live part with build123d callables baked in is not reliably picklable, but "import this module and
    call this factory" is, and it is what makes the cache-and-parallelism seat truly generic across parts.
    """

    def __init__(self, module_path: str, factory_attr: str = "make_part", cache_path: Optional[str] = None,
                 n_workers: int = 1, cache_round: int = 6, mp_start_method: str = "spawn"):
        if mp_start_method != "spawn":
            raise ValueError("mp_start_method must be 'spawn' -- 'fork' measured to HANG with build123d/OCP "
                              "already imported in the parent (see module docstring); this is a hard rail, "
                              "not a default that can be silently overridden into the hang.")
        self.module_path = module_path
        self.factory_attr = factory_attr
        self.part = load_part(module_path, factory_attr)
        self.cache_path = cache_path
        self.n_workers = max(1, int(n_workers))
        self.cache_round = cache_round
        self.mp_start_method = mp_start_method
        self._executor: Optional[ProcessPoolExecutor] = None
        self._fh = None
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.stats = {"n_requested": 0, "n_cache_hits": 0, "n_evaluated": 0, "n_errors": 0,
                      "n_worker_crashes": 0, "n_pool_rebuilds": 0, "n_crash_retries_attempted": 0,
                      "n_crash_retries_recovered": 0}
        if self.cache_path:
            self._fh = open(self.cache_path, "a", buffering=1)

    # -- bounds/name passthrough for optimizer wiring --
    @property
    def bounds(self):
        return self.part.bounds

    @property
    def param_names(self):
        return self.part.param_names

    @property
    def objective_names(self):
        return self.part.objective_names

    @property
    def constraint_names(self):
        return self.part.constraint_names

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = rec.pop("cache_key", None)
                    if key:
                        cache[key] = rec   # last write wins -- JSONL append order == recency
        return cache

    def _checkpoint(self, key: str, rec: Dict[str, Any]) -> None:
        self._cache[key] = rec
        if self._fh:
            self._fh.write(json.dumps({"cache_key": key, "ts": time.time(), **rec}, default=str) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            ctx = mp.get_context(self.mp_start_method)
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx,
                                                  initializer=_worker_init,
                                                  initargs=(self.module_path, self.factory_attr))
            self.stats["n_pool_rebuilds"] += 1
        return self._executor

    def evaluate_batch(self, thetas: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of theta dicts, IN THE GIVEN ORDER (so callers -- e.g. an optimizer's
        population matrix -- can zip results back to rows). Cache hits never touch the pool. Misses run
        parallel (n_workers>1) or inline (n_workers==1, no pool at all -- the serial-baseline code path)."""
        self.stats["n_requested"] += len(thetas)
        results: List[Optional[Dict[str, Any]]] = [None] * len(thetas)
        keys = [_cache_key(self.part.name, t, self.cache_round) for t in thetas]
        miss_idx = []
        for i, k in enumerate(keys):
            hit = self._cache.get(k)
            if hit is not None:
                rec = dict(hit)
                rec["cache_hit"] = True
                results[i] = rec
                self.stats["n_cache_hits"] += 1
            else:
                miss_idx.append(i)

        if not miss_idx:
            return results  # type: ignore[return-value]

        if self.n_workers <= 1:
            for i in miss_idx:
                rec = _evaluate_theta(self.part, thetas[i])
                rec["cache_hit"] = False
                if rec["error"]:
                    self.stats["n_errors"] += 1
                self.stats["n_evaluated"] += 1
                self._checkpoint(keys[i], rec)
                results[i] = rec
            return results  # type: ignore[return-value]

        # Measured under a fault-injection stress test: when one worker dies mid-batch, Python's
        # ProcessPoolExecutor cannot tell you WHICH pending future was the culprit -- it marks EVERY future
        # that was in-flight on the now-dead pool as BrokenProcessPool, including innocent batch-mates that
        # would have succeeded fine. A first cut at "retry the whole crashed subset together" measured
        # WRONG: if the poisonous theta is still mixed into the retry batch, the retry breaks again and
        # collaterally re-fails the same innocents a second time (confirmed live: 12/12 collateral items
        # stayed WORKER_CRASH after a batched retry, 0 recovered). Fix: phase 2 isolates crashed items ONE
        # AT A TIME on a freshly-rebuilt pool -- a poisonous theta then only ever takes itself down, and an
        # innocent bystander gets a real result once it is no longer sharing a pool lifetime with the
        # poison. Cost is O(n_crashed) extra serial round-trips, paid only on the rare crash path.
        executor = self._get_executor()
        futmap = {executor.submit(_worker_eval, thetas[i]): i for i in miss_idx}
        crashed = []
        for fut in as_completed(futmap):
            i = futmap[fut]
            try:
                rec = fut.result()
                if rec["error"]:
                    self.stats["n_errors"] += 1
                rec["cache_hit"] = False
                self.stats["n_evaluated"] += 1
                self._checkpoint(keys[i], rec)
                results[i] = rec
            except (BrokenProcessPool, Exception):
                crashed.append(i)

        if crashed:
            self.stats["n_crash_retries_attempted"] += len(crashed)
            self._executor = None  # phase-1 pool is dead; force a rebuild before phase 2
            for i in crashed:
                executor = self._get_executor()  # rebuilds iff the previous isolated retry killed it too
                fut = executor.submit(_worker_eval, thetas[i])
                try:
                    rec = fut.result()
                    if rec["error"]:
                        self.stats["n_errors"] += 1
                    self.stats["n_crash_retries_recovered"] += 1
                except (BrokenProcessPool, Exception) as e:
                    rec = _penalized(self.part, thetas[i], f"WORKER_CRASH: reproduced in isolation "
                                     f"(single-item retry) -- genuinely poisonous theta, not collateral "
                                     f"({type(e).__name__}: {e})")
                    self.stats["n_worker_crashes"] += 1
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    self._executor = None
                rec["cache_hit"] = False
                self.stats["n_evaluated"] += 1
                self._checkpoint(keys[i], rec)
                results[i] = rec
        return results  # type: ignore[return-value]

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------------------------- CLI
def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module_path")
    ap.add_argument("factory_attr", nargs="?", default="make_part")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    harness = PartHarness(args.module_path, args.factory_attr, cache_path=args.cache, n_workers=args.workers)
    rng = random.Random(args.seed)
    thetas = [harness.part.sample_theta(rng) for _ in range(args.n)]
    t0 = time.time()
    results = harness.evaluate_batch(thetas)
    dt = time.time() - t0
    for r in results:
        print(json.dumps({"theta": r["theta"], "objectives": r["objectives"],
                          "constraints": r["constraints"], "feasible": r["feasible"],
                          "error": r["error"], "cache_hit": r["cache_hit"]}))
    print(f"-- {len(thetas)} evals in {dt:.3f}s -> {len(thetas)/dt*60:.0f} evals/min "
          f"(workers={args.workers}) stats={harness.stats}", file=sys.stderr)
    harness.close()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
