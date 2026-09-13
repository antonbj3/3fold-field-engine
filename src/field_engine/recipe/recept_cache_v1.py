#!/usr/bin/env python3
"""recept_cache_v1.py -- content-addressed cache for recipe prefixes, so re-running a recipe after an
edit only rebuilds the ops that actually changed.

Each op is fingerprinted over its own canonical JSON plus the fingerprint of every op before it, so a
fingerprint identifies a whole prefix of the chain, not a single step. The solid produced at each
step is stored on disk in the OCC binary BRep format (BinTools) under `cache_dir`, and the most
recent entries are also kept in a bounded process-local memory tier.

API:
    exec_ops_cached(ops, params=None, cache_dir=..., log_path=None) -> same shape as
        cad_op_exec_v1.exec_ops, plus per-op cache_hit flags and counts
    fingerprint_ops(ops, params=None) -> list[str]

A cache hit restores the stored solid and skips the build; a fingerprint mismatch rebuilds from the
first changed op onwards. Restoring a solid is a byte-for-byte round trip, but OCC enumerates edges
and faces of a restored shape in its own order, so a selector that addresses geometry by index (never
by the geometric selectors this chain uses) can resolve differently after a restore; that is recorded
rather than hidden.

Run the selftest with `python recept_cache_v1.py`; it prints a JSON report and exits non-zero if a
check fails.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import sys
import time

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(_KERNEL_DIR, "artifacts")
sys.path.insert(0, _KERNEL_DIR)

from cad_op_exec_v1 import DISPATCH, _base_volume_for, _MISSED_CUT_TOL_MM3  # noqa: E402 -- reused, not rewritten
import build123d as bd  # noqa: E402
import OCP  # noqa: E402
from build123d.topology import downcast  # noqa: E402
from build123d.importers import topods_lut  # noqa: E402
from OCP.BinTools import BinTools  # noqa: E402
from OCP.TopoDS import TopoDS_Shape  # noqa: E402

# Folded into the chain root (see MECHANISM above) -- a HIT across two different OCP/OCC builds is
# never trusted even if the fingerprint happens to coincide (R1 grind, plan doc §5).
_OCC_BUILD_VERSION = getattr(OCP, "__version__", "unknown")

# ============================================================================ 16B MEMORY TIER
# Process-local, in-front-of-disk. Key = (cache_dir, chain_key) -- namespaced the same way the
# disk tier is namespaced (two recipes stored under different cache_dir must never collide even
# on a chain-key hash coincidence). Value = (pid, shape) -- see module docstring MEMORY TIER for
# why no fingerprint is stored/checked here: no round trip occurs on a memory hit, so there is no
# channel for drift; the chain-key equality is itself the correctness proof WITHIN one process.
_MEM_CACHE_MAXSIZE = 300
_MEM_CACHE: "collections.OrderedDict[tuple, tuple[int, object]]" = collections.OrderedDict()


def _mem_get(cache_dir: str, key: str):
    """Returns (shape, cross_process_rejected: bool). NEVER trusts an entry stored under a
    different pid (AT6d) -- falls through exactly like a disk fingerprint_mismatch."""
    mkey = (cache_dir, key)
    entry = _MEM_CACHE.get(mkey)
    if entry is None:
        return None, False
    pid, shape = entry
    if pid != os.getpid():
        return None, True  # cross-process entry present but NEVER trusted
    _MEM_CACHE.move_to_end(mkey)  # LRU touch
    return shape, False


def _mem_put(cache_dir: str, key: str, shape) -> None:
    mkey = (cache_dir, key)
    _MEM_CACHE[mkey] = (os.getpid(), shape)
    _MEM_CACHE.move_to_end(mkey)
    while len(_MEM_CACHE) > _MEM_CACHE_MAXSIZE:
        _MEM_CACHE.popitem(last=False)  # evict least-recently-used


def _mem_clear(cache_dir: str | None = None) -> None:
    """Test-only: simulate 'no memory tier available' (e.g. a fresh process) for a given
    cache_dir, or all of it if cache_dir is None. Used by AT5b to force the disk-fingerprint
    path to run even though AT5b's run1/run2 share this same Python process (without this, the
    memory tier would shadow the disk sabotage entirely -- correct per the 16B contract, but it
    means AT5b must explicitly evict to exercise the disk tier it is testing)."""
    if cache_dir is None:
        _MEM_CACHE.clear()
        return
    for mkey in [k for k in _MEM_CACHE if k[0] == cache_dir]:
        del _MEM_CACHE[mkey]


def _canon(op: dict) -> str:
    return json.dumps(op, sort_keys=True, default=str)


def _chain_keys(ops: list) -> list:
    """key_i = H(key_{i-1} || op_sig_i) -- DECLARATION only (see module docstring). This single
    function IS the scoped-invalidation mechanism: two op-lists sharing a prefix produce
    byte-identical keys for that prefix, diverging only from the first changed op onward."""
    keys = []
    prev = "ROOT|occ=" + _OCC_BUILD_VERSION
    for op in ops:
        h = hashlib.sha256((prev + "|" + _canon(op)).encode()).hexdigest()[:24]
        keys.append(h)
        prev = h
    return keys


def _fingerprint(shape) -> dict:
    bb = shape.bounding_box()
    return {
        "n_faces": len(shape.faces()), "n_edges": len(shape.edges()), "n_vertices": len(shape.vertices()),
        "volume_mm3": round(shape.volume, 6),
        "bbox": [round(bb.min.X, 4), round(bb.min.Y, 4), round(bb.min.Z, 4),
                 round(bb.max.X, 4), round(bb.max.Y, 4), round(bb.max.Z, 4)],
    }


def _bin_to_shape(bin_path: str):
    """OCP.BinTools.Read_s -> raw TopoDS_Shape -> build123d object, same downcast+LUT path
    build123d's own import_step uses internally (build123d/importers.py topods_lut) -- no re-
    derivation of that mapping, just the SAME table, reused.

    MEASURED BUG FIXED: the FIRST version of this function returned `obj.solids()[0] if obj.solids()
    else obj` -- a shortcut copied from recept_cache_v1's OLD STEP _restore. That shortcut is WRONG
    whenever the stored shape is a multi-solid Compound (pattern_linear/pattern_circular/mirror all
    return bd.Compound(copies), i.e. MULTIPLE disjoint solids in ONE shape): .solids()[0] silently
    kept only ONE instance and discarded the rest, which would have propagated WRONG (truncated)
    geometry downstream had the fingerprint check not caught it -- reproduced live on a patterned
    recipe, where several ops came back with a fraction of their own volume. That is a DISTINCT root
    cause from the STEP volume-rounding drift the binary-BRep storage already closed. Also note build123d's OWN
    live (never-cached) op results are typically ALREADY Compound-wrapped even for a single solid
    (bd.Box(...).wrapped is a TopoDS_Compound) -- so unwrapping to a bare Solid was never even
    required for type-parity with the live path; it only ever removed data.
    Fix: return the reconstructed object AS-IS, whatever DISPATCH produced (Compound/Solid/Sketch/
    Wire) -- no truncation, ever.
    """
    raw = TopoDS_Shape()
    ok = BinTools.Read_s(raw, bin_path)
    if not ok:
        raise ValueError(f"BinTools.Read_s failed for {bin_path!r}")
    ds = downcast(raw)
    return topods_lut[type(ds)](ds)


def _restore(cache_dir, key):
    meta_p = os.path.join(cache_dir, key + ".json")
    bin_p = os.path.join(cache_dir, key + ".bin")
    if not (os.path.exists(meta_p) and os.path.exists(bin_p)):
        return None, False
    meta = json.load(open(meta_p))
    if meta.get("occ_build_version") != _OCC_BUILD_VERSION:
        return None, True  # different kernel build -- never trust a cross-build HIT (R1 grind)
    try:
        sol = _bin_to_shape(bin_p)
    except Exception:
        return None, True  # entry existed but is unreadable -- treat as a mismatch, never trust it
    fp = _fingerprint(sol)
    if fp != meta.get("fingerprint"):
        return None, True  # PRESENT but WRONG -- fingerprint_mismatch, never silently used
    return sol, False


def _store(cache_dir, key, shape):
    os.makedirs(cache_dir, exist_ok=True)
    bin_p = os.path.join(cache_dir, key + ".bin")
    meta_p = os.path.join(cache_dir, key + ".json")
    ok = BinTools.Write_s(shape.wrapped, bin_p)
    if not ok:
        raise ValueError(f"BinTools.Write_s failed for key {key!r}")
    json.dump({"fingerprint": _fingerprint(shape), "occ_build_version": _OCC_BUILD_VERSION},
               open(meta_p, "w"))


def exec_ops_cached(ops: list, params: dict | None = None, cache_dir: str | None = None,
                     log_path: str | None = None) -> dict:
    """Same op-dict contract as cad_op_exec_v1.exec_ops -- a chain-key cache wraps the SAME
    DISPATCH handlers (no reimplementation of any op's geometry). raises NotImplementedError
    for an uncovered op-type, identical contract to exec_ops (AT2d mirror)."""
    params = params or {}
    cache_dir = cache_dir or os.path.join(ARTIFACTS, "recept_cache_v1_store")
    keys = _chain_keys(ops)
    ctx: dict = {}
    log: list = []
    n_mem_hits = n_disk_hits = n_misses = n_mismatch = n_mem_rejected = 0
    for op, key in zip(ops, keys):
        op_id, op_type = op.get("id"), op.get("op")
        t0 = time.time()

        mem_shape, mem_rejected = _mem_get(cache_dir, key)
        if mem_rejected:
            n_mem_rejected += 1
        if mem_shape is not None:
            ctx[op_id] = mem_shape
            n_mem_hits += 1
            log.append({"op_id": op_id, "op_type": op_type, "cache": "HIT_MEM", "key": key,
                        "wall_ms": round((time.time() - t0) * 1000.0, 4), "status": "PASS"})
            continue

        restored, was_mismatch = _restore(cache_dir, key)
        if was_mismatch:
            n_mismatch += 1
        if restored is not None:
            ctx[op_id] = restored
            n_disk_hits += 1
            _mem_put(cache_dir, key, restored)  # promote: fingerprint already paid on THIS restore
            log.append({"op_id": op_id, "op_type": op_type, "cache": "HIT_DISK", "key": key,
                        "wall_ms": round((time.time() - t0) * 1000.0, 4), "status": "PASS"})
            continue

        handler = DISPATCH.get(op_type)
        if handler is None:
            raise NotImplementedError(op_type)
        vol_before = _base_volume_for(op_type, op, ctx)
        result = handler(op, ctx, params)
        vol_after = getattr(result, "volume", None) if result is not None else None
        delta = (vol_after - vol_before) if (vol_after is not None and vol_before is not None) else None
        status, reason = "PASS", None
        if op_type in ("boolean_cut", "hole") and delta is not None and abs(delta) < _MISSED_CUT_TOL_MM3:
            status, reason = "FAIL", "MISSED_CUT"
        if result is not None:
            ctx[op_id] = result
            try:
                _store(cache_dir, key, result)  # ONCE per built op -- disk write (fingerprint) + ...
                _mem_put(cache_dir, key, result)  # ... memory insert, never per probe
            except Exception:
                pass  # a store failure never corrupts the in-memory exec result (fail-safe, not fail-open)
        n_misses += 1
        log.append({"op_id": op_id, "op_type": op_type, "cache": "MISS", "key": key,
                    "wall_ms": round((time.time() - t0) * 1000.0, 4), "status": status, "reason": reason,
                    "volymdelta_mm3": delta})

    n_hits = n_mem_hits + n_disk_hits
    stats = {"cache_hits": n_hits, "cache_misses": n_misses, "fingerprint_mismatches": n_mismatch,
             "n_ops": len(ops), "chain_keys": keys, "mem_hits": n_mem_hits, "disk_hits": n_disk_hits,
             "mem_cross_process_rejected": n_mem_rejected}
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        json.dump({"log": log, "cache_stats": stats}, open(log_path, "w"), indent=1)
    return {"solids": ctx, "log": log, "cache_stats": stats}


# =================================================================================== SELFTEST
def _at5a_prefix_reuse() -> dict:
    """AT5a: run 2 -- box+2holes recipe, ONE LATE param changed (2nd hole diameter) -- the prefix
    (sketch+extrude+hole1) MUST cache-HIT; result MUST be round-trip-identical to a fresh
    (uncached) exec_ops() run with the SAME final ops."""
    from cad_op_exec_v1 import exec_ops

    tmp = os.path.join(ARTIFACTS, "_recept_cache_v1_at5a_store")
    shutil.rmtree(tmp, ignore_errors=True)

    def make_ops(hole2_d):
        return [
            {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": 30, "height": 30, "mode": "add"}]},
            {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": 10},
            {"op": "hole", "id": "h1", "target_ref": "box", "center": [-8, 0, 5], "diameter": 3, "axis": "Z", "through": True},
            {"op": "hole", "id": "h2", "target_ref": "h1", "center": [8, 0, 5], "diameter": hole2_d, "axis": "Z", "through": True},
        ]

    run1 = exec_ops_cached(make_ops(3.0), cache_dir=tmp)
    run2 = exec_ops_cached(make_ops(5.0), cache_dir=tmp)  # LATE param changed (hole2 diameter)
    prefix_hit = (run2["log"][0]["cache"].startswith("HIT") and run2["log"][1]["cache"].startswith("HIT")
                  and run2["log"][2]["cache"].startswith("HIT"))
    late_op_miss = run2["log"][3]["cache"] == "MISS"

    fresh = exec_ops(make_ops(5.0))  # UNCACHED reference, same final ops
    cached_vol = run2["solids"]["h2"].volume
    fresh_vol = fresh["solids"]["h2"].volume
    roundtrip_identical = abs(cached_vol - fresh_vol) < 1e-6

    return {"pass": bool(prefix_hit and late_op_miss and roundtrip_identical and run2["cache_stats"]["cache_hits"] >= 1),
            "cache_hits_run2": run2["cache_stats"]["cache_hits"], "prefix_hit": prefix_hit,
            "late_op_miss": late_op_miss, "cached_vol": cached_vol, "fresh_vol": fresh_vol,
            "roundtrip_identical": roundtrip_identical}


def _at5b_fingerprint_mismatch_fallbevis() -> dict:
    """AT5b: sabotage a stored cache entry's fingerprint on disk -- next run MUST detect the
    mismatch (fingerprint_mismatches>0), rebuild, and STILL produce correct final geometry.
    16B NOTE: run1 and run2 share this process, so run1's store also populates the MEMORY tier
    for these keys -- without an explicit _mem_clear(tmp), run2 would legitimately HIT_MEM (no
    round trip => nothing to mismatch) and never touch the sabotaged disk entry at all. That is
    CORRECT per the 16B contract (a memory hit has no channel for drift), but this test exists to
    prove the DISK tier's own protection is unchanged, so it evicts memory first to force the disk
    path it targets."""
    tmp = os.path.join(ARTIFACTS, "_recept_cache_v1_at5b_store")
    shutil.rmtree(tmp, ignore_errors=True)
    ops = [
        {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": 12, "height": 8, "mode": "add"}]},
        {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": 4},
    ]
    run1 = exec_ops_cached(ops, cache_dir=tmp)
    key = run1["cache_stats"]["chain_keys"][-1]
    _mem_clear(tmp)  # force the disk path (see docstring) -- this test targets the disk tier
    meta_p = os.path.join(tmp, key + ".json")
    meta = json.load(open(meta_p))
    meta["fingerprint"]["volume_mm3"] += 999.0  # SABOTAGE, constructed
    json.dump(meta, open(meta_p, "w"))

    run2 = exec_ops_cached(ops, cache_dir=tmp)
    mismatch_detected = run2["cache_stats"]["fingerprint_mismatches"] > 0
    rebuilt = run2["log"][-1]["cache"] == "MISS"
    correct_final = abs(run2["solids"]["box"].volume - 12 * 8 * 4) < 1e-6
    return {"pass": bool(mismatch_detected and rebuilt and correct_final),
            "fingerprint_mismatches": run2["cache_stats"]["fingerprint_mismatches"],
            "rebuilt": rebuilt, "correct_final_volume": run2["solids"]["box"].volume}


def _synthetic_ops_prefix() -> list[dict]:
    """A synthetic multi-op recipe long enough to exercise the cache the way a real part does: a
    bracket plate, a boss unioned onto it, six through holes and two pockets cut from it, ending in a
    chamfer on the boss bore. Returns the op list; every op references the previous step by id, so the whole list is one
    chain."""
    ops = [
        {"op": "sketch_2d", "id": "sk_base",
         "shapes": [{"type": "rectangle", "width": 120, "height": 80, "mode": "add"}]},
        {"op": "extrude", "id": "base", "sketch_ref": "sk_base", "amount": 10},
        {"op": "sketch_2d", "id": "sk_boss",
         "plane": {"origin": [0, 0, 10], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]},
         "shapes": [{"type": "rectangle", "width": 40, "height": 30, "mode": "add"}]},
        {"op": "extrude", "id": "boss", "sketch_ref": "sk_boss", "amount": 18},
        {"op": "boolean_union", "id": "body", "base_ref": "base", "tool_refs": ["boss"]},
    ]
    prev = "body"
    for k, (cx, cy) in enumerate([(-50, -30), (-50, 30), (50, -30), (50, 30)]):
        oid = f"bolt_hole_{k}"
        ops.append({"op": "hole", "id": oid, "target_ref": prev, "center": [cx, cy, 5],
                    "diameter": 6.0, "axis": "Z", "through": True})
        prev = oid
    ops.append({"op": "hole", "id": "boss_bore", "target_ref": prev, "center": [0, 0, 14],
                "diameter": 12.0, "axis": "Z", "through": True})
    prev = "boss_bore"
    for k, (cx, w, h) in enumerate([(-25.0, 24, 16), (25.0, 24, 16)]):
        ops.append({"op": "sketch_2d", "id": f"sk_pocket_{k}",
                    "plane": {"origin": [cx, 0, 4], "x_dir": [1, 0, 0], "z_dir": [0, 0, 1]},
                    "shapes": [{"type": "rectangle", "width": w, "height": h, "mode": "add"}]})
        ops.append({"op": "extrude", "id": f"pocket_tool_{k}", "sketch_ref": f"sk_pocket_{k}",
                    "amount": 8})
        ops.append({"op": "boolean_cut", "id": f"pocket_{k}", "base_ref": prev,
                    "tool_refs": [f"pocket_tool_{k}"]})
        prev = f"pocket_{k}"
    ops.append({"op": "chamfer", "id": "bore_break", "target_ref": prev, "distance": 0.5,
                "selector": {"entity": "edge", "geometry_type": "CIRCLE", "radius": 6.0,
                             "expected_count": 2}})
    return ops


def _at5c_synthetic_full_hit() -> dict:
    """AT5c: an identical re-run of the full synthetic prefix must give cache_hits == n_ops and
    fingerprint_mismatches == 0 on the second run. This is the fidelity gap the binary-BRep storage
    closes: with a STEP round trip the re-imported volume drifted past the fingerprint's own rounding
    tolerance, so a majority of the ops reported a fingerprint mismatch on an identical re-run."""
    tmp = os.path.join(ARTIFACTS, "_recept_cache_v1_at5c_store")
    shutil.rmtree(tmp, ignore_errors=True)
    ops = _synthetic_ops_prefix()

    t0 = time.time()
    run1 = exec_ops_cached(ops, cache_dir=tmp)
    t_cold_s = time.time() - t0
    t0 = time.time()
    run2 = exec_ops_cached(ops, cache_dir=tmp)
    t_warm_s = time.time() - t0

    stats2 = run2["cache_stats"]
    all_hit = stats2["cache_hits"] == len(ops)
    zero_mismatch = stats2["fingerprint_mismatches"] == 0
    final_id = ops[-1]["id"]
    vol_cold = run1["solids"][final_id].volume
    vol_warm = run2["solids"][final_id].volume
    bit_identical = vol_cold == vol_warm
    return {"pass": bool(all_hit and zero_mismatch and bit_identical),
            "n_ops": len(ops), "cache_hits_run2": stats2["cache_hits"],
            "fingerprint_mismatches_run2": stats2["fingerprint_mismatches"],
            "t_cold_s": round(t_cold_s, 3), "t_warm_s": round(t_warm_s, 3),
            "speedup_x": round(t_cold_s / t_warm_s, 1) if t_warm_s > 0 else None,
            "vol_cold_mm3": vol_cold, "vol_warm_mm3": vol_warm, "bit_identical": bit_identical}


def _at5d_counterfactual_dirty_scope() -> dict:
    """AT5d, the counterfactual: change ONE parameter in op j of the synthetic recipe and ops 1..j-1
    must HIT (byte-identical prefix) while ops j..N must MISS, exactly -- not approximately. A cache
    that hits past the changed op is fail-open, which is as dangerous as one that never hits."""
    tmp = os.path.join(ARTIFACTS, "_recept_cache_v1_at5d_store")
    shutil.rmtree(tmp, ignore_errors=True)
    ops = _synthetic_ops_prefix()
    exec_ops_cached(ops, cache_dir=tmp)  # warm the cache with the nominal recipe

    # perturb ONE parameter mid-recipe: a 'hole' op with downstream consumers, not the first or
    # last op
    hole_indices = [i for i, op in enumerate(ops) if op.get("op") == "hole"]
    j = hole_indices[len(hole_indices) // 2]
    ops2 = [dict(o) for o in ops]
    ops2[j] = dict(ops2[j], diameter=ops2[j]["diameter"] + 1.0)

    run2 = exec_ops_cached(ops2, cache_dir=tmp)
    log2 = run2["log"]
    prefix_all_hit = all(row["cache"].startswith("HIT") for row in log2[:j])
    diverging_all_miss = all(row["cache"] == "MISS" for row in log2[j:])
    return {"pass": bool(prefix_all_hit and diverging_all_miss), "diverging_index": j,
            "n_ops": len(ops2), "prefix_hits_expected": j, "prefix_all_hit": prefix_all_hit,
            "diverging_all_miss": diverging_all_miss,
            "cache_hits_run2": run2["cache_stats"]["cache_hits"]}


def _at6a_mem_hit_same_process() -> dict:
    """AT6a: an identical re-run of the synthetic prefix IN THE SAME PROCESS must
    be satisfied ENTIRELY by the memory tier the second time -- mem_hits == n_ops, disk_hits == 0,
    fingerprint_mismatches == 0 (no round trip occurred, so the disk fingerprint path is never
    even entered). This is the mechanism behind the probe-then-bake double-verification
    disappearing (16_inkrementalitet.md §4 arligt_negativt_fynd)."""
    tmp = os.path.join(ARTIFACTS, "_recept_cache_v1_at6a_store")
    shutil.rmtree(tmp, ignore_errors=True)
    _mem_clear(tmp)
    ops = _synthetic_ops_prefix()

    t0 = time.time()
    run1 = exec_ops_cached(ops, cache_dir=tmp)
    t_cold_s = time.time() - t0
    t0 = time.time()
    run2 = exec_ops_cached(ops, cache_dir=tmp)  # SAME process -- must be a pure memory replay
    t_mem_s = time.time() - t0

    stats2 = run2["cache_stats"]
    all_mem_hit = stats2["mem_hits"] == len(ops)
    zero_disk_hit = stats2["disk_hits"] == 0
    zero_mismatch = stats2["fingerprint_mismatches"] == 0
    final_id = ops[-1]["id"]
    bit_identical = run1["solids"][final_id].volume == run2["solids"][final_id].volume
    return {"pass": bool(all_mem_hit and zero_disk_hit and zero_mismatch and bit_identical),
            "n_ops": len(ops), "mem_hits_run2": stats2["mem_hits"], "disk_hits_run2": stats2["disk_hits"],
            "fingerprint_mismatches_run2": stats2["fingerprint_mismatches"],
            "t_cold_s": round(t_cold_s, 3), "t_mem_s": round(t_mem_s, 3),
            "speedup_x": round(t_cold_s / t_mem_s, 1) if t_mem_s > 0 else None,
            "bit_identical": bit_identical}


def _at6b_cross_process_trap_fallbevis() -> dict:
    """AT6b (planted-fault test): forge a memory entry under a DIFFERENT pid for a
    real chain-key -- the next lookup for that key MUST reject it (mem_cross_process_rejected>0),
    fall through to disk/rebuild, and STILL produce correct geometry. Proves the pid guard is
    load-bearing, not decorative -- a memory-HIT across a process boundary must never happen."""
    tmp = os.path.join(ARTIFACTS, "_recept_cache_v1_at6b_store")
    shutil.rmtree(tmp, ignore_errors=True)
    _mem_clear(tmp)
    ops = [
        {"op": "sketch_2d", "id": "sk", "shapes": [{"type": "rectangle", "width": 20, "height": 15, "mode": "add"}]},
        {"op": "extrude", "id": "box", "sketch_ref": "sk", "amount": 6},
    ]
    run1 = exec_ops_cached(ops, cache_dir=tmp)  # populates disk + memory under THIS pid
    key = run1["cache_stats"]["chain_keys"][-1]

    # sabotage: overwrite the memory entry's pid to simulate a foreign process's leftover object
    # (a real cross-process read is impossible -- a Python dict cannot be read by another process
    # -- so this directly exercises the guard code path the way a fork/exec-inherited or shared-
    # memory bug hypothetically could, which is exactly what the pid check defends against).
    mkey = (tmp, key)
    pid_real, shape_real = _MEM_CACHE[mkey]
    _MEM_CACHE[mkey] = (pid_real + 999999, shape_real)  # foreign pid, constructed

    run2 = exec_ops_cached(ops, cache_dir=tmp)
    rejected = run2["cache_stats"]["mem_cross_process_rejected"] > 0
    never_used_as_hit = run2["log"][-1]["cache"] in ("HIT_DISK", "MISS")  # never HIT_MEM off the forged entry
    correct_final = abs(run2["solids"]["box"].volume - 20 * 15 * 6) < 1e-6
    return {"pass": bool(rejected and never_used_as_hit and correct_final),
            "mem_cross_process_rejected": run2["cache_stats"]["mem_cross_process_rejected"],
            "final_cache_row": run2["log"][-1]["cache"], "correct_final_volume": run2["solids"]["box"].volume}


def _at6c_lru_bound() -> dict:
    """AT6c: the memory tier is bounded (_MEM_CACHE_MAXSIZE) -- inserting more distinct keys than
    the bound must never grow the dict past it (an unbounded process-local cache is a slow memory
    leak across a long-running interactive session, the exact opposite of 16B's intent)."""
    tmp = os.path.join(ARTIFACTS, "_recept_cache_v1_at6c_store")
    _mem_clear(tmp)
    n_extra = 50
    class _FakeShape:  # noqa: N801 -- test-local stub, no OCC handle needed for an LRU-only check
        volume = 1.0
    for i in range(_MEM_CACHE_MAXSIZE + n_extra):
        _mem_put(tmp, f"fake_key_{i}", _FakeShape())
    bounded = len(_MEM_CACHE) <= _MEM_CACHE_MAXSIZE
    newest_present, _ = _mem_get(tmp, f"fake_key_{_MEM_CACHE_MAXSIZE + n_extra - 1}")
    oldest_evicted, _ = _mem_get(tmp, "fake_key_0")
    _mem_clear(tmp)
    return {"pass": bool(bounded and newest_present is not None and oldest_evicted is None),
            "mem_cache_len": len(_MEM_CACHE), "maxsize": _MEM_CACHE_MAXSIZE,
            "newest_present": newest_present is not None, "oldest_evicted": oldest_evicted is None}


def _selftest() -> dict:
    out = {"AT5a_prefix_reuse": _at5a_prefix_reuse(), "AT5b_fingerprint_mismatch": _at5b_fingerprint_mismatch_fallbevis()}
    try:
        out["AT5c_synthetic_full_hit"] = _at5c_synthetic_full_hit()
        out["AT5d_counterfactual_dirty_scope"] = _at5d_counterfactual_dirty_scope()
        out["AT6a_mem_hit_same_process"] = _at6a_mem_hit_same_process()
        out["AT6b_cross_process_trap_fallbevis"] = _at6b_cross_process_trap_fallbevis()
        out["AT6c_lru_bound"] = _at6c_lru_bound()
    except Exception as e:  # noqa: BLE001 -- record, don't crash AT5a/b's result
        out["AT5c_synthetic_full_hit"] = out.get("AT5c_synthetic_full_hit", {"pass": False, "error": f"{type(e).__name__}:{e}"})
        out["AT5d_counterfactual_dirty_scope"] = out.get("AT5d_counterfactual_dirty_scope", {"pass": False, "error": "skipped after error"})
        out["AT6a_mem_hit_same_process"] = out.get("AT6a_mem_hit_same_process", {"pass": False, "error": "skipped after error"})
        out["AT6b_cross_process_trap_fallbevis"] = out.get("AT6b_cross_process_trap_fallbevis", {"pass": False, "error": "skipped after error"})
        out["AT6c_lru_bound"] = out.get("AT6c_lru_bound", {"pass": False, "error": "skipped after error"})
    out["all_pass"] = bool(
        out["AT5a_prefix_reuse"]["pass"] and out["AT5b_fingerprint_mismatch"]["pass"]
        and out["AT5c_synthetic_full_hit"]["pass"] and out["AT5d_counterfactual_dirty_scope"]["pass"]
        and out["AT6a_mem_hit_same_process"]["pass"] and out["AT6b_cross_process_trap_fallbevis"]["pass"]
        and out["AT6c_lru_bound"]["pass"]
    )
    return out


if __name__ == "__main__":
    r = _selftest()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["all_pass"] else 1)
