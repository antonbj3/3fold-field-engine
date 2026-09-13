#!/usr/bin/env python3
"""Benchmark component: modules that do not run.

Imports every shipped module in a separate subprocess (timeout 120 s) and records OK/BROKEN plus the
error line and the import time. Every module is __main__-guarded, so an import has no side effect
beyond the Warp initialisation. This is the cheapest measurement in the family and the one that
catches a module that stopped importing at all. Writes bench/artifacts/_parts/importprobe.json.
"""
import os
import subprocess
import sys
import time

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "src", "field_engine", "bench")
sys.path.insert(0, HERE)
import _bench_common as BC  # noqa: E402

SRC, ROOT = BC.repo_paths()

MODULER = [
    ("", "faltkarna_v1", "field kernel"),
    ("", "faltkarna_v1_g7_fix", "field kernel"),
    ("", "faltkarna_v1_mesh_to_sdf", "field kernel"),
    ("", "faltkarna_v1_svep_v2_loft", "field kernel / sweep-loft"),
    ("", "mesh_to_sdf_small_features_v1", "field kernel / small features"),
    ("", "tillverkningsfalt_v1_kernel", "manufacturing field"),
    ("", "lastfalt_v1_fem", "load field"),
    ("", "lastfalt_v1_topopt", "load field"),
    ("", "lastfalt_v1_solid", "load field"),
    ("", "lastfalt_v1_verify", "load field"),
    ("", "negativrum_v1", "void-first geometry"),
    ("", "fabriksnegativrum_v1", "void-first geometry"),
    ("", "brep_aag_v1", "B-rep"),
    ("", "brep_feature_igenkann_v1", "B-rep"),
    ("", "recept_v1", "recipe library"),
    ("", "recept_familjer_egna_v1", "recipe library"),
    ("", "duct_task_gen_v1", "task generator"),
    ("ikarus_v1", "expr", "IKARUS"),
    ("ikarus_v1", "eval_warp", "IKARUS"),
    ("ikarus_v1", "eval_warp_uniform_v2", "IKARUS"),
    ("ikarus_v1", "queries", "IKARUS"),
    ("ikarus_v1", "cost_model_v1_2", "IKARUS"),
    ("ikarus_v1", "render_v2", "IKARUS"),
    ("ikarus_v1", "proofs", "IKARUS"),
    ("ikarus_v1", "lipschitz_spine_v1", "IKARUS"),
    ("loop", "recept_exec_v1", "closed loop"),
    ("loop", "ratata_loop_v1", "closed loop"),
    ("loop", "part_harness_v1", "closed loop"),
    ("loop", "surrogat_v1", "closed loop"),
    ("opt", "diff_placement_search_v1", "placement search"),
    ("factory", "virt_factory_model_v0", "factory layout"),
    ("recipe", "cad_op_schema_v1", "op schema"),
    ("recipe", "cad_op_exec_v1", "op executor"),
    ("recipe", "geometri_selektor_v1", "op executor"),
    ("recipe", "iso_thread_table_v1", "op executor"),
    ("recipe", "recept_cache_v1", "op cache"),
    ("recipe", "formfeature_v1", "form features"),
    ("recipe", "formrib_v1", "form features"),
    ("recipe", "sketch_gcs_v1", "sketch solver"),
    ("recipe", "brep_recept_synth_v1", "recipe synthesiser"),
]


def del_importer():
    rows = []
    for sub, mod, del_ in MODULER:
        d = os.path.join(SRC, sub) if sub else SRC
        cmd = [sys.executable, "-c",
               f"import sys,time; sys.path.insert(0,{d!r}); sys.path.insert(0,{SRC!r}); "
               f"t=time.time(); import {mod}; print('IMPORT_OK', round(time.time()-t,2))"]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=d,
                               env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"})
            ok = r.returncode == 0 and "IMPORT_OK" in r.stdout
            fel = None if ok else (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"rc={r.returncode}")
        except subprocess.TimeoutExpired:
            ok, fel = False, "TIMEOUT 120s"
        rows.append(dict(modul=os.path.join(sub, mod + ".py") if sub else mod + ".py", del_=del_,
                         kor=ok, fel=fel, t_s=time.time() - t0,
                         finns=os.path.exists(os.path.join(d, mod + ".py"))))
    return dict(n=len(rows), n_kor=sum(r["kor"] for r in rows), n_broken=sum(not r["kor"] for r in rows),
                rader=rows, GRON=all(r["kor"] for r in rows))


if __name__ == "__main__":
    P = BC.Part("importprobe")
    P.kor("import_alla_moduler", del_importer)
    P.skriv()
