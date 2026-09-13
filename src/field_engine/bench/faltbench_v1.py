#!/usr/bin/env python3
"""Benchmark suite for the field engine: one run = every component, then a gate.

Each component runs in its own subprocess (nice 15, at most four CPU threads) and writes one JSON to
bench/artifacts/_parts/. This driver aggregates those files into one result JSON, appends one row to
a trend file and applies a gate.

Per component the family measures (a) throughput, (b) p50/p95 latency at declared sizes S/M/L,
(c) peak RSS and VRAM, (d) scaling 1/2/4 threads or processes where the CPU stage scales and device
against CPU where both exist, (e) correctness against a closed-form reference (sphere SDF, box minus
a through cylinder -- the same reference for the field kernel and for the B-rep side), (f)
determinism (two runs bit-identical), (g) end to end on one part.

The form is deliberate: thresholds in THRESH are pre-registered, set before the run and never
adjusted to turn a gate green; the baseline is pinned and is never written by a gate run, only by
`--rebaseline --reason "..."`, because a gate that takes its own output as input is always green;
the status per gate is PASS / FAIL (correctness, determinism) / FAIL_REGRESSION (against the pinned
baseline) / INGEN_BASELINE, and the process exits 1 on FAIL.

Run:  python src/field_engine/bench/faltbench_v1.py --alla          the whole family
      python src/field_engine/bench/faltbench_v1.py --rok           short round in --snabb sizes
      python src/field_engine/bench/faltbench_v1.py --bara faltkarna,ikarus
      python src/field_engine/bench/faltbench_v1.py --hoppa occ_cad
      python src/field_engine/bench/faltbench_v1.py --ingen-korning aggregate the existing parts
      python src/field_engine/bench/faltbench_v1.py --alla --rebaseline --reason "..."
Output: bench/artifacts/faltbench_<date>[_mode].json, bench/artifacts/faltbench_trend_v1.jsonl
        (append), bench/artifacts/faltbench_baseline_v1.json (only with --rebaseline).
Sizes marked L need a CUDA device; without one the components record them as CUDA-ONLY and the
remaining sizes are measured on the Warp CPU backend.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "scripts"))
from field_paths import field_path
import _bench_common as BC  # noqa: E402

OUT_DIR = BC.ARTIFACTS
TREND = os.path.join(OUT_DIR, "faltbench_trend_v1.jsonl")
BASELINE = os.path.join(OUT_DIR, "faltbench_baseline_v1.json")
ROK_KOMPONENTER = ["importprobe", "faltkarna", "ikarus", "occ_cad", "e2e"]

# ---- PRE-REGISTERED thresholds (set before the first measured run, never adjusted to be green) ----
THRESH = dict(
    genomstromning_regress_slack_pct=15.0,   # throughput may be at most 15 % worse than baseline
    latens_p50_regress_slack_pct=20.0,       # p50 latency per size may be at most 20 % slower
    rss_regress_slack_pct=30.0,              # peak RSS per component may grow at most 30 %
    korrekthet_alla_grona=True,              # every correctness row must be green
    determinism_alla_bitidentiska=True,      # every determinism row must be True, except the one below
    determinism_forvantat_falsk=("faltkarna/determinism/reduktion_f32_atomic_bit_identisk",),
    # the legacy signing path is carried as the before column of the solid-filling measurement; it is
    # expected red and is not the shipped method, so it is excluded from the correctness gate by name
    korrekthet_forvantat_rod=("skal_floodfill",),
)

# (part name, script, timeout_s)
KOMPONENTER = [
    ("importprobe", "_komp_importprobe.py", 900),
    ("faltkarna", "_komp_faltkarna.py", 1800),
    ("mesh_sdf", "_komp_mesh_sdf.py", 1800),
    ("ikarus", "_komp_ikarus.py", 900),
    ("occ_cad", "_komp_occ.py", 1200),
    ("e2e", "_komp_e2e.py", 1200),
]

# the declared headline throughput per component: (sub-measurement, list key, value key, unit)
GENOMSTROMNING = {
    "faltkarna": ("analytiskt_testfalt_SML", "storlekar", "voxlar_per_s_v2", "voxels/s"),
    "mesh_sdf": ("storlekar_SML", "rader", "celler_per_s", "cells/s"),
    "ikarus": ("storlekar_SML", "rader", "punkter_per_s", "points/s"),
    "occ_cad": ("storlekar_step_tessellering", "rader", "trianglar_per_s", "triangles/s"),
}
LATENS = {
    "faltkarna": ("analytiskt_testfalt_SML", "storlekar", "v2_resident_p50_s", "v2_resident_p95_s"),
    "mesh_sdf": ("storlekar_SML", "rader", "p50_s", "p95_s"),
    "ikarus": ("storlekar_SML", "rader", "p50_s", "p95_s"),
}


def kor_komponent(namn, skript, timeout, snabb):
    os.makedirs(BC.PARTS_DIR, exist_ok=True)
    log = os.path.join(BC.PARTS_DIR, f"{namn}.log")
    cmd = ["nice", "-n", "15", sys.executable, str(field_path(BC.repo_paths()[1], "bench/"+skript))] + (["--snabb"] if snabb else [])
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "OPENBLAS_NUM_THREADS": "4"}
    t0 = time.time()
    print(f"[faltbench] {namn} starting ...", flush=True)
    with open(log, "w") as fh:
        try:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE, env=env, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
    wall = time.time() - t0
    print(f"[faltbench] {namn} done rc={rc} {wall:.0f}s", flush=True)
    return dict(namn=namn, rc=rc, wall_s=wall, log=os.path.relpath(log, OUT_DIR))


def las_parts():
    parts = {}
    for f in sorted(glob.glob(os.path.join(BC.PARTS_DIR, "*.json"))):
        try:
            d = json.load(open(f))
            parts[d["komponent"]] = d
        except Exception as e:  # noqa: BLE001
            parts[os.path.basename(f)] = dict(komponent=os.path.basename(f), status="OLASBAR",
                                              fel=str(e), delar={})
    return parts


def _samla(d, nyckel, prefix=""):
    """Every (path, value) pair for a key, at any depth in a part's JSON."""
    ut = []
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}/{k}" if prefix else str(k)
            if k == nyckel and not isinstance(v, (dict, list)):
                ut.append((prefix, v))
            else:
                ut.extend(_samla(v, nyckel, p))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            ut.extend(_samla(v, nyckel, f"{prefix}[{i}]"))
    return ut


def aggregera(parts, korningar, snabb):
    S = dict(datum=time.strftime("%Y-%m-%d %H:%M"), snabb=snabb, korningar=korningar,
             device=BC.device(), cpu_tradar_tak=4)
    S["status"] = {k: dict(status=v.get("status"), wall_total_s=v.get("wall_total_s")) for k, v in parts.items()}
    S["minne"] = {k: dict(rss_peak_mb=v.get("rss_peak_mb")) for k, v in parts.items()}
    S["vram_delta_mb"] = {k: v.get("vram_process_mb") for k, v in parts.items() if v.get("vram_process_mb")}

    gen, lat = [], {}
    for komp, (delnamn, listkey, valkey, enhet) in GENOMSTROMNING.items():
        rader = (parts.get(komp, {}).get("delar", {}).get(delnamn, {}) or {}).get(listkey, [])
        rader = [r for r in rader if r.get(valkey) is not None]
        if rader:
            r = rader[-1]
            gen.append(dict(komponent=komp, varde=r[valkey], enhet=enhet, storlek=r.get("label")))
    for komp, (delnamn, listkey, p50, p95) in LATENS.items():
        rader = (parts.get(komp, {}).get("delar", {}).get(delnamn, {}) or {}).get(listkey, [])
        lat[komp] = {r.get("label"): [r.get(p50), r.get(p95)] for r in rader if r.get(p50) is not None}
    S["genomstromning"] = gen
    S["latens_p50_p95_s"] = lat

    korr, det = [], {}
    for komp, d in parts.items():
        for path, v in _samla(d.get("delar", {}), "GRON"):
            relf = dict(_samla(d.get("delar", {}), "rel_fel")).get(path)
            korr.append(dict(test=f"{komp}/{path}", GRON=bool(v), rel_fel=relf))
        for nyckel in ("bit_identisk", "reduktion_f32_atomic_bit_identisk",
                       "reduktion_i64_fixpunkt_bit_identisk", "GRON_bitidentisk"):
            for path, v in _samla(d.get("delar", {}), nyckel):
                det[f"{komp}/{path}/{nyckel}" if nyckel != "bit_identisk" else f"{komp}/{path}"] = bool(v)
    S["korrekthet"] = korr
    S["determinism"] = det

    S["tradskalning_speedup"] = {}
    for komp, d in parts.items():
        for delnamn, dd in (d.get("delar") or {}).items():
            for r in (dd.get("rader") or []) if isinstance(dd, dict) else []:
                if isinstance(r, dict) and r.get("n_proc") == 4:
                    S["tradskalning_speedup"][f"{komp}/{delnamn}"] = r.get("speedup_vs_1")
                if isinstance(r, dict) and r.get("tradskalning"):
                    S["tradskalning_speedup"][f"{komp}/{delnamn}/{r.get('label')}"] = r["tradskalning"][-1].get("speedup_vs_1")
    S["gpu_vs_cpu_speedup"] = {f"{k}/{r.get('label', r.get('n_punkter'))}": r.get("gpu_speedup")
                               for k, d in parts.items()
                               for r in ((d.get("delar", {}).get("gpu_vs_cpu", {}) or {}).get("rader") or [])}
    S["e2e"] = (parts.get("e2e", {}).get("delar", {}) or {}).get("kedja_brep_mesh_sdf_falt_mesh")
    S["broken_saknas"] = [dict(komponent=k, del_=dn, status=dd.get("status"))
                          for k, d in parts.items() for dn, dd in (d.get("delar") or {}).items()
                          if isinstance(dd, dict) and dd.get("status") not in ("OK", None)]
    alla = [(f"{k}/{dn}", dd.get("wall_s", 0.0)) for k, d in parts.items()
            for dn, dd in (d.get("delar") or {}).items() if isinstance(dd, dict)]
    S["flaskhalsar_topp5"] = [dict(var=n, wall_s=w) for n, w in sorted(alla, key=lambda x: -x[1])[:5]]
    return S


def trend_rad(S, lage):
    """One row per run: the same keys every time, so runs are comparable over time."""
    return dict(datum=S["datum"], lage=lage, device=S["device"],
                genomstromning={t["komponent"]: t["varde"] for t in S["genomstromning"]},
                latens_p50_p95_s={k: {sz: dict(p50=v[0], p95=v[1]) for sz, v in d.items()}
                                  for k, d in S["latens_p50_p95_s"].items() if d},
                minne_rss_mb={k: v["rss_peak_mb"] for k, v in S["minne"].items() if v.get("rss_peak_mb")},
                korrekthet={k["test"]: dict(rel_fel=k["rel_fel"], GRON=k["GRON"]) for k in S["korrekthet"]},
                determinism=S["determinism"], gpu_vs_cpu=S["gpu_vs_cpu_speedup"],
                tradskalning=S["tradskalning_speedup"],
                e2e_total_s=(S["e2e"] or {}).get("total_s"), e2e_flaskhals=(S["e2e"] or {}).get("flaskhals"),
                flaskhalsar_topp5=[f["var"] for f in S["flaskhalsar_topp5"]],
                broken=[f"{b['komponent']}/{b['del_']}:{b['status']}" for b in S["broken_saknas"]],
                status_komponenter={k: v["status"] for k, v in S["status"].items()},
                korningstid_s={k["namn"]: k["wall_s"] for k in S["korningar"]},
                korningstid_total_s=sum(k["wall_s"] for k in S["korningar"]))


def las_trend(lage):
    rows = []
    if os.path.exists(TREND):
        for line in open(TREND):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if r.get("lage") == lage:
                        rows.append(r)
                except Exception:  # noqa: BLE001
                    pass
    return rows


def grind(S, rad, baseline):
    """PASS / FAIL (absolute: correctness, determinism) / FAIL_REGRESSION (against the pinned
    baseline) / INGEN_BASELINE. The baseline is never written here."""
    g = dict(trosklar=THRESH, resultat={}, regressioner=[], fail=[])
    kor_roda = [k["test"] for k in S["korrekthet"]
                if not k["GRON"] and not any(x in k["test"] for x in THRESH["korrekthet_forvantat_rod"])]
    det_roda = [k for k, v in S["determinism"].items()
                if not v and not any(k.endswith(x) or x in k for x in THRESH["determinism_forvantat_falsk"])]
    g["resultat"]["korrekthet"] = dict(status="PASS" if not kor_roda else "FAIL", roda=kor_roda, n=len(S["korrekthet"]))
    g["resultat"]["determinism"] = dict(status="PASS" if not det_roda else "FAIL", roda=det_roda, n=len(S["determinism"]))
    g["fail"] = kor_roda + det_roda
    if not baseline:
        g["resultat"]["regression"] = dict(status="INGEN_BASELINE",
                                           not_="run --rebaseline --reason '...' to pin this run")
    else:
        b = baseline.get("rad", {})
        for k, v in rad["genomstromning"].items():
            bv = b.get("genomstromning", {}).get(k)
            if bv and v < bv * (1 - THRESH["genomstromning_regress_slack_pct"] / 100):
                g["regressioner"].append(dict(matt="genomstromning", komponent=k, nu=v, baseline=bv,
                                              forandring_pct=100 * (v / bv - 1)))
        for k, v in rad["latens_p50_p95_s"].items():
            for sz, pv in v.items():
                bv = b.get("latens_p50_p95_s", {}).get(k, {}).get(sz, {}).get("p50")
                if bv and pv["p50"] > bv * (1 + THRESH["latens_p50_regress_slack_pct"] / 100):
                    g["regressioner"].append(dict(matt=f"latens p50 {sz}", komponent=k, nu=pv["p50"],
                                                  baseline=bv, forandring_pct=100 * (pv["p50"] / bv - 1)))
        for k, v in rad["minne_rss_mb"].items():
            bv = b.get("minne_rss_mb", {}).get(k)
            if bv and v > bv * (1 + THRESH["rss_regress_slack_pct"] / 100):
                g["regressioner"].append(dict(matt="rss_peak_mb", komponent=k, nu=v, baseline=bv,
                                              forandring_pct=100 * (v / bv - 1)))
        g["resultat"]["regression"] = dict(status="PASS" if not g["regressioner"] else "FAIL_REGRESSION",
                                           n=len(g["regressioner"]), baseline_datum=baseline.get("datum"),
                                           baseline_reason=baseline.get("reason"))
    st = [r["status"] for r in g["resultat"].values()]
    g["status"] = ("FAIL" if "FAIL" in st else
                   ("FAIL_REGRESSION" if "FAIL_REGRESSION" in st else
                    ("INGEN_BASELINE" if "INGEN_BASELINE" in st else "PASS")))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alla", action="store_true")
    ap.add_argument("--rok", action="store_true")
    ap.add_argument("--snabb", action="store_true")
    ap.add_argument("--bara", default=None)
    ap.add_argument("--hoppa", default=None)
    ap.add_argument("--ingen-korning", action="store_true")
    ap.add_argument("--rebaseline", action="store_true")
    ap.add_argument("--reason", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    snabb = a.snabb or a.rok
    valda = [k for k in KOMPONENTER]
    if a.rok:
        valda = [k for k in valda if k[0] in ROK_KOMPONENTER]
    if a.bara:
        namn = set(a.bara.split(","))
        valda = [k for k in valda if k[0] in namn]
    if a.hoppa:
        namn = set(a.hoppa.split(","))
        valda = [k for k in valda if k[0] not in namn]

    os.makedirs(OUT_DIR, exist_ok=True)
    korningar = []
    if not a.ingen_korning:
        for namn, skript, timeout in valda:
            korningar.append(kor_komponent(namn, skript, timeout, snabb))

    parts = las_parts()
    S = aggregera(parts, korningar, snabb)
    lage = "snabb" if snabb else "full"
    rad = trend_rad(S, lage)
    baseline = json.load(open(BASELINE)) if os.path.exists(BASELINE) else None
    S["grind"] = grind(S, rad, baseline)

    out = a.out or os.path.join(OUT_DIR, f"faltbench_{time.strftime('%Y-%m-%d')}"
                                         f"{'_' + lage if lage != 'full' else ''}.json")
    with open(out, "w") as fh:
        json.dump(S, fh, indent=1, default=BC._json_default)
    with open(TREND, "a") as fh:
        fh.write(json.dumps(rad, default=BC._json_default) + "\n")

    if a.rebaseline:
        if not a.reason:
            print("--rebaseline requires --reason", file=sys.stderr)
            return 2
        with open(BASELINE, "w") as fh:
            json.dump(dict(datum=S["datum"], reason=a.reason, rad=rad), fh, indent=1, default=BC._json_default)
        print(f"[faltbench] pinned baseline: {BASELINE} ({a.reason})")

    print(json.dumps(dict(datum=S["datum"], lage=lage, device=S["device"],
                          genomstromning=rad["genomstromning"], grind=S["grind"]["status"],
                          fail=S["grind"]["fail"], n_korrekthet=len(S["korrekthet"]),
                          n_determinism=len(S["determinism"]),
                          broken=rad["broken"], n_trend_rader=len(las_trend(lage))), indent=1))
    print("wrote", out)
    return 1 if S["grind"]["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
