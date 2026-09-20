#!/usr/bin/env python3
"""Route a duct by letting current find the room, with the distance field as the conductance.

The geodesic front already in this engine asks the shortest question: which way is nearest. A route
through a crowded space has a second property the shortest path does not see -- how much room it has
while it goes -- and a shortest path will happily thread a gap that is exactly wide enough.

The move here is to treat the free space as a RESISTOR NETWORK. Every pair of neighbouring free
cells gets a conductance from the Euclidean distance transform, ((EDT_u + EDT_v) / 2 / pitch)^1.5,
so wide corridors conduct and tight ones throttle. Injecting unit current at the inlet and drawing
it at the outlet and solving one Laplace system gives a potential whose edge currents mark every
corridor that carries flow, in proportion to how much room it has. The route is then the cheapest
path through that current field.

Two things fall out of the same solve, and the second one is the more useful:

  THE ROUTE            extracted from the current corridor, it prefers room over shortness. On the
                       11-task duct corpus it reaches the outlet in 11 of 11 cases, as does A*,
                       where the fixed two-segment centreline is infeasible in 8 of 11. On 6 of 11
                       tasks its minimum clearance is at least as good as A*'s, and where it is
                       better it is much better: +65.0 % on one task and +46.5 % on another.
  THE DIFFICULTY       the effective resistance between inlet and outlet is a single number for how
                       constricted the whole space is, and it correlates with the route's minimum
                       clearance at a Spearman rho of -0.8636 (Pearson -0.8315). The measured-hard
                       tasks average 0.02105 against 0.01275 for the easy ones, 65 % apart. That is
                       a property of the SPACE, available before any route is committed to, and it
                       is what this module is actually worth.

WHERE IT DOES NOT WIN. Routing several ducts at once through a factory hall does NOT beat routing
them one at a time along geodesics. At two routes the flow solution is 387.0 mm against 345.0 mm for
sequential geodesics, both with zero crossings, and it takes 1.43 s against 0.28 s. At four routes
both reach 813.0 mm, and the flow solution has ONE crossing where trying all 24 sequential orders has
none, for 30.25 s against 35.67 s. Simultaneity is not the advantage; the difficulty measure is.

A NUMBER THAT WAS CLAIMED AND IS NOT REPRODUCED. A summary of the measured run reported the flow
route costing 0.03 % less pressure drop than the geodesic one on the hardest task. Its own stored
result says the opposite sign: 2.5506 against 2.4794 in the judge's units, which is 2.87 % MORE. The
pressure-drop claim is withdrawn; the clearance and difficulty results above are the ones the data
supports. Two other summary figures (a 7-of-11 clearance count and a mean-length pair) likewise do
not match the stored per-task rows, and the rows are what is carried here.
"""
import heapq
import json
import os
import sys

import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import cg

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD_ENGINE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(FIELD_ENGINE))
for p in (HERE, FIELD_ENGINE, os.path.join(FIELD_ENGINE, "ikarus_v1")):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA = os.path.join(ROOT, "data", "laplaceflode_v1")
KORPUS = os.environ.get("FIELD_ENGINE_DUCT_CORPUS", os.path.join(ROOT, "data", "corpus", "duct_v1"))

P_KONDUKTANS = 1.5                 # conductance exponent on the clearance
GAMMA_STROM = 1.5                  # how sharply the path extraction prefers a high-current edge
KONDUKTANS_GOLV = 0.1              # a cell at the wall still conducts a little, or the graph splits


def bygg_gitterlaplacian(occ, edt_mm, pitch_mm, p_kond=P_KONDUKTANS):
    """The free space as a resistor network: one node per free cell, conductance from the clearance.

    Returns the node count, the node map, the edge endpoints, the conductances and the assembled
    sparse Laplacian."""
    fri = ~occ
    n = int(fri.sum())
    nodkarta = -np.ones(occ.shape, dtype=np.int32)
    nodkarta[fri] = np.arange(n, dtype=np.int32)

    u_lista, v_lista, c_lista = [], [], []
    for axel in range(occ.ndim):
        a = [slice(None)] * occ.ndim
        b = [slice(None)] * occ.ndim
        a[axel], b[axel] = slice(None, -1), slice(1, None)
        m = fri[tuple(a)] & fri[tuple(b)]
        u_lista.append(nodkarta[tuple(a)][m])
        v_lista.append(nodkarta[tuple(b)][m])
        c_lista.append(0.5 * (edt_mm[tuple(a)][m] + edt_mm[tuple(b)][m]) / pitch_mm)

    u = np.concatenate(u_lista)
    v = np.concatenate(v_lista)
    kond = (np.maximum(np.concatenate(c_lista), KONDUKTANS_GOLV) ** p_kond).astype(np.float64)
    rad = np.concatenate([u, v, u, v])
    kol = np.concatenate([v, u, u, v])
    varde = np.concatenate([-kond, -kond, kond, kond])
    L = sparse.coo_matrix((varde, (rad, kol)), shape=(n, n)).tocsr()
    return n, nodkarta, u, v, kond, L


def los_potential(L, n, s_nod, t_nod, tol=1e-10, maxiter=2000):
    """One Laplace solve: unit current in at the inlet, out at the outlet.

    The Laplacian is singular by one constant (a floating network has no absolute potential), so the
    solution is only defined up to an additive constant; the effective resistance, which is a
    DIFFERENCE of potentials, is not. That is the quantity read off here."""
    b = np.zeros(n)
    b[s_nod], b[t_nod] = 1.0, -1.0
    M = sparse.diags(1.0 / np.maximum(L.diagonal(), 1e-12))
    x, info = cg(L, b, rtol=tol, maxiter=maxiter, M=M)
    x = x - x.mean()
    return x, float(x[s_nod] - x[t_nod]), int(info)


def _grannlista(n, u, v):
    grannar = [[] for _ in range(n)]
    for e, (a, b) in enumerate(zip(u, v)):
        grannar[a].append((b, e))
        grannar[b].append((a, e))
    return grannar


def extrahera_strombana(grannar, koordinater, s_nod, t_nod, kantstrom, gamma=GAMMA_STROM):
    """Cheapest path when travelling along a high-current edge is cheap and against one is not."""
    i = np.abs(kantstrom)
    i_norm = i / max(float(i.max()) if len(i) else 1.0, 1e-12)
    avstand = {s_nod: 0.0}
    foreg = {}
    ko = [(0.0, s_nod)]
    besokta = set()
    while ko:
        d, nu = heapq.heappop(ko)
        if nu in besokta:
            continue
        besokta.add(nu)
        if nu == t_nod:
            break
        p_nu = koordinater[nu]
        for granne, e in grannar[nu]:
            if granne in besokta:
                continue
            geo = float(np.linalg.norm(p_nu - koordinater[granne]))
            d_ny = d + geo / (i_norm[e] ** gamma + 1e-4)
            if d_ny < avstand.get(granne, float("inf")):
                avstand[granne] = d_ny
                foreg[granne] = nu
                heapq.heappush(ko, (d_ny, granne))
    if t_nod not in foreg and s_nod != t_nod:
        return []
    bana, nu = [t_nod], t_nod
    while nu in foreg:
        nu = foreg[nu]
        bana.append(nu)
    bana.reverse()
    return [tuple(koordinater[k]) for k in bana]


def rutt_genom_flode(occ, pitch_mm, start, mal, min_r_mm):
    """The whole chain on one task: EDT, network, solve, extract. Returns the path, its minimum
    clearance, its length and the effective resistance."""
    fri = ~occ
    edt_mm = distance_transform_edt(fri) * pitch_mm
    n, nodkarta, u, v, kond, L = bygg_gitterlaplacian(occ, edt_mm, pitch_mm)
    s, t = int(nodkarta[start]), int(nodkarta[mal])
    pot, r_eff, _ = los_potential(L, n, s, t)
    koordinater = np.argwhere(fri)
    bana = extrahera_strombana(_grannlista(n, u, v), koordinater, s, t, kond * (pot[u] - pot[v]))
    if not bana:
        return {"bana": [], "clearance_mm": float("nan"), "langd_mm": float("nan"),
                "r_eff": r_eff, "marginal_mm": float("nan")}
    clr = float(min(edt_mm[p] for p in bana))
    langd = float(sum(np.linalg.norm(np.array(bana[i]) - np.array(bana[i + 1])) * pitch_mm
                      for i in range(len(bana) - 1)))
    return {"bana": bana, "clearance_mm": clr, "langd_mm": langd, "r_eff": r_eff,
            "marginal_mm": clr - min_r_mm}


def korpusuppgifter(korpus=None):
    with open(os.path.join(korpus or KORPUS, "_index.json")) as fh:
        return json.load(fh)["tasks"]


def kor_uppgift(task_id, korpus=None):
    """One corpus task, both routers, on the same distance field and the same endpoints."""
    import f33_1_topologi_v1 as TOP

    k = korpus or KORPUS
    with open(os.path.join(k, task_id, "task.json")) as fh:
        task = json.load(fh)
    npz = np.load(os.path.join(k, task_id, "occupancy_task.npz"))
    occ = npz["occupancy"].astype(bool)
    pitch_mm = float(npz["pitch_mm"])
    min_r_mm = float(task["min_bend_r_floor_mm"])

    fri = ~occ
    edt_mm = distance_transform_edt(fri) * pitch_mm
    giltig = fri & (edt_mm >= min_r_mm + 2.0 * pitch_mm)

    import duct_growth_diff_v1 as DGD
    fall = DGD.synth_case(task_id)
    start = TOP._nearest_valid(TOP._mm_to_idx(fall["path_pts"][0], pitch_mm, occ.shape), giltig)
    mal = TOP._nearest_valid(TOP._mm_to_idx(fall["path_pts"][2], pitch_mm, occ.shape), giltig)

    p_astar, _ = TOP.astar(giltig, start, mal, pitch_mm)
    clr_astar = float(min(edt_mm[p] for p in p_astar)) if p_astar else float("nan")
    langd_astar = (float(sum(np.linalg.norm(np.array(p_astar[i]) - np.array(p_astar[i + 1])) * pitch_mm
                             for i in range(len(p_astar) - 1))) if p_astar else float("nan"))

    flode = rutt_genom_flode(occ, pitch_mm, start, mal, min_r_mm)
    return {"task_id": task_id, "min_r_mm": min_r_mm,
            "astar_loste": bool(p_astar), "flode_loste": bool(flode["bana"]),
            "clr_astar_mm": clr_astar, "clr_flode_mm": flode["clearance_mm"],
            "langd_astar_mm": langd_astar, "langd_flode_mm": flode["langd_mm"],
            "r_eff": flode["r_eff"]}


def las_korpusreferens(sokvag=None):
    with open(os.path.join(sokvag or DATA, "korpus.json")) as fh:
        return json.load(fh)


def sammanfatta(rader):
    """Counts and the clearance comparison, recomputed from per-task rows."""
    lost = sum(1 for r in rader if r.get("len_lap_mm", r.get("langd_flode_mm")) == r.get(
        "len_lap_mm", r.get("langd_flode_mm")))     # a NaN length means no route was found
    battre = 0
    for r in rader:
        a = r.get("clr_astar_mm")
        f = r.get("clr_lap_mm", r.get("clr_flode_mm"))
        if a == a and f == f and f >= a - 1e-9:
            battre += 1
    return {"n": len(rader), "flode_loste": lost, "clearance_minst_lika": battre}


def _selftest():
    ref = las_korpusreferens()
    rader = ref["uppgifter"]
    s = sammanfatta(rader)
    print(f"carried corpus: {s['flode_loste']}/{s['n']} routed by flow, "
          f"{sum(1 for r in rader if r['naive_feasible'])}/{s['n']} by the fixed centreline, "
          f"clearance at least as good on {s['clearance_minst_lika']}/{s['n']}")
    print(f"carried difficulty: Spearman {ref['svarighet']['spearman_reff_vs_clearance']:.4f}, "
          f"hard {ref['svarighet']['mean_reff_hard_tasks']:.5f} against easy "
          f"{ref['svarighet']['mean_reff_easy_tasks']:.5f}")
    h = ref["fabrikshall"]
    print(f"carried hall: K=2 flow {h['K=2']['lap_tot_len_mm']:.0f} mm / "
          f"{h['K=2']['lap_crossings']} crossings against sequential "
          f"{h['K=2']['seq_best_len_mm']:.0f} mm / {h['K=2']['seq_best_crossings']}; "
          f"K=4 flow {h['K=4']['lap_crossings']} crossings against {h['K=4']['seq_best_crossings']}")

    uppg = korpusuppgifter()
    print(f"live on {len(uppg)} available tasks, running the two hardest by carried resistance:")
    ordnade = sorted(rader, key=lambda r: -r["r_eff"])[:2]
    live = []
    for r in ordnade:
        ut = kor_uppgift(r["task_id"])
        live.append(ut)
        print(f"  {ut['task_id']}: flow clearance {ut['clr_flode_mm']:.1f} mm against A* "
              f"{ut['clr_astar_mm']:.1f} mm, R_eff {ut['r_eff']:.5f}")

    fel = []
    if s["flode_loste"] != 11 or len(rader) != 11:
        fel.append("the carried corpus must be 11 tasks routed 11 times")
    if s["clearance_minst_lika"] != 6:
        fel.append(f"clearance at least as good on {s['clearance_minst_lika']}/11, measured 6/11")
    if ref["svarighet"]["spearman_reff_vs_clearance"] > -0.8:
        fel.append("the difficulty correlation is the result; it must be carried as measured")
    for ut in live:
        if not ut["flode_loste"]:
            fel.append(f"{ut['task_id']}: the flow route found no path")
        if not ut["astar_loste"]:
            fel.append(f"{ut['task_id']}: the geodesic baseline found no path")
    if fel:
        raise SystemExit("FAIL: " + "; ".join(fel))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
