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
from scipy.sparse.csgraph import connected_components
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
INFO_SKILDA_KOMPONENTER = -1       # los_potential: inlet and outlet in different components, no solve


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


# Dirichlet physics is distinct from the floating, no-flux routing network above.
def bygg_sdf_dirichlet(sd_mm, pitch_mm, randvarde=0.0, theta_merge=1e-3):
    """Symmetric cut-boundary Poisson operator from the engine SDF (sd<0 in the domain).

    Returns a system dict with A, P, offset, node_map, free, boundary_rhs and diagnostics.
    A is in lattice units; physical source f enters as pitch_mm**2 * P.T @ f. P reconstructs
    the original inside nodes: u = P @ reduced_u + offset. Constant Dirichlet value only.
    The domain must be enclosed by nonnegative samples; no artificial box boundary is invented.

    Boundary energy is (u-g)^2/(2*theta), with theta from linear SDF interpolation. For a tiny
    crossing, merge toward the opposite interior neighbour with u_i-g=theta/(1+theta)*(u_j-g).
    This is a linear reconstruction, not exact Schur elimination. Since P has one positive entry
    per row, P.T A P retains symmetric M-matrix signs. A subthreshold node without a mergeable
    neighbour keeps its eps-floored 1/theta row; a node with no free neighbour at all is refused.
    Accuracy depends on the supplied zero set, and is not upgraded by calling an EDT a SDF.
    Only sign changes between samples are seen: a solid wall thinner than a pitch that leaves both
    neighbouring samples negative is invisible, and a spurious nonnegative sample inside the domain
    becomes a silent interior Dirichlet hole. Neither is detected here.
    """
    from sdf_rand_v1 import sdf_randlankar
    sd = np.asarray(sd_mm, dtype=np.float64)
    h, g = float(pitch_mm), float(randvarde)
    if not np.isfinite(h) or h <= 0 or not np.isfinite(g):
        raise ValueError("positive finite pitch and finite constant boundary value required")
    if not np.isfinite(theta_merge) or not 0 <= theta_merge < 0.5:
        raise ValueError("theta_merge must be in [0, 0.5)")
    directions = np.concatenate([np.eye(sd.ndim, dtype=int), -np.eye(sd.ndim, dtype=int)])
    free, links = sdf_randlankar(sd, directions)
    for axis in range(sd.ndim):
        if np.take(free, [0, sd.shape[axis]-1], axis=axis).any():
            raise ValueError("Dirichlet domain needs a nonnegative SDF halo on every side")
    n, nk, _, _, _, L = bygg_gitterlaplacian(~free, np.ones(sd.shape), h, p_kond=0.0)
    if n == 0:
        raise ValueError("empty Dirichlet domain")
    boundary_nodes = np.concatenate([nk.ravel()[idx] for idx, _, _, _ in links])
    theta = np.concatenate([t for _, t, _, _ in links])
    if np.any(theta <= 0):
        raise ValueError("nonpositive SDF crossing")
    d = np.zeros(n)
    # Floor only at floating-point scale; report its use. Never clamp to the merge threshold.
    theta_safe = np.maximum(theta, np.finfo(float).eps)
    np.add.at(d, boundary_nodes, 1.0/theta_safe)
    A_full = L + sparse.diags(d)
    small = np.zeros(n, bool)
    small[boundary_nodes[theta < theta_merge]] = True
    master = np.arange(n)
    alpha = np.ones(n)
    best = np.ones(n)
    for idx, t, back, back_inside in links:
        nodes = nk.ravel()[idx]
        back_nodes = nk.ravel()[back]
        good = (t < theta_merge) & back_inside
        good &= ~small[np.maximum(back_nodes, 0)]
        good &= t < best[nodes]
        master[nodes[good]] = back_nodes[good]
        alpha[nodes[good]] = t[good]/(1.0+t[good])
        best[nodes[good]] = t[good]
    # A subthreshold node whose interior neighbours are all subthreshold too (the corner of a box whose
    # faces lie on lattice planes, or within the winding query's 1e-4*pitch jitter of them) keeps its
    # own unknown with the eps-floored 1/theta: same signs, solved like theta_merge=0. Only a node with
    # no free lattice neighbour at all is a sliver the grid cannot resolve.
    unmerged = small & (master == np.arange(n))
    if unmerged.any():
        free_deg = np.asarray((L != 0).sum(axis=1)).ravel() - 1
        if np.any(unmerged & (free_deg <= 0)):
            raise ValueError("subthreshold SDF sliver has no interior merge neighbour; refine geometry/grid")
        small &= ~unmerged
    keep = ~small
    ids = -np.ones(n, dtype=int)
    ids[keep] = np.arange(keep.sum())
    P = sparse.coo_matrix((alpha, (np.arange(n), ids[master])), shape=(n, int(keep.sum()))).tocsr()
    offset = (1.0-alpha)*g
    A = (P.T @ A_full @ P).tocsr()
    boundary_rhs = np.asarray(P.T @ (d*alpha*g - L @ offset)).ravel()
    return dict(A=A, P=P, offset=offset, free=free, node_map=nk,
                boundary_rhs=boundary_rhs, theta=theta, boundary_nodes=boundary_nodes,
                pitch_mm=h, stats=dict(nodes=n, reduced_nodes=int(keep.sum()),
                merged_nodes=int(small.sum()), unmerged_small_nodes=int(unmerged.sum()),
                theta_min=float(theta.min()),
                floored_links=int(np.sum(theta < np.finfo(float).eps)), nnz=int(A.nnz)))


def los_sdf_poisson(sd_mm, pitch_mm, rhs=1.0, randvarde=0.0, theta_merge=1e-3,
                    tol=1e-11, maxiter=20000):
    """Solve -lap u=rhs with constant Dirichlet data, returning (dense field, system, CG info).

    A failed CG solve produces NaNs inside, never an unconverged physics answer; one correction solve
    on the explicitly recomputed residual precedes that verdict. rhs is a scalar or an array of the
    SDF shape. Empty/out-of-halo domains and isolated slivers raise ValueError.
    """
    system = bygg_sdf_dirichlet(sd_mm, pitch_mm, randvarde, theta_merge)
    source = np.broadcast_to(np.asarray(rhs, dtype=float), system["free"].shape)[system["free"]]
    if not np.all(np.isfinite(source)):
        raise ValueError("source must be finite in the domain")
    b = pitch_mm**2 * np.asarray(system["P"].T @ source).ravel() + system["boundary_rhs"]
    A = system["A"]
    M = sparse.diags(1.0/A.diagonal())
    x, info = cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, M=M)
    residual = float(np.linalg.norm(A @ x - b)/max(np.linalg.norm(b), 1e-300))
    if info == 0 and residual > max(10*tol, 1e-14):
        # CG's recursive residual drifts from the true one on fine grids (2D N=768: 1.6e-10 after a
        # reported convergence at 1e-11). One correction solve on the explicit residual, then re-gate.
        dx, info = cg(A, b - A @ x, rtol=0.01, atol=0.0, maxiter=maxiter, M=M)
        x = x + dx
        residual = float(np.linalg.norm(A @ x - b)/max(np.linalg.norm(b), 1e-300))
        system["stats"]["residual_correction"] = True
    system["stats"]["relative_residual"] = residual
    field = np.full(system["free"].shape, np.nan)
    if info == 0 and np.all(np.isfinite(x)) and residual <= max(10*tol, 1e-14):
        field[system["free"]] = system["P"] @ x + system["offset"]
    else:
        info = int(info) if info else -2
    return field, system, int(info)


def nollrum_komponenter(L):
    """The null space of the network Laplacian, as component labels.

    L's null space is spanned by one indicator vector per connected component of the free space, not
    only by the global constant. Returns (number of components = null-space dimension, label per
    node); the basis vector of component k is (label == k). A stored entry of value zero is not a
    conductance, so explicit zeros are dropped before labelling (csgraph counts them as edges)."""
    G = sparse.csr_matrix(L, copy=True)
    G.eliminate_zeros()
    return connected_components(G, directed=False)


def los_potential(L, n, s_nod, t_nod, tol=1e-10, maxiter=2000):
    """One Laplace solve: unit current in at the inlet, out at the outlet.

    The Laplacian is singular by one constant per connected component (a floating network has no
    absolute potential), so the solution is only defined up to an additive constant per component;
    the effective resistance, which is a DIFFERENCE of potentials, is not. That is the quantity read
    off here.

    The system is consistent only when N^T b = 0 for the null-space basis N, i.e. when inlet and
    outlet lie in the same component. When they do not, no current can flow and the effective
    resistance is +inf exactly; CG is then not run, since on that inconsistent system it returns a
    finite potential difference set by its iteration count (a wall-split 24 x 24 room read
    R_eff = -5.09e14 at maxiter 2000). The return is (nan potential, inf, INFO_SKILDA_KOMPONENTER).
    A CG run that does not converge (info != 0) returns R_eff = nan, uncertain, never the
    unconverged difference."""
    if s_nod == t_nod:
        return np.zeros(n), 0.0, 0         # no separation, R_eff = 0 exactly (b would be -e_s, inconsistent)
    _nk, etikett = nollrum_komponenter(L)
    if etikett[s_nod] != etikett[t_nod]:
        return np.full(n, np.nan), float("inf"), INFO_SKILDA_KOMPONENTER
    b = np.zeros(n)
    b[s_nod], b[t_nod] = 1.0, -1.0
    M = sparse.diags(1.0 / np.maximum(L.diagonal(), 1e-12))
    x, info = cg(L, b, rtol=tol, maxiter=maxiter, M=M)
    x = x - x.mean()
    r_eff = float(x[s_nod] - x[t_nod])
    if info != 0 or not np.isfinite(r_eff):
        r_eff = float("nan")
    return x, r_eff, int(info)


def r_eff_status(r_eff, info):
    """INFEASIBEL: inlet and outlet in different components, R_eff = inf is exact. OSÄKER: the solve
    did not converge or gave a non-finite difference. OK otherwise."""
    if info == INFO_SKILDA_KOMPONENTER and r_eff == float("inf"):
        return "INFEASIBEL"
    if info != 0 or not np.isfinite(r_eff):
        return "OSÄKER"
    return "OK"


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
    pot, r_eff, info = los_potential(L, n, s, t)
    status = r_eff_status(r_eff, info)
    if status == "INFEASIBEL" or not np.all(np.isfinite(pot)):
        # No current path (or no usable potential): no route. An OSÄKER solve still has a finite
        # potential, and the route extracted from it is a real path whose clearance and length are
        # measured on the path itself, so it is kept; only R_eff is withheld (nan, OSÄKER).
        return {"bana": [], "clearance_mm": float("nan"), "langd_mm": float("nan"),
                "r_eff": r_eff, "marginal_mm": float("nan"), "r_eff_status": status}
    koordinater = np.argwhere(fri)
    bana = extrahera_strombana(_grannlista(n, u, v), koordinater, s, t, kond * (pot[u] - pot[v]))
    if not bana:
        return {"bana": [], "clearance_mm": float("nan"), "langd_mm": float("nan"),
                "r_eff": r_eff, "marginal_mm": float("nan"), "r_eff_status": status}
    clr = float(min(edt_mm[p] for p in bana))
    langd = float(sum(np.linalg.norm(np.array(bana[i]) - np.array(bana[i + 1])) * pitch_mm
                      for i in range(len(bana) - 1)))
    return {"bana": bana, "clearance_mm": clr, "langd_mm": langd, "r_eff": r_eff,
            "marginal_mm": clr - min_r_mm, "r_eff_status": status}


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
            "r_eff": flode["r_eff"], "r_eff_status": flode["r_eff_status"]}


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
