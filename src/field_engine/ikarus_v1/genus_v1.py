#!/usr/bin/env python3
"""genus_v1 -- the sanctioned genus / hole primitive: one source, with a proved reference answer.

TWO INSTRUMENTS, ONE IDENTITY. Hand-rolled Euler counting per boundary component and
manifold3d.Manifold.genus() do NOT measure the same quantity. Both are correct; they answer
different questions.

  handles_summa  = SUM over each connected BOUNDARY COMPONENT of g_i = (2 - chi_i)/2
                   "how many HANDLES (through tunnels) do the boundary surfaces have together"
  m3d_genus      = manifold3d.Manifold.genus() = 1 - chi_total/2, where chi_total is counted over
                   the WHOLE mesh at once (all components). manifold3d's own docstring says so:
                   "It is only meaningful for a single mesh, so it is best to call Decompose()
                   first."

EXACT IDENTITY (derived, and machine-verified in facit_selftest for every body):
    chi_total = SUM_i (2 - 2 g_i) = 2 n_comp - 2 SUM_i g_i
    m3d_genus = 1 - chi_total/2 = 1 - n_comp + SUM_i g_i
  =>  m3d_genus  ==  handles_summa - (n_comp - 1)
  =>  they coincide EXACTLY when n_comp == 1 and differ by the number of ENCLOSED VOIDS otherwise.

THE FAIL-OPEN THIS REVEALS (the important result here): m3d_genus is a DIFFERENCE. A body with 1
unintended tunnel AND 1 enclosed void gives m3d_genus == 0 -- exactly the same number as a flawless
solid block. The tunnel cancels against the void and DISAPPEARS from the verdict. m3d_genus must
therefore NEVER be the only gate quantity; a gate must read handles_summa (tunnels) and n_hallrum
(enclosed voids) SEPARATELY. The facit body "tunnel1_hallrum1" is that counterexample and it lives
in this file so that a future simplification falls on it.

ACCOUNTING RULE: for a duct with n_designportar mouths (and n_bulthal_design intentional through
bolt holes):
    expected handles_summa = (n_designportar - 1) + n_bulthal_design
    n_oavsiktliga_hal      = handles_summa - (n_designportar - 1) - n_bulthal_design
A straight pipe with 2 mouths has handles_summa == 1. handles_summa == 0 means there is NO through
lumen at all (the solid is a BLOCK).

THE MEASUREMENT PATH: a B-rep Euler count on the shell is NOT the boundary surface's Euler
characteristic. OCC's periodic surfaces (cylinder/cone/torus) carry SEAM edges and degenerate edges
that are counted once but are topologically two or zero, which yields BROKEN (half-integer) genus.
The measurement is therefore always taken on the TRIANGULATION, after welding the tessellation's
duplicate vertices. A triangulation has no seam edges.
CONVERGENCE REQUIREMENT: a genus that changes with the tessellation deflection is a MEASUREMENT
ERROR, not a hole.

I/O: a build123d shape or a STEP file in; a per-solid topology dict out. --selftest builds the
synthetic reference bodies and writes artifacts/genus_v1_facit.json next to this module.

Run:  python genus_v1.py --selftest          (synthetic reference, N=0/1/2/3 tunnels + voids)
      python genus_v1.py --fil <step> [--tol 0.05]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TOL_MM = 0.05          # tessellation deflection
SVETS_DEC = 4          # weld vertices at 1e-4 mm (0.1 um) -- far below the smallest CAD feature
TOL_KONVERGENS = (0.05, 0.02)   # two deflections; different answers => MEASUREMENT ERROR, not a hole


# ------------------------------------------------------------------ boundary topology (reference)
def mesh_topologi(P, T):
    """(verts Nx3, tris Mx3) -> topology dict. A PURE function: no OCC, testable in isolation.

    Returns V/E/F/chi/genus per connected boundary component, plus the two quantities a gate should
    judge on: handles_summa (tunnels) and n_hallrum (enclosed voids)."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg

    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=np.int64)

    # --- weld duplicate vertices (the tessellation duplicates them along every B-rep edge)
    nyckel = np.round(P, SVETS_DEC)
    uniq, inv = np.unique(nyckel, axis=0, return_inverse=True)
    T = inv.reshape(-1)[T]
    ok = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    n_degen = int((~ok).sum())
    T = T[ok]
    anv = np.unique(T)
    omap = np.full(uniq.shape[0], -1, dtype=np.int64)
    omap[anv] = np.arange(anv.size)
    Pw = uniq[anv]
    T = omap[T]

    nV, nF = int(anv.size), int(T.shape[0])
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=0), axis=1)
    uE, cnt = np.unique(E, axis=0, return_counts=True)
    nE = int(uE.shape[0])
    n_kant_ej2 = int((cnt != 2).sum())

    # --- connected boundary components over triangle adjacency
    tri_id = np.tile(np.arange(nF), 3)
    _, ei = np.unique(E, axis=0, return_inverse=True)
    M = sp.coo_matrix((np.ones(ei.size), (ei.reshape(-1), tri_id)), shape=(nE, nF)).tocsr()
    ncomp, lab = csg.connected_components((M.T @ M) > 0, directed=False)

    komp = []
    for c in range(ncomp):
        Tc = T[lab == c]
        Vc = int(np.unique(Tc).size)
        Ec = int(np.unique(np.sort(np.concatenate(
            [Tc[:, [0, 1]], Tc[:, [1, 2]], Tc[:, [2, 0]]]), axis=1), axis=0).shape[0])
        Fc = int(Tc.shape[0])
        chi = Vc - Ec + Fc
        komp.append({"V": Vc, "E": Ec, "F": Fc, "chi": chi, "genus": (2 - chi) / 2.0})

    a, b, c3 = Pw[T[:, 0]], Pw[T[:, 1]], Pw[T[:, 2]]
    mvol = float(abs(np.einsum("ij,ij->i", a, np.cross(b, c3)).sum()) / 6.0)

    handles = float(sum(k["genus"] for k in komp))
    chi_tot = int(sum(k["chi"] for k in komp))
    return {
        "nV": nV, "nE": nE, "nF": nF, "n_degenererade_trianglar": n_degen,
        "n_kanter_ej_delade_av_2": n_kant_ej2,
        "vattentat_mesh": bool(n_kant_ej2 == 0),
        "n_randkomponenter": int(ncomp), "komponenter": komp,
        "handles_summa": handles,          # <-- TUNNELS. The gate's number.
        "n_hallrum": int(ncomp - 1),       # <-- ENCLOSED VOIDS. The gate's second number.
        "chi_totalt": chi_tot,
        # backward-compatible aliases, so existing consumers keep their report key names
        "genus_summa": handles, "chi_summa": chi_tot,
        # the same formula manifold3d.genus() uses, computed from OUR mesh -- this makes the
        # identity checkable without manifold3d even being installed
        "m3d_genus_formel": float(1 - chi_tot / 2.0),
        "meshvolym_mm3": round(mvol, 6),
        "heltaligt": bool(all(float(k["genus"]).is_integer() for k in komp)),
        # the WELDED mesh out -- manifold3d MUST get exactly the same mesh, otherwise we are
        # comparing two geometries again instead of two instruments
        "_svetsad": (Pw, T),
    }


# ------------------------------------------------------------------ the OCC side
def _solid_mesh(sol, tol):
    import numpy as np
    verts, tris = sol.tessellate(tol)
    return (np.array([[v.X, v.Y, v.Z] for v in verts], dtype=float),
            np.array(tris, dtype=np.int64))


def _m3d_genus(P, T):
    """manifold3d's OWN number on exactly the same mesh (independent library, independent path)."""
    try:
        import numpy as np
        import manifold3d as m3d
        man = m3d.Manifold(m3d.Mesh(
            vert_properties=np.ascontiguousarray(P, dtype=np.float32),
            tri_verts=np.ascontiguousarray(T, dtype=np.uint32)))
        st = str(man.status())
        if man.is_empty():
            return {"tillganglig": True, "konstruerad": False, "status": st, "genus": None}
        return {"tillganglig": True, "konstruerad": True, "status": st,
                "genus": int(man.genus()), "volym_mm3": float(man.volume())}
    except ImportError:
        return {"tillganglig": False, "genus": None}
    except Exception as e:  # noqa: BLE001 -- escalate, never a silent drop
        return {"tillganglig": True, "konstruerad": False,
                "fel": f"{type(e).__name__}: {e}"[:200], "genus": None}


def mat_shape(shape, tol=TOL_MM, med_m3d=True):
    """build123d shape -> per-solid topology plus sums. Both instruments on the SAME mesh."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    solids = list(shape.solids())
    ut = {"n_solids": len(solids), "tol_mm": tol, "solids": []}
    for si, sol in enumerate(solids):
        P, T = _solid_mesh(sol, tol)
        topo = mesh_topologi(P, T)
        Pw, Tw = topo.pop("_svetsad")
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(sol.wrapped, g)
        bvol = float(g.Mass())
        avvik = abs(topo["meshvolym_mm3"] - bvol) / max(abs(bvol), 1e-9)
        topo.update({"solid_index": si, "brepvolym_mm3": round(bvol, 6),
                     "volymavvikelse_rel": round(avvik, 6),
                     "tessellation_giltig": bool(avvik < 0.01)})
        if med_m3d:
            topo["manifold3d"] = _m3d_genus(Pw, Tw)
        ut["solids"].append(topo)

    ut["handles_summa"] = float(sum(s["handles_summa"] for s in ut["solids"]))
    ut["n_randkomponenter_totalt"] = int(sum(s["n_randkomponenter"] for s in ut["solids"]))
    ut["n_hallrum_totalt"] = int(sum(s["n_hallrum"] for s in ut["solids"]))
    ut["vattentat_mesh"] = all(s["vattentat_mesh"] for s in ut["solids"])
    ut["tessellation_giltig"] = all(s["tessellation_giltig"] for s in ut["solids"])
    ut["heltaligt"] = all(s["heltaligt"] for s in ut["solids"])
    m3ds = [s.get("manifold3d", {}).get("genus") for s in ut["solids"]] if med_m3d else []
    ut["m3d_genus_per_solid"] = m3ds
    # backward-compatible alias: earlier consumers called handles_summa "genus_totalt"
    ut["genus_totalt"] = ut["handles_summa"]
    return ut


def mat_step(path, tol=TOL_MM, med_m3d=True):
    import build123d as bd
    return mat_shape(bd.import_step(path), tol=tol, med_m3d=med_m3d)


def mat_step_konvergerad(path, toler=TOL_KONVERGENS, med_m3d=True):
    """CONVERGENCE REQUIREMENT: a genus that changes with the tessellation is a MEASUREMENT ERROR."""
    per = {}
    for t in toler:
        try:
            per[str(t)] = mat_step(path, tol=t, med_m3d=med_m3d)
        except Exception as e:  # noqa: BLE001
            per[str(t)] = {"fel": f"{type(e).__name__}: {e}"[:300]}
    gs = [v.get("handles_summa") for v in per.values()]
    hs = [v.get("n_hallrum_totalt") for v in per.values()]
    konv = bool(len(set(gs)) == 1 and gs[0] is not None and len(set(hs)) == 1)
    return {"step": path, "toleranser_mm": list(toler), "per_tol": per,
            "konvergerat": konv,
            "handles_summa": gs[0] if konv else None,
            "n_hallrum_totalt": hs[0] if konv else None,
            "m3d_genus_per_solid": (list(per.values())[0].get("m3d_genus_per_solid")
                                    if konv else None)}


def n_oavsiktliga_hal(handles_summa, n_designportar, n_bulthal_design=0):
    """THE ACCOUNTING RULE. n_designportar mouths + n_bulthal_design intentional bolt holes."""
    if handles_summa is None:
        return None
    return float(handles_summa) - (int(n_designportar) - 1) - int(n_bulthal_design)


# ------------------------------------------------------------------ SYNTHETIC REFERENCE
def _facit_kroppar():
    """Bodies with a KNOWN number of through holes and known voids. The reference answer is
    CONSTRUCTED (the topology follows from how the body is built), the measurement is MEASURED --
    that is the whole point of the test."""
    import build123d as bd

    def block(w=40, d=40, h=12):
        return bd.Solid.make_box(w, d, h)

    def cyl(r, h, x, y, z=-1.0):
        return bd.Solid.make_cylinder(r, h).locate(bd.Location((x, y, z)))

    kroppar = []
    # N through holes in a block: ONE boundary component, genus == N
    for n in (0, 1, 2, 3):
        b = block()
        for i in range(n):
            b = b.cut(cyl(3.0, 14.0, 8.0 + 10.0 * i, 20.0))
        kroppar.append({"namn": f"tunnel{n}", "shape": b,
                        "facit_handles": float(n), "facit_hallrum": 0,
                        "facit_m3d": float(n),   # n_komp == 1 => instrumenten MASTE sammanfalla
                        "hur": f"block 40x40x12 with {n} through cylindrical holes of r=3"})
    # an enclosed void: TWO boundary components, both spherical => handles 0, m3d = -1
    b = block().cut(bd.Solid.make_sphere(6.0).locate(bd.Location((20, 20, 6))))
    kroppar.append({"namn": "hallrum1", "shape": b,
                    "facit_handles": 0.0, "facit_hallrum": 1, "facit_m3d": -1.0,
                    "hur": "block with ONE enclosed spherical cavity of r=6 (no tunnel)"})
    # THE COUNTEREXAMPLE: 1 tunnel + 1 void => handles 1, voids 1, but m3d_genus == 0.
    # A flawless block ALSO gives m3d_genus == 0. The tunnel DISAPPEARS from the m3d number.
    b = block().cut(cyl(3.0, 14.0, 8.0, 20.0)).cut(
        bd.Solid.make_sphere(5.0).locate(bd.Location((28, 20, 6))))
    kroppar.append({"namn": "tunnel1_hallrum1", "shape": b,
                    "facit_handles": 1.0, "facit_hallrum": 1, "facit_m3d": 0.0,
                    "hur": "block with 1 through hole AND 1 enclosed cavity -- m3d_genus == 0 "
                           "cancels the tunnel away (the fail-open counterexample)"})
    # torus: classic genus 1, ONE component (checks curved periodic surfaces and seam edges)
    kroppar.append({"namn": "torus", "shape": bd.Solid.make_torus(20.0, 5.0),
                    "facit_handles": 1.0, "facit_hallrum": 0, "facit_m3d": 1.0,
                    "hur": "torus R=20 r=5 -- periodic surfaces, tests the seam-edge trap"})
    return kroppar


def facit_selftest():
    """Falsification: both instruments against a KNOWN reference plus the identity
    m3d == handles - (n_comp-1)."""
    rader = []
    for k in _facit_kroppar():
        m = mat_shape(k["shape"], tol=TOL_MM, med_m3d=True)
        m3d_uppmatt = (m["m3d_genus_per_solid"] or [None])[0]
        r = {
            "kropp": k["namn"], "hur": k["hur"],
            "facit_handles": k["facit_handles"], "uppmatt_handles": m["handles_summa"],
            "facit_hallrum": k["facit_hallrum"], "uppmatt_hallrum": m["n_hallrum_totalt"],
            "facit_m3d_genus": k["facit_m3d"], "uppmatt_m3d_genus": m3d_uppmatt,
            "n_randkomponenter": m["n_randkomponenter_totalt"],
            "m3d_genus_ur_var_egen_formel": (m["solids"][0]["m3d_genus_formel"]
                                             if m["solids"] else None),
            "vattentat_mesh": m["vattentat_mesh"],
            "tessellation_giltig": m["tessellation_giltig"],
            "heltaligt": m["heltaligt"],
        }
        r["handles_RATT"] = bool(r["uppmatt_handles"] == k["facit_handles"])
        r["hallrum_RATT"] = bool(r["uppmatt_hallrum"] == k["facit_hallrum"])
        r["m3d_RATT"] = bool(m3d_uppmatt is not None and float(m3d_uppmatt) == k["facit_m3d"])
        # THE DERIVED IDENTITY, machine-verified per body
        r["identitet_m3d_eq_handles_minus_hallrum"] = bool(
            m3d_uppmatt is not None
            and float(m3d_uppmatt) == r["uppmatt_handles"] - (r["n_randkomponenter"] - 1))
        r["PASS"] = bool(r["handles_RATT"] and r["hallrum_RATT"] and r["m3d_RATT"]
                         and r["identitet_m3d_eq_handles_minus_hallrum"]
                         and r["vattentat_mesh"] and r["tessellation_giltig"])
        rader.append(r)
        print(json.dumps({q: r[q] for q in ("kropp", "facit_handles", "uppmatt_handles",
                                            "facit_m3d_genus", "uppmatt_m3d_genus",
                                            "n_randkomponenter", "PASS")},
                         ensure_ascii=False), flush=True)

    # THE DISCRIMINATING ROW: is there a pair of bodies where m3d_genus is EQUAL but
    # handles/voids DIFFER? Then m3d_genus is PROVED insufficient as the only gate.
    par = []
    for i in range(len(rader)):
        for j in range(i + 1, len(rader)):
            a, b = rader[i], rader[j]
            if (a["uppmatt_m3d_genus"] == b["uppmatt_m3d_genus"]
                    and (a["uppmatt_handles"], a["uppmatt_hallrum"])
                    != (b["uppmatt_handles"], b["uppmatt_hallrum"])):
                par.append({"a": a["kropp"], "b": b["kropp"],
                            "samma_m3d_genus": a["uppmatt_m3d_genus"],
                            "a_handles_hallrum": [a["uppmatt_handles"], a["uppmatt_hallrum"]],
                            "b_handles_hallrum": [b["uppmatt_handles"], b["uppmatt_hallrum"]]})

    res = {
        "cell": "genus_v1_facit_selftest",
        "metod": "SYNTHETIC REFERENCE (the topology follows from how the body is built), measured "
                 "with BOTH instruments on the SAME triangulation; the identity "
                 "m3d == handles - (n_comp-1) is verified per body.",
        "rader": rader,
        "n_kroppar": len(rader),
        "n_pass": sum(1 for r in rader if r["PASS"]),
        "n_identitet_haller": sum(1 for r in rader
                                  if r["identitet_m3d_eq_handles_minus_hallrum"]),
        "kollisionspar_m3d_lika_men_topologi_olik": par,
        "n_kollisionspar": len(par),
        "SLUTSATS": None,
    }
    res["ALLA_PASS"] = bool(res["n_pass"] == res["n_kroppar"])
    res["IDENTITETEN_HALLER_OVERALLT"] = bool(res["n_identitet_haller"] == res["n_kroppar"])
    res["M3D_ENSAM_AR_OTILLRACKLIG"] = bool(res["n_kollisionspar"] > 0)
    res["SLUTSATS"] = (
        "The instruments measure different quantities and both are correct: "
        "m3d_genus == handles_summa - n_hallrum. They coincide exactly when n_randkomponenter==1 "
        "(every pure-tunnel body). A gate must judge handles_summa AND n_hallrum separately; "
        "m3d_genus alone cancels a tunnel against a void."
        if res["ALLA_PASS"] and res["IDENTITETEN_HALLER_OVERALLT"]
        else "THE REFERENCE FAILED -- see rader[].")
    res["VERDICT"] = "PASS" if (res["ALLA_PASS"] and res["IDENTITETEN_HALLER_OVERALLT"]) else "FAIL"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="synthetic reference bodies")
    ap.add_argument("--fil", help="STEP file to measure")
    ap.add_argument("--tol", type=float, default=TOL_MM)
    ap.add_argument("--konvergens", action="store_true", help="run both deflections")
    ap.add_argument("--ut", help="write the JSON here")
    a = ap.parse_args()

    if a.selftest:
        r = facit_selftest()
        p = a.ut or os.path.join(HERE, "artifacts", "genus_v1_facit.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(r, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(json.dumps({k: v for k, v in r.items() if k != "rader"},
                         indent=1, ensure_ascii=False))
        print("WROTE " + p)
        return 0 if r["VERDICT"] == "PASS" else 1

    if a.fil:
        r = (mat_step_konvergerad(a.fil) if a.konvergens else mat_step(a.fil, tol=a.tol))
        print("@@JSON@@" + json.dumps(r, ensure_ascii=False))
        if a.ut:
            json.dump(r, open(a.ut, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
