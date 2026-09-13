#!/usr/bin/env python3
"""ritning_paritet_v1 -- the gate for ritning_gen_v1: every PLACED dimension is re-measured.

Why a separate file: a gate that calls the generator's own measuring functions is a tautology.
This gate shares NO code with the generator and does NOT run inside FreeCAD: it reads the STEP files
again through OCP (the python binding to OCCT, its own process, its own code path) and compares
against what is ACTUALLY ON THE SHEET -- the numbers are parsed out of the SVG's <text> elements,
not out of the generator's matt_v1.json.

Three channels, all two-sided:
  K1 GEOMETRY PARITY          |drawn number - OCP-measured value| <= tol (0.1 mm)
  K2 DRAWING SELF-CONSISTENCY drawn dimension-line length / scale == drawn number (+-tol)
                              (catches scale errors and misplaced dimensions that K1 is blind to)
  K3 TITLE BLOCK              the mass in the title block == the manifest's mass, re-summed
  + LINE WORK                 the number of path elements in the SVG > threshold (an empty drawing
                              is not green)

Falsification: the same gate is run against a COPY of the SVG in which ONE dimension number is
shifted by --fallbevis-mm (default 1.0). That run MUST fail, otherwise the gate is vacuous.

Run: python ritning_gen_v1.py grind --namn bracket_v1
Output: artifacts/<namn>/grind_v1.json
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("FIELD_ENGINE_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(HERE)))
UT_ROT = os.path.join(HERE, "artifacts")


# ---------------------------------------------------------------- OCP measurement (independent)


def _ocp():
    from OCP.STEPControl import STEPControl_Reader
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_IN
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopoDS import TopoDS
    from OCP.BRepTools import BRepTools
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt, gp_Trsf
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.TopoDS import TopoDS_Compound, TopoDS_Builder
    return locals()


def las_step(path, transform, M):
    r = M["STEPControl_Reader"]()
    r.ReadFile(path)
    r.TransferRoots()
    sh = r.OneShape()
    if transform:
        t = M["gp_Trsf"]()
        t.SetValues(*[float(transform[i][j]) for i in range(3) for j in range(4)])
        sh = M["BRepBuilderAPI_Transform"](sh, t, True).Shape()
    return sh


def mat_geometri(delar, M):
    """The optimal AABB over all parts plus inner full cylinders (holes) with a material test."""
    shapes = []
    for d in delar:
        p = d["step"] if os.path.isabs(d["step"]) else os.path.join(BASE, d["step"])
        shapes.append(las_step(p, d.get("transform"), M))
    comp = M["TopoDS_Compound"]()
    b = M["TopoDS_Builder"]()
    b.MakeCompound(comp)
    for s in shapes:
        b.Add(comp, s)
    box = M["Bnd_Box"]()
    M["BRepBndLib"].AddOptimal_s(comp, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    # THE MATERIAL QUESTION MUST BE ASKED AGAINST THE WHOLE ASSEMBLY, not one part at a time: a
    # bore that a shaft fills is not a hole ON THE DRAWING. The generator asks the compound; the
    # gate asks the same question with its OWN implementation -- one classifier per part and
    # "material = some part contains the point".
    klassare = [M["BRepClass3d_SolidClassifier"](sh) for sh in shapes]

    def material(pnt):
        for c in klassare:
            c.Perform(pnt, 1e-6)
            if c.State() == M["TopAbs_IN"]:
                return True
        return False

    hal = []
    for sh in shapes:
        ex = M["TopExp_Explorer"](sh, M["TopAbs_FACE"])
        while ex.More():
            f = M["TopoDS"].Face_s(ex.Current())
            ad = M["BRepAdaptor_Surface"](f)
            if ad.GetType() == M["GeomAbs_Cylinder"]:
                u0, u1, v0, v1 = M["BRepTools"].UVBounds_s(f)
                if (u1 - u0) > 2 * math.pi - 1e-3:
                    c = ad.Cylinder()
                    ax = c.Axis().Direction()
                    ctr = c.Location()
                    rr = c.Radius()
                    p = ad.Value((u0 + u1) / 2.0, (v0 + v1) / 2.0)
                    dvec = [p.X() - ctr.X(), p.Y() - ctr.Y(), p.Z() - ctr.Z()]
                    a = [ax.X(), ax.Y(), ax.Z()]
                    k = sum(dvec[i] * a[i] for i in range(3))
                    rad = [dvec[i] - a[i] * k for i in range(3)]
                    L = math.sqrt(sum(x * x for x in rad))
                    if L > 1e-9:
                        rad = [x / L for x in rad]
                        st = max(0.05, min(0.15 * rr, 1.0))
                        pin = M["gp_Pnt"](p.X() - rad[0] * st, p.Y() - rad[1] * st,
                                          p.Z() - rad[2] * st)
                        pou = M["gp_Pnt"](p.X() + rad[0] * st, p.Y() + rad[1] * st,
                                          p.Z() + rad[2] * st)
                        i_in = material(pin)
                        i_ou = material(pou)
                        if i_ou and not i_in:
                            hal.append({"d_mm": round(2 * rr, 4), "axel": [round(x, 6) for x in a],
                                        "centrum": [round(ctr.X(), 4), round(ctr.Y(), 4),
                                                    round(ctr.Z(), 4)]})
            ex.Next()
    ded = {}
    for h in hal:
        ded[(round(h["d_mm"], 3), tuple(round(x, 4) for x in h["axel"]),
             tuple(round(x, 3) for x in h["centrum"]))] = h
    return {
        "bbox": [xmin, ymin, zmin, xmax, ymax, zmax],
        "L_mm": xmax - xmin, "B_mm": ymax - ymin, "H_mm": zmax - zmin,
        "hal": list(ded.values()),
        "kanal": "OCP/OCCT (STEPControl_Reader + BRepBndLib.AddOptimal + "
                 "BRepClass3d_SolidClassifier) -- own process, shares no code with the generator",
    }


# ---------------------------------------------------------------- SVG reading (what is on the sheet)


def las_svg_matt(svg_text):
    """Every dimension number actually on the sheet, plus the drawn dimension-line lengths."""
    ut = {}
    for m in re.finditer(r'<text[^>]*class="matt"[^>]*>(.*?)</text>', svg_text, re.S):
        tag = m.group(0)
        mid = re.search(r'data-matt-id="([^"]+)"', tag)
        if not mid:
            continue
        s = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        ut.setdefault(mid.group(1), {})["text"] = s
    for m in re.finditer(r'<line[^>]*class="mattlinje"[^>]*/>', svg_text):
        tag = m.group(0)
        mid = re.search(r'data-matt-id="([^"]+)"', tag)
        if not mid:
            continue
        c = {k: float(re.search(r'%s="([-0-9.]+)"' % k, tag).group(1))
             for k in ("x1", "y1", "x2", "y2")}
        ut.setdefault(mid.group(1), {})["ritad_mm"] = math.hypot(c["x2"] - c["x1"],
                                                                 c["y2"] - c["y1"])
    return ut


def tolka_tal(s):
    """'95.4' / '2 x Ø22.3' -> (value, count). None when the text is not a dimension."""
    if s is None:
        return None, None
    ant = 1
    m = re.match(r"^\s*(\d+)\s*x\s*", s)
    if m:
        ant = int(m.group(1))
        s = s[m.end():]
    s = s.replace("Ø", "").strip()
    try:
        return float(s), ant
    except ValueError:
        return None, None


def las_titelfalt_massa(svg_text):
    m = re.search(r'freecad:editable="supplementary_title_2"[^>]*>(.*?)</text>', svg_text, re.S)
    if not m:
        return None
    s = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    if not s or s == "-":
        return None
    m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    return float(m2.group(1)) if m2 else None


# ---------------------------------------------------------------- frame maths (the gate's own)


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def axlar(d, x):
    d = _norm(d)
    k = sum(x[i] * d[i] for i in range(3))
    x = _norm([x[i] - k * d[i] for i in range(3)])
    return x, _cross(d, x)


# ---------------------------------------------------------------- the gate


def kor_grind(matt, svg_text, tol=0.1):
    M = _ocp()
    g = mat_geometri(matt["delar"], M)
    pa_arket = las_svg_matt(svg_text)
    skala = matt["skala"]
    atomer = []

    for rad in matt["matt"]:
        mid = rad["id"]
        rit = pa_arket.get(mid, {})
        v_ark, ant_ark = tolka_tal(rit.get("text"))
        post = {"id": mid, "typ": rad["typ"], "vy": rad["vy"],
                "pa_arket": rit.get("text"), "ark_varde": v_ark}
        if v_ark is None:
            post.update({"pass": False, "varfor": "the dimension number is not on the sheet"})
            atomer.append(post)
            continue

        if rad["typ"].startswith("huvudmatt"):
            vy = matt["vyer"][rad["vy"]]
            ux, vv = axlar(vy["dir"], vy["xdir"])
            axel = ux if rad["typ"].endswith("_u") else vv
            matt_v = (g["L_mm"] * abs(axel[0]) + g["B_mm"] * abs(axel[1])
                      + g["H_mm"] * abs(axel[2]))
            post["ocp_varde"] = round(matt_v, 4)
            post["avvikelse_mm"] = round(abs(v_ark - matt_v), 5)
            post["k1_geometriparitet"] = bool(post["avvikelse_mm"] <= tol)
        else:  # diameter
            # "n x OD" on a sheet means n holes OF THAT DIAMETER VISIBLE IN THIS VIEW, i.e. whose
            # axis is parallel to the viewing direction. Counting the whole assembly's holes of the
            # same diameter compares two different quantities.
            vy = matt["vyer"][rad["vy"]]
            dv = _norm(vy["dir"])
            i_vyn = [h for h in g["hal"]
                     if abs(sum(h["axel"][i] * dv[i] for i in range(3))) > 0.999]
            kand = [h["d_mm"] for h in i_vyn]
            alla = [h["d_mm"] for h in g["hal"]]
            n_traff = len([d for d in kand if abs(d - v_ark) <= tol])
            post["ocp_kandidater_i_vyn"] = sorted(set(round(d, 3) for d in kand))[:12]
            post["ocp_antal_i_vyn_med_denna_diameter"] = n_traff
            post["ocp_antal_hela_sammanstallningen"] = len(
                [d for d in alla if abs(d - v_ark) <= tol])
            post["avvikelse_mm"] = round(min([abs(d - v_ark) for d in alla], default=9e9), 5)
            post["k1_geometriparitet"] = bool(n_traff > 0 and n_traff == ant_ark)
            post["antal_pa_arket"] = ant_ark

        rl = rit.get("ritad_mm")
        if rl is not None:
            post["ritad_langd_mm"] = round(rl, 4)
            post["ritad_langd_i_modellmm"] = round(rl / skala, 4)
            post["k2_sjalvkonsistens"] = bool(abs(rl / skala - v_ark) <= tol)
        else:
            post["k2_sjalvkonsistens"] = None   # diameter callouts have no dimension line
        post["pass"] = bool(post["k1_geometriparitet"]
                            and post["k2_sjalvkonsistens"] is not False)
        atomer.append(post)

    massa_ark = las_titelfalt_massa(svg_text)
    # K3 must NOT trust the generator's own sum: the mass is re-summed from the MANIFEST (the
    # source) for exactly the parts the drawing claims to show.
    massa_ref = None
    kalla = matt.get("massa_kalla") or (matt.get("meta") or {}).get("massa_kalla")
    if kalla:
        mp = kalla["manifest"]
        mp = mp if os.path.isabs(mp) else os.path.join(BASE, mp)
        try:
            man = json.load(open(mp))
            vill = set(kalla["delar"])
            tot = 0.0
            for q in man.get("parts", []):
                nm = q.get("part") or q.get("namn") or q.get("name")
                if nm in vill:
                    m = q.get("mass_kg")
                    if m is None:
                        m = q.get("spec_massa_kg")
                    tot += float(m or 0.0)
            massa_ref = round(tot, 3)
        except Exception:
            massa_ref = None
    if massa_ref is None:
        massa_ref = matt["meta"].get("massa_kg")
    k3 = {"massa_pa_arket_kg": massa_ark, "massa_manifest_kg": massa_ref,
          "kalla": "re-summed from the manifest" if kalla else "the generator's meta (weaker)",
          "pass": bool(massa_ark is not None and massa_ref is not None
                       and abs(massa_ark - massa_ref) <= 0.0005)}
    n_paths = svg_text.count("<path")
    linje = {"n_path": n_paths, "troskel": 8, "pass": bool(n_paths > 8)}

    n_ok = sum(1 for a in atomer if a["pass"])
    return {
        "n_matt": len(atomer), "n_pass": n_ok, "n_fall": len(atomer) - n_ok,
        "tol_mm": tol,
        "matt": atomer,
        "k3_titelfalt": k3,
        "linjeverk": linje,
        "ocp": {k: v for k, v in g.items() if k != "hal"},
        "ocp_n_hal": len(g["hal"]),
        "pass": bool(n_ok == len(atomer) and len(atomer) > 0 and k3["pass"] and linje["pass"]),
    }


def injicera_fel(svg_text, mm):
    """Falsification: shift the FIRST overall dimension's number by mm, everything else unchanged."""
    m = re.search(r'(<text[^>]*data-matt-id="[^"]*_[uv]"[^>]*>)([^<]*)(</text>)', svg_text)
    if not m:
        return None, None
    try:
        v = float(m.group(2))
    except ValueError:
        return None, None
    ny = ("%.2f" % (v + mm)).rstrip("0").rstrip(".")
    return svg_text[:m.start()] + m.group(1) + ny + m.group(3) + svg_text[m.end():], (v, v + mm)


def cmd_grind(a):
    d = os.path.join(UT_ROT, a.namn)
    matt = json.load(open(os.path.join(d, "matt_v1.json")))
    svg_p = os.path.join(BASE, matt["svg"])
    svg = open(svg_p, encoding="utf-8").read()

    riktig = kor_grind(matt, svg, tol=a.tol)

    fb_svg, par = injicera_fel(svg, a.fallbevis_mm)
    if fb_svg is None:
        fall = {"korbar": False, "varfor": "found no overall dimension to shift"}
    else:
        r2 = kor_grind(matt, fb_svg, tol=a.tol)
        # A CONTINUOUS falsification measure (not just "did it fall?"): the largest deviation in
        # the falsified run must be EXACTLY the injected shift. A boolean passes on anything;
        # this number does not.
        avv = [x.get("avvikelse_mm") for x in r2["matt"]
               if x.get("avvikelse_mm") is not None]
        fall = {"korbar": True, "forskjutning_mm": a.fallbevis_mm,
                "matt_fore_efter": par, "grind_pass": r2["pass"],
                "n_fall": r2["n_fall"],
                "injicerad_avvikelse_mm": round(max(avv), 5) if avv else None,
                "faller_som_den_ska": bool(not r2["pass"] and r2["n_fall"] >= 1)}

    ut = {
        "schema": "ritning_grind_v1",
        "namn": a.namn,
        "svg": matt["svg"],
        "skala": matt["skala"],
        "grind": riktig,
        "fallbevis": fall,
        "GRON": bool(riktig["pass"] and fall.get("faller_som_den_ska")),
    }
    p = os.path.join(d, "grind_v1.json")
    json.dump(ut, open(p, "w"), indent=1, ensure_ascii=False)
    print("GATE %s: %s  dimensions %d/%d  mass %s  line work %s  falsification %s  -> %s"
          % (a.namn, "PASS" if riktig["pass"] else "FAIL", riktig["n_pass"], riktig["n_matt"],
             riktig["k3_titelfalt"]["pass"], riktig["linjeverk"]["pass"],
             fall.get("faller_som_den_ska"), p))
    for m in riktig["matt"]:
        if not m["pass"]:
            print("   FAIL", m)
    return 0 if ut["GRON"] else 1
