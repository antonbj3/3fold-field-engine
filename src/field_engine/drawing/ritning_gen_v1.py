#!/usr/bin/env python3
"""ritning_gen_v1 -- generator: STEP in, a real technical drawing out (SVG, and PDF if available).

The route is measured, not assumed:
  * FreeCAD 1.1 runs in CONSOLE mode (no GUI, no Xvfb). TechDraw's page SVG export lives in
    TechDrawGui and is GUI-bound, but the VIEW engine is not: TechDraw::DrawViewPart plus
    TechDraw.viewPartAsSvg() yields OCCT's hidden-line-removal projection as SVG paths headless.
    The sheet (frame, title block, dimensioning, parts list) is composed here, on top of FreeCAD's
    OWN ISO 5457/7200 templates, which are copied out of the FreeCAD installation on first use and
    sha-recorded in the measurement JSON.
  * Declared reframe: "SVG via TechDraw headless" means TechDraw's PROJECTION headless plus our own
    sheet composition. The alternative (a GUI under Xvfb) would have given TechDraw's own
    dimensioning but made dimension parity unmeasurable for us, because the dimensions would be
    TechDraw's internal objects rather than numbers we can gate per millimetre.

The 3D -> sheet frame is MEASURED, not guessed: ritning_gen_v1_fcworker.calibrate_frame tries all
four sign choices of (u,v) against the view's OWN view vertices and reports the hit fraction. A view
whose frame is contradicted gets NO dimensions (fail-closed).

FreeCAD is an optional dependency and has no default path. Set the environment variable FREECAD_CMD
(or pass --freecad-cmd) to the FreeCAD executable or AppImage that provides the console interpreter.

Run:
    # a detail drawing from one part of a parts manifest
    python ritning_gen_v1.py rita --namn bracket_v1 \
        --manifest artifacts/bracket_manifest_v1.json --part bracket_v1 --lokal
    # an assembly (general arrangement) drawing with a parts list
    python ritning_gen_v1.py rita --namn bracket_ga \
        --manifest artifacts/bracket_manifest_v1.json --assembly --format A2
    # the gate (dimension parity plus a falsification run)
    python ritning_gen_v1.py grind --namn bracket_v1
    # calibration selftest inside FreeCAD, plus an end-to-end run on the synthetic bracket
    python ritning_gen_v1.py selftest

Output: artifacts/<namn>/{<namn>.svg,<namn>.pdf,matt_v1.json,grind_v1.json}
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("FIELD_ENGINE_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(HERE)))
WORKER = os.path.join(HERE, "ritning_gen_v1_fcworker.py")
UT_ROT = os.path.join(HERE, "artifacts")
MALLAR = os.path.join(UT_ROT, "_mallar")


def freecad_cmd(explicit=None):
    """The FreeCAD executable providing the console interpreter. No default path."""
    cmd = explicit or os.environ.get("FREECAD_CMD")
    if not cmd:
        raise SystemExit(
            "FREECAD_CMD is not set. Point it at a FreeCAD 1.1 executable or AppImage, "
            "e.g. FREECAD_CMD=/path/to/FreeCAD-1.1.AppImage, or pass --freecad-cmd.")
    return cmd


# ISO 5457 A3/A2 template drawing area and title block placement (read out of the template files)
FORMAT = {
    # yta = (x0, y0, x1, y1_at_the_title_block); y1_full applies when the layout fits
    # TO THE LEFT of the title block (x < tb_x), where the views may run all the way down.
    # The y1 values come from the templates' OWN coordinates: the ISO 7200 block's top text band
    # sits 3 mm ABOVE title_block_frame (y=236.2 on A3, 359.2 on A2).
    "A3": {"mall": "A3_Landscape_ISO5457_advanced.svg", "w": 420.0, "h": 297.0,
           "yta": (25.0, 15.0, 405.0, 231.0), "y1_full": 282.0, "tb": (230.0, 239.0)},
    "A2": {"mall": "A2_Landscape_ISO5457_advanced.svg", "w": 594.0, "h": 420.0,
           "yta": (25.0, 15.0, 579.0, 353.0), "y1_full": 405.0, "tb": (404.0, 362.0)},
}
SKALOR = [50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

VYER = [
    {"id": "front", "dir": [0, -1, 0], "xdir": [1, 0, 0], "titel": "FRONT VIEW"},
    {"id": "left", "dir": [1, 0, 0], "xdir": [0, 1, 0], "titel": "VIEW FROM LEFT"},
    {"id": "top", "dir": [0, 0, -1], "xdir": [1, 0, 0], "titel": "TOP VIEW"},
    {"id": "iso", "dir": [0.5774, -0.5774, 0.5774], "xdir": [0.7071, 0.7071, 0.0],
     "titel": "ISOMETRIC", "smooth": True},
]

TEXT = 3.5          # ISO text height, mm
TUNN = 0.25         # dimension / extension line
GROV = 0.5          # visible edge line on the sheet


# ---------------------------------------------------------------- helpers


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def fmt_matt(v):
    """ISO dimension text: integers without decimals, otherwise at most 2 (and never -0)."""
    if abs(v - round(v)) < 5e-4:
        return "%d" % int(round(v))
    s = "%.2f" % v
    return s.rstrip("0").rstrip(".")


def valj_skala(bredd_mm, hojd_mm, max_b, max_h):
    for s in SKALOR:
        if bredd_mm * s <= max_b and hojd_mm * s <= max_h:
            return s
    return SKALOR[-1]


def skala_text(s):
    if s >= 1:
        return "%s : 1" % fmt_matt(s)
    return "1 : %s" % fmt_matt(1.0 / s)


def job_nyckel(job):
    """Cache key: the job, the STEP files' size/mtime and the WORKER's OWN sha.

    The layout (scale, placement, dimensioning) changes often; the HLR takes minutes to an hour and
    does NOT depend on the layout. The key must still fall as soon as the geometry OR the worker
    code changes, otherwise the cache is a way of drawing stale geometry.
    """
    h = hashlib.sha256()
    h.update(json.dumps(job, sort_keys=True).encode())
    for p in job["parts"]:
        ap = p["step"] if os.path.isabs(p["step"]) else os.path.join(BASE, p["step"])
        try:
            st = os.stat(ap)
            h.update(("%s:%d:%d" % (os.path.basename(ap), st.st_size, int(st.st_mtime))).encode())
        except OSError:
            h.update(("%s:MISSING" % os.path.basename(ap)).encode())
    h.update(sha256(WORKER).encode())
    return h.hexdigest()


def kor_worker(job, tag, cache_path=None, cmd=None, timeout=3600):
    """Start FreeCAD in console mode and run the fcworker against the job."""
    nyckel = job_nyckel(job) if job.get("parts") else None
    if cache_path and nyckel and os.path.exists(cache_path):
        try:
            c = json.load(open(cache_path))
            if c.get("_nyckel") == nyckel:
                print("HLR from cache (%s)" % os.path.basename(cache_path))
                return c["res"], None, cache_path
        except Exception:
            pass
    exe = freecad_cmd(cmd)
    jd = tempfile.mkdtemp(prefix="ritning_")
    jp = os.path.join(jd, "job_%s.json" % tag)
    op = os.path.join(jd, "out_%s.json" % tag)
    json.dump(job, open(jp, "w"))
    env = dict(os.environ, RITNING_JOB=jp, RITNING_JOB_OUT=op)
    p = subprocess.run([exe, "--console", WORKER], env=env,
                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    if not os.path.exists(op):
        sys.stderr.write(p.stdout.decode("utf-8", "replace")[-4000:])
        raise SystemExit("fcworker produced no output")
    res = json.load(open(op))
    if cache_path and nyckel:
        json.dump({"_nyckel": nyckel, "res": res}, open(cache_path, "w"))
    return res, jp, op


def hamta_mallar(fmt, cmd=None):
    """Copy FreeCAD's own sheet template for this format out of the FreeCAD installation.

    The templates ship with FreeCAD and are not carried by this repository; they are fetched once
    into artifacts/_mallar and reused.
    """
    namn = [FORMAT[fmt]["mall"], "A3_Landscape_blank.svg"]
    if all(os.path.exists(os.path.join(MALLAR, n)) for n in namn):
        return
    res, _, _ = kor_worker({"template_dir": MALLAR, "templates": namn}, "mallar",
                           cmd=cmd, timeout=600)
    for n, r in res.get("templates", {}).items():
        if "fel" in r:
            raise SystemExit("template %s: %s" % (n, r["fel"]))


# ---------------------------------------------------------------- frame maths (the driver's own,
# cross-validated against the worker's)


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def vy_axlar(spec):
    d = _norm(spec["dir"])
    x = spec["xdir"]
    k = sum(x[i] * d[i] for i in range(3))
    x = _norm([x[i] - k * d[i] for i in range(3)])
    y = _cross(d, x)
    return x, y


def punkt_till_vy(P, spec, calib):
    """3D -> view coordinate, with the sign choice the worker MEASURED."""
    x, y = vy_axlar(spec)
    gc = calib["gc"]
    su = calib.get("su", 1)
    sv = calib.get("sv", -1)
    u = sum(P[i] * x[i] for i in range(3)) - sum(gc[i] * x[i] for i in range(3))
    v = sum(P[i] * y[i] for i in range(3)) - sum(gc[i] * y[i] for i in range(3))
    return (su * u, sv * v)


# ---------------------------------------------------------------- SVG primitives (sheet mm)


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def txt(x, y, s, size=TEXT, anchor="middle", cls="", extra="", rot=None):
    tr = ' transform="rotate(%s %s %s)"' % (rot, x, y) if rot is not None else ""
    return ('<text x="%.3f" y="%.3f" font-family="DejaVu Sans, Arial, sans-serif" '
            'font-size="%.2f" text-anchor="%s" fill="#000"%s%s%s>%s</text>'
            % (x, y, size, anchor, ' class="%s"' % cls if cls else "", extra, tr, esc(s)))


def linje(x1, y1, x2, y2, w=TUNN, extra=""):
    return ('<line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="#000" '
            'stroke-width="%.3f"%s/>' % (x1, y1, x2, y2, w, extra))


def pil(x, y, dx, dy, l=3.0, b=1.0):
    """Filled ISO arrow head at (x,y) pointing along (dx,dy)."""
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    x1, y1 = x - ux * l + px * b / 2, y - uy * l + py * b / 2
    x2, y2 = x - ux * l - px * b / 2, y - uy * l - py * b / 2
    return ('<polygon points="%.3f,%.3f %.3f,%.3f %.3f,%.3f" fill="#000"/>'
            % (x, y, x1, y1, x2, y2))


def matt_horisontell(x1, x2, y_geom, y_dim, varde, mid, matt_id):
    """Horizontal dimension: extension lines from the geometry down to the dimension line,
    arrows, number."""
    o = []
    for xx in (x1, x2):
        o.append(linje(xx, y_geom, xx, y_dim + (2.0 if y_dim > y_geom else -2.0)))
    o.append(linje(x1, y_dim, x2, y_dim, extra=' class="mattlinje" data-matt-id="%s"' % matt_id))
    o.append(pil(x1, y_dim, -1, 0))
    o.append(pil(x2, y_dim, 1, 0))
    o.append(txt((x1 + x2) / 2.0, y_dim - 1.2, fmt_matt(varde), cls="matt",
                 extra=' data-matt-id="%s" data-matt-varde="%.4f"' % (matt_id, varde)))
    return "".join(o)


def matt_vertikal(y1, y2, x_geom, x_dim, varde, matt_id):
    o = []
    for yy in (y1, y2):
        o.append(linje(x_geom, yy, x_dim + (2.0 if x_dim > x_geom else -2.0), yy))
    o.append(linje(x_dim, y1, x_dim, y2, extra=' class="mattlinje" data-matt-id="%s"' % matt_id))
    o.append(pil(x_dim, y1, 0, -1))
    o.append(pil(x_dim, y2, 0, 1))
    ym = (y1 + y2) / 2.0
    # the rotated text is read by the gate through the same data attributes
    o.append('<text x="%.3f" y="%.3f" font-family="DejaVu Sans, Arial, sans-serif" '
             'font-size="%.2f" text-anchor="middle" fill="#000" class="matt" '
             'data-matt-id="%s" data-matt-varde="%.4f" transform="rotate(-90 %.3f %.3f)">'
             '%s</text>' % (x_dim - 1.2, ym, TEXT, matt_id, varde, x_dim - 1.2, ym,
                            esc(fmt_matt(varde))))
    return "".join(o)


def matt_diameter(cx, cy, r_ark, varde, antal, matt_id, ut_dx, ut_dy, lang=12.0):
    """Hole callout: arrow at the circle, leader line, 'n x OD' text."""
    n = math.hypot(ut_dx, ut_dy) or 1.0
    ux, uy = ut_dx / n, ut_dy / n
    ax, ay = cx + ux * r_ark, cy + uy * r_ark
    bx, by = ax + ux * lang, ay + uy * lang
    cx2 = bx + (6.0 if ux >= 0 else -6.0)
    lbl = ("%d x " % antal if antal > 1 else "") + "Ø" + fmt_matt(varde)
    o = [linje(ax, ay, bx, by), linje(bx, by, cx2, by), pil(ax, ay, -ux, -uy)]
    o.append(txt(cx2 + (1.5 if ux >= 0 else -1.5), by - 1.2, lbl, cls="matt",
                 anchor="start" if ux >= 0 else "end",
                 extra=' data-matt-id="%s" data-matt-varde="%.4f"' % (matt_id, varde)))
    return "".join(o)


# ---------------------------------------------------------------- template + title block


def las_mall(fmt):
    p = os.path.join(MALLAR, FORMAT[fmt]["mall"])
    return open(p, encoding="utf-8").read(), p


def fyll_titelfalt(mall_svg, falt):
    """Set the tspan text of every freecad:editable field."""
    ut = mall_svg
    for k, v in falt.items():
        pat = re.compile(r'(<text[^>]*freecad:editable="%s"[^>]*>)(.*?)(</text>)' % re.escape(k),
                         re.S)
        m = pat.search(ut)
        if not m:
            continue
        ut = ut[:m.start()] + m.group(1) + "<tspan>" + esc(v) + "</tspan>" + m.group(3) + ut[m.end():]
    return ut


def mall_kropp(mall_svg):
    """Take the template's content without the <svg> wrapper (we build our own sheet)."""
    i = mall_svg.find(">", mall_svg.find("<svg"))
    j = mall_svg.rfind("</svg>")
    return mall_svg[i + 1:j]


# ---------------------------------------------------------------- composition


def ga_urval(parts, frac=0.03):
    """General-arrangement practice: the views show the MAIN parts, not every bolt -- but never in
    a way that changes the overall dimension.

    Small parts (bbox diagonal < frac * the assembly's diagonal) are removed FROM THE VIEWS, after
    which each of the six extreme positions is restored: when no remaining part reaches an extreme,
    the filtered-out part that does is put back. The final check is two-sided: the union bbox of the
    remaining parts must be identical (<= 0.1 mm) to that of the whole selection. If it is not,
    NO filtering happens (fail-closed) -- a slow drawing is better than a dimensioned overall size
    that is not the assembly's. The parts list and the mass refer to the WHOLE selection, not to
    what is drawn, and that is stated on the sheet.
    """
    med = [p for p in parts if p.get("bbox")]
    if len(med) < 12:
        return parts, {"aktiv": False, "varfor": "too few parts with a bbox"}

    def uni(ps):
        b = [min(q["bbox"][i] for q in ps) for i in range(3)]
        return b + [max(q["bbox"][i + 3] for q in ps) for i in range(3)]

    full = uni(med)
    diag = math.sqrt(sum((full[i + 3] - full[i]) ** 2 for i in range(3)))

    def d(p):
        b = p["bbox"]
        return math.sqrt(sum((b[i + 3] - b[i]) ** 2 for i in range(3)))

    keep = [p for p in med if d(p) >= frac * diag]
    bort = [p for p in med if d(p) < frac * diag]
    if not keep:
        return parts, {"aktiv": False, "varfor": "the filter removed everything"}
    ater = []
    for i in range(6):
        j = i
        while True:
            kb = uni(keep)
            if abs(kb[j] - full[j]) <= 0.1:
                break
            kand = None
            for p in bort:
                if p in ater:
                    continue
                if abs(p["bbox"][j] - full[j]) <= 0.1:
                    kand = p
                    break
            if kand is None:
                break
            keep.append(kand)
            ater.append(kand)
    kb = uni(keep)
    avv = max(abs(kb[i] - full[i]) for i in range(6))
    if avv > 0.1:
        return parts, {"aktiv": False,
                       "varfor": "the overall dimension would have changed (%.3f mm)" % avv}
    return keep, {"aktiv": True, "frac": frac, "n_fore": len(med), "n_ritade": len(keep),
                  "n_bortfiltrerade": len(med) - len(keep), "n_ateradda": len(ater),
                  "bbox_avvikelse_mm": round(avv, 5)}


def bygg_ritning(namn, delar, meta, fmt="A3", vy_ids=("front", "left", "top", "iso"),
                 hidden=False, bom=None, ut_dir=None, grov=False, urvalsnot=None, cmd=None):
    ut_dir = ut_dir or os.path.join(UT_ROT, namn)
    os.makedirs(ut_dir, exist_ok=True)
    hamta_mallar(fmt, cmd=cmd)
    specar = [v for v in VYER if v["id"] in vy_ids]
    job = {
        "base_dir": BASE,
        "parts": delar,
        "views": [{"id": v["id"], "dir": v["dir"], "xdir": v["xdir"],
                   "hidden": hidden and v["id"] != "iso",
                   "smooth": (v["id"] == "iso") and not grov,
                   "coarse": grov} for v in specar],
        "holes": meta.get("hal", True),
    }
    res, jp, op = kor_worker(job, namn, cache_path=os.path.join(ut_dir, "_hlr_cache_v1.json"),
                             cmd=cmd)
    if "fel" in res:
        raise SystemExit("fcworker: %s" % res["fel"])
    geom = res["geom"]

    F = FORMAT[fmt]
    x0, y0, x1, y1 = F["yta"]

    # --- view sizes in model mm
    vinfo = {}
    for v in specar:
        d = res["views"].get(v["id"], {})
        if "svg" not in d:
            continue
        # The view's EXTENT comes from the projected 3D bbox corners (the worker gives them through
        # TechDraw's own projectPoint), not from the path data: the corners are exact and
        # independent of how the path string looks. bbox_svg is kept as diagnostics.
        bb = d["bbox_svg"] or [0, 0, 1, 1]
        hz = d.get("bbox3d_corners_svg")
        if hz:
            us = [c[0] for c in hz]
            vs = [c[1] for c in hz]
            w, h = max(us) - min(us), max(vs) - min(vs)
        else:
            w, h = bb[2] - bb[0], bb[3] - bb[1]
        vinfo[v["id"]] = {"spec": v, "d": d, "w": w, "h": h, "bb": bb}

    # --- LAYOUT: every view has a FOOTPRINT that contains its own dimension zones. The left-hand
    # dimension takes M_V mm to the left, the horizontal one M_H mm downwards and the view caption
    # a further M_T mm. The scale is chosen against the FOOTPRINTS, not against the raw geometry,
    # otherwise the neighbouring view collides with the dimension line.
    M_V, M_H, M_T, LUFT = 15.0, 15.0, 8.0, 12.0

    def layout(s):
        """-> (pos, bbox) in local mm for the scale s. bbox = (w, h) including dimension zones."""
        p = {}
        fw = vinfo["front"]["w"] * s if "front" in vinfo else 0.0
        fh = vinfo["front"]["h"] * s if "front" in vinfo else 0.0
        lw = vinfo["left"]["w"] * s if "left" in vinfo else 0.0
        lh = vinfo["left"]["h"] * s if "left" in vinfo else 0.0
        tw = vinfo["top"]["w"] * s if "top" in vinfo else 0.0
        th = vinfo["top"]["h"] * s if "top" in vinfo else 0.0
        cx = M_V + fw / 2.0
        cy = fh / 2.0
        if "front" in vinfo:
            p["front"] = (cx, cy)
        if "left" in vinfo:
            p["left"] = (cx + fw / 2.0 + LUFT + M_V + lw / 2.0, cy)
        if "top" in vinfo:
            p["top"] = (cx, cy + fh / 2.0 + M_H + M_T + LUFT + th / 2.0)
        hoger = max([p[k][0] + (vinfo[k]["w"] * s) / 2.0 for k in p] or [0])
        nedre = max([p[k][1] + (vinfo[k]["h"] * s) / 2.0 + M_H + M_T for k in p] or [0])
        return p, (hoger, nedre)

    y1_full = F.get("y1_full", y1)
    tbx = F["tb"][0]
    s = SKALOR[-1]
    for kand in SKALOR:
        _, (bw, bh) = layout(kand)
        # if the block fits to the left of the title block it may use the whole sheet height
        tak = y1_full if (x0 + bw) <= (tbx - 6.0) else y1
        if bw <= (x1 - x0) and bh <= (tak - y0):
            s = kand
            break
    pos_lokal, (bw, bh) = layout(s)
    ox = x0 + max(0.0, ((x1 - x0) - bw) * 0.12)
    oy = y0 + max(0.0, ((y1 - y0) - bh) * 0.10)
    pos = {k: (ox + v[0], oy + v[1]) for k, v in pos_lokal.items()}

    # THE ISOMETRIC VIEW MUST NOT DRIVE THE MAIN SCALE. It is an orientation picture, not a
    # dimensioned view; it gets whatever space is left and its OWN scale (stated in its caption).
    s_iso = None
    if "iso" in vinfo:
        iw0, ih0 = vinfo["iso"]["w"], vinfo["iso"]["h"]
        # the free area is bounded by the title block AND by the parts list (which grows upwards
        # from the title block), otherwise the isometric caption lands on top of the list's.
        bom_topp = (F["tb"][1] - 22.0 - (len(bom["rader"]) + 1) * 5.0) if bom else F["tb"][1]
        fri = []
        hoger_kant = ox + bw
        fri.append((hoger_kant + LUFT, y0, x1, min(y1_full, bom_topp - 12.0)))
        fri.append((x0, oy + bh + LUFT, tbx - 6.0, y1_full))
        bast = None
        for (fx0, fy0, fx1, fy1) in fri:
            bw_f, bh_f = fx1 - fx0 - 6.0, fy1 - fy0 - 10.0
            if bw_f < 25 or bh_f < 25:
                continue
            for kand in SKALOR:
                if kand > s:
                    continue
                if iw0 * kand <= bw_f and ih0 * kand <= bh_f:
                    if bast is None or kand > bast[0]:
                        bast = (kand, fx0 + 3.0 + iw0 * kand / 2.0,
                                fy0 + 5.0 + ih0 * kand / 2.0)
                    break
        if bast:
            s_iso, ix, iy = bast
            pos["iso"] = (ix, iy)
        else:
            vinfo.pop("iso", None)

    body = []
    matt_lista = []
    rubriker = []

    for vid, (px, py) in pos.items():
        info = vinfo[vid]
        d = info["d"]
        sv_ = s_iso if (vid == "iso" and s_iso) else s
        inner = re.sub(r'stroke-width="[0-9.]+"', 'stroke-width="%.4f"' % (GROV / sv_),
                       d["svg"], count=1)
        body.append('<g transform="translate(%.4f,%.4f) scale(%.6f)">%s</g>'
                    % (px, py, sv_, inner))
        rub = info["spec"]["titel"]
        if vid == "iso" and s_iso and abs(s_iso - s) > 1e-9:
            rub = "%s  (%s)" % (rub, skala_text(s_iso))
        rubriker.append((vid, px, py + (info["h"] * sv_) / 2.0
                         + (M_H if vid != "iso" else 4.0) + 5.0, rub))

    # --- overall dimensions per view (only views with a MEASURED frame)
    for vid in ("front", "left", "top"):
        if vid not in pos:
            continue
        if not geom.get("mattbara"):
            body.append(txt(pos[vid][0], pos[vid][1] + (vinfo[vid]["h"] * s) / 2.0 + 11.0,
                            "DIMENSIONS OMITTED: no tight bbox (%s)"
                            % geom.get("bbox_optimal_fel", "?"), size=2.6))
            continue
        info = vinfo[vid]
        d = info["d"]
        cal = d.get("calib", {})
        if not cal.get("ok"):
            body.append(txt(pos[vid][0], pos[vid][1] + (info["h"] * s) / 2.0 + 11.0,
                            "DIMENSIONS OMITTED: the frame is contradicted "
                            "(hit fraction %.3f, best alternative %.3f)"
                            % (cal.get("traffandel", -1),
                               cal.get("basta_alternativ_traffandel", -1)), size=2.6))
            continue
        px, py = pos[vid]
        corners = d["bbox3d_corners_svg"]
        us = [c[0] for c in corners]
        vs = [c[1] for c in corners]
        u_min, u_max, v_min, v_max = min(us), max(us), min(vs), max(vs)
        # sheet coordinates for the AABB corners
        ax1 = px + u_min * s
        ax2 = px + u_max * s
        ay1 = py + v_min * s
        ay2 = py + v_max * s
        xax, yax = vy_axlar(info["spec"])
        Lx = geom["L_mm"] * abs(xax[0]) + geom["B_mm"] * abs(xax[1]) + geom["H_mm"] * abs(xax[2])
        Lv = geom["L_mm"] * abs(yax[0]) + geom["B_mm"] * abs(yax[1]) + geom["H_mm"] * abs(yax[2])
        y_dim = ay2 + 11.0
        x_dim = ax1 - 11.0
        body.append(matt_horisontell(ax1, ax2, ay2, y_dim, Lx, (ax1 + ax2) / 2, "%s_u" % vid))
        body.append(matt_vertikal(ay1, ay2, ax1, x_dim, Lv, "%s_v" % vid))
        matt_lista.append({"id": "%s_u" % vid, "vy": vid, "typ": "huvudmatt_u",
                           "varde_mm": round(Lx, 4), "axel": xax,
                           "ritad_langd_mm": round(abs(ax2 - ax1), 4), "skala": s})
        matt_lista.append({"id": "%s_v" % vid, "vy": vid, "typ": "huvudmatt_v",
                           "varde_mm": round(Lv, 4), "axel": yax,
                           "ritad_langd_mm": round(abs(ay2 - ay1), 4), "skala": s})

        # hole callouts: grouped per diameter
        grupper = {}
        for h in d.get("holes_2d", []):
            grupper.setdefault(round(h["d_mm"], 3), []).append(h)
        for i, (dia, hs) in enumerate(sorted(grupper.items(), key=lambda kv: -kv[1][0]["d_mm"])[:6]):
            h0 = hs[0]
            cxp = px + h0["u"] * s
            cyp = py + h0["vv"] * s
            ux = 1.0 if (h0["u"] >= 0) else -1.0
            mid = "%s_hal%d" % (vid, i)
            body.append(matt_diameter(cxp, cyp, dia * s / 2.0, dia, len(hs), mid,
                                      ux * 0.85, -0.53, lang=10.0 + 7.0 * i))
            matt_lista.append({"id": mid, "vy": vid, "typ": "diameter", "antal": len(hs),
                               "varde_mm": round(dia, 4), "skala": s})

    for vid, rx, ry, rub in rubriker:
        body.append(txt(rx, ry, rub, size=3.0))

    # --- parts list for assemblies
    if bom:
        body.append(rita_bom(bom, F, fmt))

    # --- title block
    idag = datetime.date.today().isoformat()

    def klipp(v, n):
        """The title block cells are 80 mm wide at 3.5 mm text, i.e. about n characters. Longer
        text OVERWRITES the neighbouring field in the template and is therefore cut."""
        v = str(v)
        return v if len(v) <= n else v[:n - 1] + "…"

    falt = {
        "title": klipp(meta.get("titel", namn), 34),
        "supplementary_title_1": klipp(meta.get("maskin", ""), 34),
        "supplementary_title_2": ("MASS %.3f kg" % meta["massa_kg"])
        if meta.get("massa_kg") is not None else "MASS -",
        "part_material": klipp(meta.get("material", "-"), 26),
        "scale": skala_text(s),
        "drawing_number": meta.get("ritningsnr", "FE-%s" % namn.upper()),
        "date_of_issue": idag,
        "document_type": meta.get("dokumenttyp", "Component Drawing"),
        "document_status": "GENERATED FROM STEP",
        "creator": "ritning_gen_v1",
        "approval_person": "gate: ritning_paritet_v1",
        "responsible_department": "-",
        "revision_index": "A",
        "sheet_number": "1 / 1",
        "language_code": "EN",
        "general_tolerances": "ISO 2768-m",
        "legal_owner_1": "-",
        "legal_owner_2": klipp(meta.get("kalla_kort", ""), 22),
    }
    mall, mallp = las_mall(fmt)
    mall = fyll_titelfalt(mall, falt)
    kropp = mall_kropp(mall)

    if urvalsnot and urvalsnot.get("aktiv"):
        body.append(txt(x0 + 2.0, y0 - 2.0,
                        "VIEWS SHOW %d OF %d PARTS (parts under %.0f%% of the overall diagonal "
                        "omitted; the extremes are restored, the overall dimension is unchanged "
                        "to %.3f mm). THE PARTS LIST AND THE MASS REFER TO THE WHOLE ASSEMBLY."
                        % (urvalsnot["n_ritade"], urvalsnot["n_fore"], 100 * urvalsnot["frac"],
                           urvalsnot["bbox_avvikelse_mm"]), size=2.6, anchor="start"))
    projnot = txt(x0 + 2.0, y1 + 4.0, "PROJECTION: first angle (ISO 5456)  |  DIMENSIONS IN MM  |  "
                  "GENERATED FROM STEP BY ritning_gen_v1 (FreeCAD TechDraw HLR)",
                  size=2.6, anchor="start")

    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'xmlns:freecad="https://www.freecad.org/wiki/index.php?title=Svg_Namespace" '
           'width="%(w)smm" height="%(h)smm" viewBox="0 0 %(w)s %(h)s">'
           '<rect x="0" y="0" width="%(w)s" height="%(h)s" fill="#fff"/>'
           '%(mall)s<g id="ritning">%(body)s</g>%(not)s</svg>'
           % {"w": F["w"], "h": F["h"], "mall": kropp, "body": "".join(body), "not": projnot})

    svg_p = os.path.join(ut_dir, "%s.svg" % namn)
    open(svg_p, "w", encoding="utf-8").write(svg)
    pdf_p = svg_till_pdf(svg_p, F["w"], F["h"])

    matt_json = {
        "schema": "ritning_matt_v1",
        "namn": namn,
        "format": fmt,
        "skala": s,
        "skala_text": skala_text(s),
        "skala_iso": s_iso,
        "delar": delar,
        "meta": meta,
        "geom": geom,
        "vyer": {k: {"calib": vinfo[k]["d"].get("calib"), "n_paths": vinfo[k]["d"].get("n_paths"),
                     "bbox_svg": vinfo[k]["bb"], "dir": vinfo[k]["spec"]["dir"],
                     "xdir": vinfo[k]["spec"]["xdir"]} for k in vinfo},
        "matt": matt_lista,
        "massa_kalla": meta.get("massa_kalla"),
        "svg": os.path.relpath(svg_p, BASE),
        "pdf": os.path.relpath(pdf_p, BASE) if pdf_p else None,
        "mall": os.path.relpath(mallp, BASE),
        "mall_sha256": sha256(mallp),
        "freecad": res.get("freecad_version"),
        "bom": bom,
        "urval": urvalsnot,
        "grov_hlr": grov,
    }
    mp = os.path.join(ut_dir, "matt_v1.json")
    json.dump(matt_json, open(mp, "w"), indent=1, ensure_ascii=False)
    return matt_json


def rita_bom(bom, F, fmt):
    """ISO-style parts list: numbered, bottom up, above the title block."""
    tbx, tby = F["tb"]
    kol = [(0, 10, "POS"), (10, 62, "DESCRIPTION"), (72, 12, "QTY"),
           (84, 54, "MATERIAL"), (138, 26, "MASS kg")]
    bredd = 164.0
    rh = 5.0
    x = tbx + (180.0 - bredd)
    # 22 mm: the ISO 5457 template's projection symbol sits directly above the title block
    # (y ~226-238 on A3) -- the parts list has to start above THAT, not above the frame.
    y_bas = tby - 22.0
    o = []
    n = len(bom["rader"])
    for i, r in enumerate(bom["rader"]):
        yy = y_bas - i * rh
        o.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="none" '
                 'stroke="#000" stroke-width="0.25"/>' % (x, yy - rh, bredd, rh))
        vals = [str(r["pos"]), r["namn"][:34], str(r["antal"]), r["material"][:26],
                "%.3f" % r["massa_kg"]]
        for (cx, cw, _), v in zip(kol, vals):
            o.append(txt(x + cx + 1.2, yy - 1.5, v, size=2.6, anchor="start",
                         cls="bom", extra=' data-bom-pos="%s"' % r["pos"]))
        for cx, cw, _ in kol[1:]:
            o.append(linje(x + cx, yy - rh, x + cx, yy, w=0.25))
    yy = y_bas - n * rh
    o.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="none" stroke="#000" '
             'stroke-width="0.35"/>' % (x, yy - rh, bredd, rh))
    for cx, cw, h in kol:
        o.append(txt(x + cx + 1.2, yy - 1.5, h, size=2.6, anchor="start"))
        if cx:
            o.append(linje(x + cx, yy - rh, x + cx, yy, w=0.35))
    o.append(txt(x, yy - rh - 2.0, "PARTS LIST (%d items, %s of a total %s kg)"
                 % (n, fmt_matt(bom["summa_visad_kg"]), fmt_matt(bom["summa_total_kg"])),
                 size=2.6, anchor="start"))
    return "".join(o)


def svg_till_pdf(svg_path, w_mm, h_mm):
    """SVG -> PDF via a headless Chromium (playwright) when it is installed; otherwise None."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None
    pdf_path = os.path.splitext(svg_path)[0] + ".pdf"
    html = ("<html><head><meta charset='utf-8'><style>@page{size:%smm %smm;margin:0}"
            "html,body{margin:0;padding:0}svg{display:block}</style></head><body>%s</body></html>"
            % (w_mm, h_mm, open(svg_path, encoding="utf-8").read().split("?>", 1)[-1]))
    hp = svg_path + ".print.html"
    open(hp, "w", encoding="utf-8").write(html)
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page()
            pg.goto("file://" + hp)
            pg.pdf(path=pdf_path, width="%smm" % w_mm, height="%smm" % h_mm,
                   print_background=True, margin={"top": "0", "bottom": "0",
                                                  "left": "0", "right": "0"})
            b.close()
    except Exception as e:
        sys.stderr.write("PDF error: %s\n" % e)
        return None
    finally:
        if os.path.exists(hp):
            os.remove(hp)
    return pdf_path if os.path.exists(pdf_path) else None


# ---------------------------------------------------------------- input from a parts manifest


def las_manifest(path):
    """Read ONE parts register and normalise it to {namn, step, transform_mm, mass_kg, material}.

    Two manifest schemas are accepted: parts_manifest_v1 (namn/step/transform_mm, a 4x4 matrix) and
    a pure-translation variant (name/step_path/transform_translate_mm). The generator must not know
    both -- it only sees the normalised form.
    """
    apath = os.path.join(BASE, path) if not os.path.isabs(path) else path
    man = json.load(open(apath))
    mdir = os.path.dirname(apath)
    ut = []
    for p in man.get("parts", []):
        namn = p.get("namn") or p.get("name") or p.get("part")
        step = p.get("step") or p.get("step_path")
        if not step:
            # a manifest that lists only meshes: the STEP lies as <directory>/<part>.step
            kand = os.path.join(mdir, "%s.step" % (p.get("part") or namn or ""))
            if os.path.exists(kand):
                step = os.path.relpath(kand, BASE)
        if not step:
            continue
        tr = p.get("transform_mm")
        if tr is None and p.get("transform_translate_mm"):
            t = p["transform_translate_mm"]
            tr = [[1, 0, 0, t[0]], [0, 1, 0, t[1]], [0, 0, 1, t[2]], [0, 0, 0, 1]]
        massa = p.get("mass_kg")
        if massa is None:
            massa = p.get("spec_massa_kg")
        ut.append({"namn": namn, "part": p.get("part") or namn, "step": step,
                   "transform_mm": tr, "mass_kg": massa, "bbox": p.get("bbox"),
                   "grupp": p.get("grupp"),
                   "material": p.get("material", "-"),
                   "subassembly": p.get("subassembly"), "id": p.get("id")})
    return {"parts": ut, "schema": man.get("schema"), "ra": man}


def del_ur_manifest(man, namn):
    for p in man["parts"]:
        if p.get("namn") == namn or p.get("part") == namn or p.get("id") == namn:
            return p
    raise SystemExit("the part '%s' is not in the manifest" % namn)


def bom_ur_delar(parts, max_rader=22):
    agg = {}
    for p in parts:
        k = p.get("part") or p.get("namn")
        a = agg.setdefault(k, {"namn": k, "antal": 0, "massa_kg": 0.0,
                               "material": p.get("material", "-")})
        a["antal"] += 1
        a["massa_kg"] += float(p.get("mass_kg") or 0.0)
    rader = sorted(agg.values(), key=lambda r: -r["massa_kg"])
    total = sum(r["massa_kg"] for r in rader)
    visa = rader[:max_rader]
    if len(rader) > max_rader:
        rest = rader[max_rader:]
        visa.append({"namn": "OTHERS (%d items)" % len(rest),
                     "antal": sum(r["antal"] for r in rest),
                     "massa_kg": sum(r["massa_kg"] for r in rest), "material": "-"})
    for i, r in enumerate(visa, 1):
        r["pos"] = i
        r["massa_kg"] = round(r["massa_kg"], 3)
    return {"rader": visa, "summa_visad_kg": round(sum(r["massa_kg"] for r in visa), 3),
            "summa_total_kg": round(total, 3), "n_artiklar": len(rader)}


# ---------------------------------------------------------------- CLI


def cmd_rita(a):
    delar = []
    meta = {"titel": a.namn}
    bom = None
    urvalsnot = None
    if a.manifest:
        man = las_manifest(a.manifest)
        meta["maskin"] = os.path.basename(os.path.dirname(
            a.manifest if os.path.isabs(a.manifest) else os.path.join(BASE, a.manifest)))
        if a.assembly or a.prefix:
            ps = man["parts"]
            if a.prefix:
                ps = [p for p in ps if (p.get("namn") or "").startswith(a.prefix)]
            if a.urval:
                rx = re.compile(a.urval, re.I)
                ps = [p for p in ps
                      if rx.search("%s|%s|%s" % (p.get("part") or "", p.get("grupp") or "",
                                                 p.get("subassembly") or ""))]
            if not ps:
                raise SystemExit("no parts matched")
            ritade, urvalsnot = (ga_urval(ps, a.ga_filter) if a.ga_filter > 0
                                 else (ps, {"aktiv": False, "varfor": "not requested"}))
            if urvalsnot.get("aktiv"):
                print("GA SELECTION: %d of %d parts drawn (%d put back for the extremes), "
                      "overall dimension unchanged to %.3f mm"
                      % (urvalsnot["n_ritade"], urvalsnot["n_fore"], urvalsnot["n_ateradda"],
                         urvalsnot["bbox_avvikelse_mm"]))
            elif a.ga_filter > 0:
                print("GA SELECTION OFF: %s" % urvalsnot.get("varfor"))
            for p in ritade:
                delar.append({"name": p.get("namn"), "step": p["step"],
                              "transform": None if a.lokal else p.get("transform_mm")})
            meta["massa_kg"] = round(sum(float(p.get("mass_kg") or 0) for p in ps), 3)
            meta["massa_kalla"] = {"manifest": a.manifest,
                                   "delar": [p.get("part") or p.get("namn") for p in ps]}
            mats = sorted({p.get("material", "-") for p in ps})
            meta["material"] = mats[0] if len(mats) == 1 else "MIXED (%d materials)" % len(mats)
            meta["dokumenttyp"] = "General Arrangement" if a.assembly else "Assembly Drawing"
            meta["titel"] = a.titel or a.namn
            bom = bom_ur_delar(ps, max_rader=a.max_bom)
        else:
            p = del_ur_manifest(man, a.part or a.namn)
            delar.append({"name": p.get("namn"), "step": p["step"],
                          "transform": None if a.lokal else p.get("transform_mm")})
            meta["massa_kg"] = p.get("mass_kg")
            meta["massa_kalla"] = {"manifest": a.manifest,
                                   "delar": [p.get("part") or p.get("namn")]}
            meta["material"] = p.get("material", "-")
            meta["dokumenttyp"] = "Component Drawing"
            meta["titel"] = a.titel or p.get("namn")
            meta["kalla_kort"] = os.path.basename(p["step"])
    else:
        for sp in a.step:
            delar.append({"name": os.path.basename(sp), "step": sp, "transform": None})
        meta["massa_kg"] = a.massa
        meta["material"] = a.material or "-"
        meta["dokumenttyp"] = "Component Drawing"
        meta["titel"] = a.titel or a.namn
    meta["hal"] = not a.utan_hal
    vy = tuple(a.vyer.split(",")) if a.vyer else ("front", "left", "top", "iso")
    out = bygg_ritning(a.namn, delar, meta, fmt=a.format, vy_ids=vy, bom=bom,
                       hidden=a.dolda, grov=a.grov, urvalsnot=urvalsnot,
                       cmd=getattr(a, "freecad_cmd", None))
    print("DRAWING %s  scale %s  views %s  dimensions %d  svg %s  pdf %s"
          % (a.namn, out["skala_text"], ",".join(out["vyer"]), len(out["matt"]),
             out["svg"], out["pdf"]))
    return 0


def cmd_selftest(a):
    """Calibration proof inside FreeCAD, then a full run plus gate on the synthetic bracket."""
    res, _, _ = kor_worker({"selftest": True}, "selftest", cmd=getattr(a, "freecad_cmd", None),
                           timeout=900)
    st = res["selftest"]
    ok = True
    for vid, c in st.items():
        sar = c.get("sarskiljbar")
        print("CALIB %-6s hit fraction %s  variants %s  distinguishable %s  ok %s"
              % (vid, c.get("traffandel"), c.get("varianter_traffandel"), sar, c.get("ok")))
        ok = ok and bool(c.get("ok")) and bool(sar)
    print("FREECAD %s" % (res.get("freecad_version") or [None])[0:4])

    sys.path.insert(0, os.path.join(BASE, "examples", "parts"))
    import bracket_step_v1 as BS
    man_path = BS.write_step_and_manifest(os.path.join(UT_ROT, "bracket_synth"))
    print("SYNTHETIC BRACKET manifest %s" % os.path.relpath(man_path, BASE))

    class _A:
        namn = "bracket_v1"
        titel = None
        manifest = man_path
        part = "bracket_v1"
        prefix = None
        urval = None
        assembly = False
        lokal = True
        step = []
        massa = None
        material = None
        format = "A3"
        vyer = "front,left,top,iso"
        utan_hal = False
        dolda = False
        max_bom = 22
        ga_filter = 0.0
        grov = False
        freecad_cmd = getattr(a, "freecad_cmd", None)
    cmd_rita(_A())

    class _G:
        namn = "bracket_v1"
        tol = 0.1
        fallbevis_mm = 1.0
    rc = _grind(_G())
    print("SELFTEST %s" % ("PASS" if (ok and rc == 0) else "FAIL"))
    return 0 if (ok and rc == 0) else 1


def _grind(a):
    sys.path.insert(0, HERE)
    from ritning_paritet_v1 import cmd_grind
    return cmd_grind(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freecad-cmd", help="FreeCAD 1.1 executable or AppImage "
                                          "(defaults to the environment variable FREECAD_CMD)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rita")
    r.add_argument("--namn", required=True)
    r.add_argument("--titel")
    r.add_argument("--manifest")
    r.add_argument("--part")
    r.add_argument("--prefix")
    r.add_argument("--urval", help="regex against 'part|grupp|subassembly' -- selects a subset")
    r.add_argument("--assembly", action="store_true")
    r.add_argument("--lokal", action="store_true",
                   help="ignore the manifest transform (draw the part in its own frame)")
    r.add_argument("--step", nargs="*", default=[])
    r.add_argument("--massa", type=float)
    r.add_argument("--material")
    r.add_argument("--format", default="A3", choices=list(FORMAT))
    r.add_argument("--vyer")
    r.add_argument("--utan-hal", action="store_true")
    r.add_argument("--dolda", action="store_true",
                   help="draw hidden edges (inner detail) in the straight views")
    r.add_argument("--max-bom", type=int, default=22)
    r.add_argument("--ga-filter", type=float, default=0.0,
                   help="GA: exclude parts smaller than FRAC of the overall diagonal FROM THE "
                        "VIEWS (extremes restored; 0 = off). The parts list and mass are not "
                        "affected.")
    r.add_argument("--grov", action="store_true",
                   help="CoarseView HLR (fast polygonal projection for large assemblies)")
    r.set_defaults(func=cmd_rita)

    g = sub.add_parser("grind")
    g.add_argument("--namn", required=True)
    g.add_argument("--tol", type=float, default=0.1)
    g.add_argument("--fallbevis-mm", type=float, default=1.0)
    g.set_defaults(func=_grind)

    s = sub.add_parser("selftest")
    s.set_defaults(func=cmd_selftest)

    a = ap.parse_args()
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
