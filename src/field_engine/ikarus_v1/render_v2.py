#!/usr/bin/env python3
"""Sphere-tracing renderer: turns an SDF expression tree straight into a pixel image.

No tessellation and no marching cubes are involved in the render itself. Method:
  1. ray generation: pinhole camera (eye/target/up/vfov), per-pixel primary ray direction, vectorised;
  2. empty-space skip: closed-form ray/AABB slab test against each part's own box, so rays that miss
     never evaluate the field and rays that hit start marching at the box entry, not at the eye;
  3. sphere trace: step by the current SDF value (the field is Lipschitz-1), terminate on |sdf| < eps
     (hit), on passing the box exit (miss) or on exhausting max_steps (miss, declared);
  4. normals from a central-difference gradient at the hit points only;
  5. part id per hit point: evaluate each part's own SDF and take the one closest to its own zero set,
     since the union node only returns the combined distance;
  6. shading: Lambert diffuse plus ambient plus a second sphere trace toward the light, with the classic
     min(k*d/t) penumbra estimate taken from that march's own samples.

Correctness gate: an independent silhouette is rasterised from triangle meshes obtained by marching
cubes of the same trees, using the identical camera projection (ray generation and vertex projection are
algebraic inverses), and the IoU between the traced foreground mask and that silhouette is the
machine check.

The scene is analytic: a plate with a through bore and a ball, both expr.py primitive trees.
Run `python render_v2.py [--width 480 --height 270]`; writes PNGs and a report JSON under artifacts/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expr as E
import eval_warp as W
import queries as Q

OUT_DIR = os.path.join(HERE, "artifacts", "render_v2")

# ------------------------------------------------------------------------------- analytic scene
PART_COLORS = [(0.72, 0.55, 0.22), (0.20, 0.45, 0.75)]


def build_scene():
    """Builds the two-part analytic scene.

    Returns (union_node, [part nodes]): a plate with a through bore, and a ball beside it.
    """
    plate = E.subtract(E.box(half_extents=(30.0, 22.0, 8.0), center=(0.0, 0.0, 0.0)),
                       E.cylinder(radius=9.0, height=60.0, axis="z", center=(0.0, 0.0, 0.0)), k=0.0)
    ball = E.sphere(radius=14.0, center=(52.0, 0.0, 4.0))
    parts = [plate, ball]
    node = parts[0]
    for p in parts[1:]:
        node = E.union(node, p, k=0.0)
    return node, parts


def scene_bbox_mm(parts):
    """Returns (union_lo, union_hi, per_part_boxes) in mm.

    Each part is marched from its own box entry rather than the outer union box entry, so a ray crossing
    the gap between two parts never starts inside a region no part covers.
    """
    boxes = [Q.bbox_of(p, pad=1.0) for p in parts]
    lo = np.min(np.stack([b[0] for b in boxes]), axis=0)
    hi = np.max(np.stack([b[1] for b in boxes]), axis=0)
    return lo, hi, boxes


# --------------------------------------------------------------------------------------------- camera
class Camera:
    def __init__(self, eye, target, up, vfov_deg, width, height):
        self.eye = np.asarray(eye, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up, dtype=np.float64))
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        self.forward, self.right, self.up = forward, right, true_up
        self.width, self.height = width, height
        self.aspect = width / height
        self.tan_fov = np.tan(np.radians(vfov_deg) / 2.0)

    def ray_dirs(self):
        cols = np.arange(self.width)
        rows = np.arange(self.height)
        ndc_x = (cols + 0.5) / self.width * 2.0 - 1.0          # (-1,1) left->right
        ndc_y = 1.0 - (rows + 0.5) / self.height * 2.0          # (-1,1) top->bottom row
        gx, gy = np.meshgrid(ndc_x, ndc_y)                      # (H,W)
        gx = gx.reshape(-1); gy = gy.reshape(-1)
        d = (self.forward[None, :]
             + gx[:, None] * self.tan_fov * self.aspect * self.right[None, :]
             + gy[:, None] * self.tan_fov * self.up[None, :])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.tile(self.eye[None, :], (d.shape[0], 1))
        return o, d

    def project(self, V):
        """V: (N,3) world points -> (N,2) float pixel coords (col,row); also returns z_cam (N,) for a
        front/behind-eye validity mask (independent of the sphere-tracer, used only by the mesh-baseline
        rasterizer -- the algebraic inverse of ray_dirs())."""
        rel = V - self.eye[None, :]
        zc = rel @ self.forward
        xc = rel @ self.right
        yc = rel @ self.up
        safe_zc = np.where(np.abs(zc) < 1e-9, 1e-9, zc)
        ndc_x = xc / (safe_zc * self.tan_fov * self.aspect)
        ndc_y = yc / (safe_zc * self.tan_fov)
        col = (ndc_x + 1.0) / 2.0 * self.width - 0.5
        row = (1.0 - ndc_y) / 2.0 * self.height - 0.5
        return np.stack([col, row], axis=1), zc


def ray_aabb_slab(o, d, lo, hi):
    """Vectorized numpy slab test. Returns (t0, t1, hit) each (N,); t0/t1 undefined where hit is False."""
    inv_d = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    t_lo = (lo[None, :] - o) * inv_d
    t_hi = (hi[None, :] - o) * inv_d
    t_min = np.minimum(t_lo, t_hi)
    t_max = np.maximum(t_lo, t_hi)
    t0 = np.max(t_min, axis=1)
    t1 = np.min(t_max, axis=1)
    t0 = np.maximum(t0, 0.0)
    hit = t1 > t0
    return t0, t1, hit


# --------------------------------------------------------------------------------------------- sphere trace
def sphere_trace(node, o, d, per_leaf_boxes, eps=0.08, max_steps=128, exit_margin=2.0, device=W.DEVICE,
                  eval_fn=None):
    """Marches every ray in (o,d) simultaneously (no thread compaction -- terminated rays keep evaluating
    a harmless constant/near-zero step, simpler code, still ONE eval_batch launch per march step for the
    WHOLE frame regardless of how many rays already converged). Entry/exit distance is the NEAREST-entry/
    FARTHEST-exit across the leaves' own individual AABBs (not their outer union box -- see
    scene_bbox_mm()'s docstring for why the union box can contain real inter-leaf gaps). Returns dict with
    t, hit mask, steps array, launch count/timing for the profile."""
    # eval_fn defaults to eval_warp.eval_batch; a caller previewing a parameter-native tree can pass a
    # different evaluator with the same (node, points, device) signature.
    eval_fn = eval_fn or W.eval_batch
    n = o.shape[0]
    t0 = np.full(n, np.inf)
    t1 = np.full(n, -np.inf)
    box_hit = np.zeros(n, dtype=bool)
    for lo, hi in per_leaf_boxes:
        lt0, lt1, lhit = ray_aabb_slab(o, d, lo, hi)
        t0 = np.where(lhit, np.minimum(t0, lt0), t0)
        t1 = np.where(lhit, np.maximum(t1, lt1), t1)
        box_hit |= lhit
    t0 = np.where(box_hit, t0, 0.0)
    t1 = np.where(box_hit, t1, 0.0)

    t = t0.copy()
    # full-N bookkeeping arrays, always indexed by the CURRENT step's `idx` subset -- kept full-size (not
    # shrunk to len(idx)) so they stay valid to index next iteration even as `active` shrinks.
    prev_t = t0.copy()
    prev_sdf = np.full(n, np.inf)     # +inf on step 0 -> a sign-cross literally cannot fire before any sample exists
    hit = np.zeros(n, dtype=bool)
    active = box_hit.copy()          # rays that never touched any leaf's box are immediately-declared misses
    steps_taken = np.zeros(n, dtype=np.int32)
    t_max = t1 + exit_margin

    launch_times = []
    n_launches = 0
    for step in range(max_steps):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        p = o[idx] + t[idx, None] * d[idx]
        tt0 = time.time()
        sdf = eval_fn(node, p, device=device)
        launch_times.append(time.time() - tt0)
        n_launches += 1
        sdf = sdf.astype(np.float64)

        close = np.abs(sdf) < eps
        # Sign-crossing refine: an interpolated field is only approximately Lipschitz-1, so a full
        # |sdf|-length hop can overshoot through a thin feature into the interior; feeding the resulting
        # negative value into a max(raw_sdf, floor) step then crawls deeper forever. Detect
        # prev_sdf > 0 and sdf <= 0, take the linearly interpolated zero crossing between the two
        # samples as the hit point, and always step by |sdf|, never the raw signed value.
        prev_sdf_i = prev_sdf[idx]
        prev_t_i = prev_t[idx]
        crossed = np.isfinite(prev_sdf_i) & (prev_sdf_i > 0) & (sdf <= 0)   # isfinite excludes each ray's own first-ever sample (prev_sdf inited to +inf, not a real crossing)
        denom = np.where(crossed, prev_sdf_i - sdf, 1.0)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        safe_prev_sdf_i = np.where(crossed, prev_sdf_i, 0.0)   # avoid inf*0 -> nan warnings on masked-out lanes
        with np.errstate(invalid="ignore"):
            t_interp = prev_t_i + (safe_prev_sdf_i / denom) * (t[idx] - prev_t_i)

        newly_hit = close | crossed
        idx_hit = idx[newly_hit]
        hit[idx_hit] = True
        active[idx_hit] = False
        steps_taken[idx_hit] = step + 1
        refine_mask = crossed & ~close
        t[idx[refine_mask]] = t_interp[refine_mask]

        prev_t[idx] = t[idx]
        prev_sdf[idx] = sdf

        # step size: |sdf| (a true Lipschitz-1 safe-sphere-radius forward hop) -- see fix note above.
        step_dist = np.maximum(np.abs(sdf), eps * 0.5)
        t[idx] += step_dist
        overrun = t[idx] > t_max[idx]
        idx_over = idx[overrun]
        active[idx_over] = False
        steps_taken[idx[~newly_hit & ~overrun]] = step + 1

    return {
        "t": t, "hit": hit, "steps": steps_taken,
        "n_launches": n_launches, "launch_wall_s": launch_times,
        "n_rays": n, "n_box_candidates": int(box_hit.sum()),
    }


def classify_part(hit_pts, parts_leaves, device=W.DEVICE, eval_fn=None):
    """Which leaf owns each hit point: whichever leaf's OWN sdf magnitude is smaller at that point."""
    eval_fn = eval_fn or W.eval_batch
    vals = np.stack([np.abs(eval_fn(L, hit_pts, device=device)) for L in parts_leaves], axis=1)
    return np.argmin(vals, axis=1)


def shadow_factor(node, p0, light_dir, box_lo, box_hi, max_steps=48, k_soft=16.0, eps=0.05, device=W.DEVICE,
                   eval_fn=None):
    """Soft shadow via the sphere-trace penumbra estimate: march toward the light from each surface
    point and track min(k * sdf / t) along the way.

    Takes hit points, the light direction and the scene box; returns a (N,) visibility factor in [0,1].
    Uses the same field, one extra march, no separate occluder pass.
    """
    eval_fn = eval_fn or W.eval_batch
    n = p0.shape[0]
    t = np.full(n, 0.5, dtype=np.float64)   # start slightly off the surface (avoid immediate self-hit)
    res = np.ones(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    # exit bound: distance from p0 to the far side of the scene box along light_dir (generous constant fallback)
    diag = float(np.linalg.norm(box_hi - box_lo))
    t_max = diag * 1.5
    for _ in range(max_steps):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        p = p0[idx] + t[idx, None] * light_dir[None, :]
        sdf = eval_fn(node, p, device=device).astype(np.float64)
        blocked = sdf < eps * 0.25
        cand = np.clip(k_soft * sdf / np.maximum(t[idx], 1e-6), 0.0, 1.0)
        res[idx] = np.minimum(res[idx], cand)          # penumbra darkens as any step grazes an occluder
        res[idx[blocked]] = 0.0                          # a genuine occluder hit -> fully shadowed, final
        active[idx[blocked]] = False
        t[idx] += np.maximum(sdf, eps * 0.5)
        over = t[idx] > t_max
        active[idx[over]] = False
    return np.clip(res, 0.0, 1.0)


# --------------------------------------------------------------------------------------------- mesh-silhouette baseline (independent ground truth)
def marching_cubes_triangles(parts, boxes, pitch_mm=0.5, device=None):
    """Independent triangle meshes for the silhouette baseline.

    Takes the part nodes and their boxes; returns a list of (T,3,3) world-space triangle arrays obtained
    by marching cubes of each part's own field, never fed into the SDF march.
    """
    from skimage import measure
    device = device or W.DEVICE
    out = []
    for node, (lo, hi) in zip(parts, boxes):
        n = np.maximum(np.ceil((hi - lo) / pitch_mm).astype(int) + 1, 2)
        xs = [lo[a] + pitch_mm * np.arange(n[a]) for a in range(3)]
        X, Y, Z = np.meshgrid(xs[0], xs[1], xs[2], indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        vals = W.eval_batch(node, pts, device=device).reshape(*n)
        verts, faces, _n, _v = measure.marching_cubes(vals, level=0.0,
                                                      spacing=(pitch_mm, pitch_mm, pitch_mm))
        verts = verts + lo[None, :]
        out.append(verts[faces])
    return out


def mesh_silhouette_mask(cam, width, height, tris_per_part):
    """Rasterizes the baseline triangle meshes with the SAME camera projection as the sphere tracer.

    Takes the camera, the image size and one (T,3,3) array per part; returns (mask, n_triangles_drawn).
    Pure polygon fill, painter's algorithm without a depth test, which is correct for a binary
    silhouette regardless of draw order or facet orientation.
    """
    img = Image.new("L", (width, height), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    n_tris_drawn = 0
    for tris in tris_per_part:
        tris = np.asarray(tris, dtype=np.float64)   # (T,3,3)
        V = tris.reshape(-1, 3)
        uv, zc = cam.project(V)
        uv = uv.reshape(-1, 3, 2)
        zc = zc.reshape(-1, 3)
        front = (zc > 1e-6).all(axis=1)
        for t_idx in np.nonzero(front)[0]:
            poly = [(float(uv[t_idx, k, 0]), float(uv[t_idx, k, 1])) for k in range(3)]
            draw.polygon(poly, fill=255)
            n_tris_drawn += 1
    mask = np.array(img) > 0
    return mask, n_tris_drawn


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


# --------------------------------------------------------------------------------------------- one full render
LIGHT_DIR = np.array([0.45, -0.55, 0.70])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
BG = np.array([0.06, 0.07, 0.09])
AMBIENT = 0.16


def render_view(node, leaves, cam, box_lo, box_hi, per_leaf_boxes, device=W.DEVICE, eps=0.08, max_steps=128,
                 eval_fn=None, grad_fn=None, part_colors=None):
    """Renders one view.

    Takes the scene node, the per-part nodes (may be empty, in which case per-part classification is
    skipped), a Camera, the scene box and the per-part boxes. eval_fn/grad_fn default to
    eval_warp.eval_batch/gradient_batch and can be swapped for an evaluator with the same signature.
    Returns (image (H,W,3) float, hit mask (H,W), profile dict).
    """
    eval_fn = eval_fn or W.eval_batch
    grad_fn = grad_fn or W.gradient_batch
    colors = np.asarray(part_colors if part_colors is not None else PART_COLORS, dtype=np.float64)
    width, height = cam.width, cam.height
    o, d = cam.ray_dirs()

    t_trace0 = time.time()
    tr = sphere_trace(node, o, d, per_leaf_boxes, eps=eps, max_steps=max_steps, device=device, eval_fn=eval_fn)
    trace_wall_s = time.time() - t_trace0

    hit = tr["hit"]
    img = np.tile(BG[None, :], (width * height, 1))

    if hit.any():
        hit_pts = o[hit] + tr["t"][hit, None] * d[hit]
        t_norm0 = time.time()
        grad = grad_fn(node, hit_pts, h=0.06, device=device)
        norm_wall_s = time.time() - t_norm0
        gnorm = np.linalg.norm(grad, axis=1, keepdims=True)
        normals = grad / np.where(gnorm < 1e-9, 1e-9, gnorm)

        if leaves:
            t_part0 = time.time()
            part_id = classify_part(hit_pts, leaves, device=device, eval_fn=eval_fn)
            part_wall_s = time.time() - t_part0
            base_color = colors[part_id]
        else:
            part_wall_s = 0.0
            base_color = np.tile(colors[0][None, :], (hit_pts.shape[0], 1))

        t_shadow0 = time.time()
        shadow = shadow_factor(node, hit_pts + normals * 0.15, LIGHT_DIR, box_lo, box_hi, device=device,
                                eval_fn=eval_fn)
        shadow_wall_s = time.time() - t_shadow0

        ndotl = np.clip(np.sum(normals * LIGHT_DIR[None, :], axis=1), 0.0, 1.0)
        lit = AMBIENT + (1.0 - AMBIENT) * ndotl * (0.3 + 0.7 * shadow)
        img[hit] = base_color * lit[:, None]
    else:
        norm_wall_s = part_wall_s = shadow_wall_s = 0.0

    img = np.clip(img, 0.0, 1.0).reshape(height, width, 3)
    profile = {
        "trace_wall_s": trace_wall_s,
        "normal_wall_s": norm_wall_s,
        "part_classify_wall_s": part_wall_s,
        "shadow_wall_s": shadow_wall_s,
        "n_eval_batch_launches_trace": tr["n_launches"],
        "n_rays_total": tr["n_rays"],
        "n_rays_box_candidates": tr["n_box_candidates"],
        "n_rays_hit": int(hit.sum()),
        "mean_steps_to_hit": float(tr["steps"][hit].mean()) if hit.any() else 0.0,
        "max_steps_to_hit": int(tr["steps"][hit].max()) if hit.any() else 0,
    }
    return img, hit.reshape(height, width), profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--max_steps", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--baseline_pitch_mm", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = W.DEVICE

    node, parts = build_scene()
    box_lo, box_hi, per_part_boxes = scene_bbox_mm(parts)
    center = (box_lo + box_hi) / 2.0
    extent = box_hi - box_lo
    diag = float(np.linalg.norm(extent))

    eye_iso = center + np.array([0.9, -1.1, 0.75]) * diag * 0.62
    eye_front = center + np.array([0.0, -1.0, 0.12]) * diag * 0.85
    eye_top = center + np.array([0.001, 0.001, 1.0]) * diag * 0.9
    up_top = np.array([0.0, 1.0, 0.0])

    views = [
        ("iso", eye_iso, np.array([0.0, 0.0, 1.0])),
        ("front", eye_front, np.array([0.0, 0.0, 1.0])),
        ("top", eye_top, up_top),
    ]

    tris_per_part = marching_cubes_triangles(parts, per_part_boxes, pitch_mm=args.baseline_pitch_mm,
                                             device=device)

    report = {
        "cell": "ikarus_v2_sphere_tracing_render",
        "device": device,
        "scene": {
            "parts": ["plate_with_bore", "ball"],
            "bbox_lo_mm": box_lo.tolist(), "bbox_hi_mm": box_hi.tolist(), "diag_mm": diag,
            "baseline_pitch_mm": args.baseline_pitch_mm,
            "baseline_n_triangles": [int(t.shape[0]) for t in tris_per_part],
        },
        "resolution": {"width": args.width, "height": args.height},
        "views": {},
    }

    for name, eye, up in views:
        cam = Camera(eye, center, up, vfov_deg=32.0, width=args.width, height=args.height)
        t0 = time.time()
        img, hit_mask, profile = render_view(node, parts, cam, box_lo, box_hi, per_part_boxes,
                                             device=device, eps=args.eps, max_steps=args.max_steps)
        wall_s = time.time() - t0

        png_path = os.path.join(OUT_DIR, f"sdf_v2_{name}.png")
        Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(png_path)

        t_mesh0 = time.time()
        mesh_mask, n_tris = mesh_silhouette_mask(cam, args.width, args.height, tris_per_part)
        mesh_wall_s = time.time() - t_mesh0
        mesh_png = os.path.join(OUT_DIR, f"mesh_baseline_{name}.png")
        Image.fromarray((mesh_mask.astype(np.uint8) * 255)).save(mesh_png)

        the_iou = iou(hit_mask, mesh_mask)

        report["views"][name] = {
            "png": os.path.relpath(png_path, HERE),
            "mesh_baseline_png": os.path.relpath(mesh_png, HERE),
            "eye_mm": eye.tolist(),
            "wall_s_total": wall_s,
            "profile": profile,
            "mesh_baseline_wall_s": mesh_wall_s,
            "mesh_baseline_n_triangles_drawn": n_tris,
            "silhouette_iou_vs_mesh_baseline": the_iou,
        }
        print(f"[{name}] wall={wall_s:.3f}s trace={profile['trace_wall_s']:.3f}s "
              f"hit={profile['n_rays_hit']}/{profile['n_rays_total']} IoU={the_iou:.4f}")

    iou_values = [v["silhouette_iou_vs_mesh_baseline"] for v in report["views"].values()]
    wall_values = [v["wall_s_total"] for v in report["views"].values()]
    report["summary"] = {
        "iou_min": min(iou_values), "iou_mean": sum(iou_values) / len(iou_values),
        "wall_s_max_view": max(wall_values), "wall_s_mean_view": sum(wall_values) / len(wall_values),
        "target_iou": 0.95, "target_wall_s_per_view": 10.0,
        "iou_gate_pass": min(iou_values) >= 0.95,
        "speed_gate_pass": max(wall_values) < 10.0,
    }

    report_path = os.path.join(OUT_DIR, "sphere_tracing_v2.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nreport -> {report_path}")
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["iou_gate_pass"]:
        raise SystemExit(f"IoU gate FAILED: min={report['summary']['iou_min']:.4f} < 0.95")


if __name__ == "__main__":
    main()
