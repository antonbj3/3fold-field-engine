#!/usr/bin/env python3
"""Synthetic duct-task generator: the data factory behind the duct corpus.

Generates synthetic duct tasks: a randomised box volume (declared base extent with +-50 % jitter per
axis), 2-6 randomised obstacles (boxes/cylinders, one of them always a cone), an inlet and an outlet
port on two randomly chosen distinct walls of the box (a 120x120 class flange plus mouth), and a
minimum bend radius drawn from a band around a declared floor.

Seed driven: numpy.random.default_rng(seed) is the only source of randomness, so the same seed gives
a byte-identical task (round-trip testable).

Format (the same NPZ schema the duct solvers read):
  <out_dir>/<uuid>/occupancy_task.npz  -- occupancy (bool as uint8), source_id (uint8),
                                          global_origin_mm, pitch_mm, shape
  <out_dir>/<uuid>/task.json           -- box_mm, pitch_mm, seed, ports{in,out}, obstacles[],
                                          min_bend_r_floor_mm, ir_params

Grid budget: pitch_mm is chosen per task so that Lx*Ly*Lz/pitch_mm^3 <= GRID_BUDGET (default 150 000,
half of a 300 000 point ceiling, leaving headroom because the solver runs SDF/smoothing on the same
grid several times per task).

Run: python duct_task_gen_v1.py --n 10 --seed 20260730 --out-dir <dir>
"""
import argparse
import json
import os
import uuid

import numpy as np

# Declared scale anchors of the synthetic corpus (override with the CLI flags where available).
WORKSPACE_BOX_MM = (600.0, 400.0, 250.0)   # full working volume the tasks are drawn inside
MIN_BEND_R_FLOOR_MM = 24.0                 # declared minimum bend-radius floor, band = [0.6x, 1.4x]

GRID_BUDGET = 150_000            # half of a 300k points-per-process ceiling, headroom for multi-pass SDF/smoothing
FLANGE_MM = (120.0, 120.0)       # port flange size, mm
FAN_OPEN_AREA_MM2 = 6400.0       # declared open area of the fan face, mm2
WALL_CLEAR_MM = 6.0               # min offset kept between any port/obstacle and the box shell

FACES = ["xlo", "xhi", "ylo", "yhi", "zlo", "zhi"]


def _real_map_box_mm():
    """World-mm extent of the working volume the tasks are drawn inside."""
    return np.asarray(WORKSPACE_BOX_MM, dtype=float)


def _min_bend_r_band_mm():
    """Band around the declared minimum-bend-radius floor: [0.6x, 1.4x]."""
    real = MIN_BEND_R_FLOOR_MM
    return real * 0.6, real * 1.4, real


def sample_box_mm(rng, real_box_mm):
    base = real_box_mm * 0.70
    jitter = rng.uniform(0.5, 1.5, size=3)
    box = base * jitter
    box = np.clip(box, 150.0, real_box_mm)   # never exceed the real owned map's own extent
    return box


def pick_pitch_mm(box_mm, budget=GRID_BUDGET):
    vol = float(np.prod(box_mm))
    pitch = (vol / budget) ** (1.0 / 3.0)
    return float(np.clip(pitch, 2.0, 12.0))


def sample_wall_point(rng, face, box_mm, size_mm, margin=WALL_CLEAR_MM):
    """A port center on `face`, in-plane coordinates kept >= size_mm/2+margin from every edge."""
    ax = {"xlo": 0, "xhi": 0, "ylo": 1, "yhi": 1, "zlo": 2, "zhi": 2}[face]
    other = [i for i in range(3) if i != ax]
    p = np.zeros(3)
    p[ax] = 0.0 if face.endswith("lo") else box_mm[ax]
    for o in other:
        lo = size_mm / 2 + margin
        hi = box_mm[o] - size_mm / 2 - margin
        if hi <= lo:
            lo, hi = box_mm[o] * 0.3, box_mm[o] * 0.7
        p[o] = rng.uniform(lo, hi)
    normal = np.zeros(3)
    normal[ax] = -1.0 if face.endswith("lo") else 1.0
    return p, normal, ax


def sample_obstacle(rng, kind, box_mm, sid, is_cone=False):
    """kind in {box, cylinder}; a cone (is_cone=True) is stored as a tapered cylinder (r0, r1, axis, h)
    -- the 'light-cone' obstacle: reserved source_id=10 (the convention the corpus readers expect for the
    projector light cone, negativ_volym_v2_sannljus docstring, duct_engine_v4.py:6-9)."""
    center = rng.uniform(box_mm * 0.2, box_mm * 0.8)
    if kind == "box":
        size = rng.uniform(20.0, np.minimum(box_mm * 0.35, 140.0))
        return dict(type="box", sid=int(sid), center_mm=center.tolist(), size_mm=size.tolist())
    axis = int(rng.integers(0, 3))
    h = float(rng.uniform(60.0, box_mm[axis] * 0.6))
    if is_cone:
        r0 = float(rng.uniform(15.0, 45.0))
        r1 = float(rng.uniform(5.0, r0 * 0.6))
        return dict(type="cone", sid=int(sid), center_mm=center.tolist(), axis=axis, height_mm=h,
                    r0_mm=r0, r1_mm=r1, light_cone=True)
    r = float(rng.uniform(10.0, 45.0))
    return dict(type="cylinder", sid=int(sid), center_mm=center.tolist(), axis=axis, height_mm=h, r_mm=r)


def rasterize_task(box_mm, pitch_mm, obstacles):
    shape = np.maximum(np.ceil(box_mm / pitch_mm).astype(int), 8)
    xs = (np.arange(shape[0]) + 0.5) * pitch_mm
    ys = (np.arange(shape[1]) + 0.5) * pitch_mm
    zs = (np.arange(shape[2]) + 0.5) * pitch_mm
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    occ = np.zeros(shape, dtype=np.uint8)
    sid = np.zeros(shape, dtype=np.uint8)
    for ob in obstacles:
        c = np.array(ob["center_mm"])
        if ob["type"] == "box":
            s = np.array(ob["size_mm"])
            m = (np.abs(gx - c[0]) <= s[0] / 2) & (np.abs(gy - c[1]) <= s[1] / 2) & (np.abs(gz - c[2]) <= s[2] / 2)
        else:
            ax = ob["axis"]
            coords = [gx, gy, gz]
            along = coords[ax] - c[ax]
            other = [coords[i] - c[i] for i in range(3) if i != ax]
            rad = np.sqrt(other[0] ** 2 + other[1] ** 2)
            h = ob["height_mm"]
            in_h = np.abs(along) <= h / 2
            if ob["type"] == "cylinder":
                m = in_h & (rad <= ob["r_mm"])
            else:  # cone: radius tapers linearly r0 (at -h/2) -> r1 (at +h/2)
                t = np.clip((along + h / 2) / h, 0.0, 1.0)
                r_at = ob["r0_mm"] * (1 - t) + ob["r1_mm"] * t
                m = in_h & (rad <= r_at)
        occ[m] = 1
        # first writer wins for overlapping obstacles (declared, not hidden): don't stomp an already-set sid
        newly = m & (sid == 0)
        sid[newly] = ob["sid"]
    return occ, sid, shape


def gen_one(rng, task_id, real_box_mm, rmin_lo, rmin_hi):
    box_mm = sample_box_mm(rng, real_box_mm)
    pitch_mm = pick_pitch_mm(box_mm)

    n_obs = int(rng.integers(2, 7))  # 2-6 inclusive
    obstacles = []
    cone_idx = int(rng.integers(0, n_obs))
    for i in range(n_obs):
        if i == cone_idx:
            obstacles.append(sample_obstacle(rng, "cylinder", box_mm, sid=10, is_cone=True))
        else:
            kind = "box" if rng.random() < 0.5 else "cylinder"
            obstacles.append(sample_obstacle(rng, kind, box_mm, sid=2 + i))

    face_in, face_out = rng.choice(FACES, size=2, replace=False)
    p_in, n_in, ax_in = sample_wall_point(rng, face_in, box_mm, FLANGE_MM[0])
    mynning_size = float(np.sqrt(FAN_OPEN_AREA_MM2))  # square-equivalent side of the real fan-open area
    p_out, n_out, ax_out = sample_wall_point(rng, face_out, box_mm, mynning_size)

    min_bend_r_mm = float(rng.uniform(rmin_lo, rmin_hi))

    occ, sid, shape = rasterize_task(box_mm, pitch_mm, obstacles)

    task = dict(
        task_id=task_id, seed=int(rng.bit_generator.state["state"]["state"]) if False else None,  # filled by caller
        box_mm=box_mm.tolist(), pitch_mm=pitch_mm, grid_shape=[int(x) for x in shape],
        n_gridpoints=int(np.prod(shape)),
        ports=dict(
            flange_in=dict(role="flange", wall=face_in, center_mm=p_in.tolist(), normal=n_in.tolist(),
                            axis=int(ax_in), size_mm=list(FLANGE_MM)),
            mynning_out=dict(role="mynning", wall=face_out, center_mm=p_out.tolist(), normal=n_out.tolist(),
                              axis=int(ax_out), size_mm=[mynning_size, mynning_size],
                              area_mm2=FAN_OPEN_AREA_MM2),
        ),
        obstacles=obstacles,
        min_bend_r_floor_mm=min_bend_r_mm,
        n_obstacles=n_obs,
        light_cone_source_id=10,
    )
    return task, occ, sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "duct_corpus"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    real_box_mm = _real_map_box_mm()
    rmin_lo, rmin_hi, rmin_real = _min_bend_r_band_mm()
    print(f"[task_gen] real_box_mm(anchor)={real_box_mm.tolist()} min_bend_r_band=[{rmin_lo:.1f},{rmin_hi:.1f}] "
          f"(real={rmin_real:.1f}mm) grid_budget={GRID_BUDGET}")

    master_rng = np.random.default_rng(args.seed)
    task_seeds = master_rng.integers(0, 2**31 - 1, size=args.n)

    written = []
    for i in range(args.n):
        tseed = int(task_seeds[i])
        rng = np.random.default_rng(tseed)
        task_id = f"t{args.seed}_{i:04d}_{uuid.UUID(int=rng.integers(0, 2**63)).hex[:8]}"
        task, occ, sid = gen_one(rng, task_id, real_box_mm, rmin_lo, rmin_hi)
        task["seed"] = tseed
        task["gen_seed_top"] = args.seed
        task["gen_index"] = i

        tdir = os.path.join(args.out_dir, task_id)
        os.makedirs(tdir, exist_ok=True)
        npz_path = os.path.join(tdir, "occupancy_task.npz")
        np.savez_compressed(npz_path, occupancy=occ, source_id=sid,
                             global_origin_mm=np.zeros(3), pitch_mm=np.float64(task["pitch_mm"]),
                             shape=np.array(task["grid_shape"]))
        json_path = os.path.join(tdir, "task.json")
        with open(json_path, "w") as f:
            json.dump(task, f, indent=1, ensure_ascii=False)
        written.append(dict(task_id=task_id, dir=tdir, n_gridpoints=task["n_gridpoints"],
                             occ_frac=float(occ.mean())))
        print(f"[task_gen] {i+1}/{args.n} {task_id} box_mm={[round(x,1) for x in task['box_mm']]} "
              f"pitch={task['pitch_mm']:.2f} grid={task['grid_shape']} occ_frac={occ.mean():.3f}")

    idx_path = os.path.join(args.out_dir, "_index.json")
    with open(idx_path, "w") as f:
        json.dump(dict(seed=args.seed, n=args.n, real_box_mm_anchor=real_box_mm.tolist(),
                        min_bend_r_band_mm=[rmin_lo, rmin_hi], tasks=written), f, indent=1)
    print(f"[task_gen] wrote {len(written)} tasks -> {args.out_dir} (index: {idx_path})")


if __name__ == "__main__":
    main()
