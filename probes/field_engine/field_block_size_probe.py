"""Measure smaller spatial blocks using the frozen multires implementation."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
from pathlib import Path
import numpy as np
import faltkarna_v1_multires as reference

ROOT = Path(__file__).resolve().parents[2]


def main():
    wp, _ = reference.M2S._import_warp()
    vertices, faces = reference._bracket_mesh()
    fine_pitch, _ = reference.M2S.valj_pitch_for_feature(vertices, faces, 2., "auto")
    coarse_pitch = 4 * fine_pitch
    origin = vertices.min(axis=0) - 3 * coarse_pitch
    coarse = reference._global_fall(wp, "cpu", vertices, faces, coarse_pitch, origin, "coarse")
    fine = reference._global_fall(wp, "cpu", vertices, faces, fine_pitch, origin, "fine")
    points = vertices[faces].mean(axis=1)
    rng = np.random.default_rng(20260912)
    points = points[rng.choice(len(points), size=min(4000, len(points)), replace=False)]
    exact = fine["prov"](points)
    limits = dict(median=float(np.median(np.abs(exact))),
                  p95=float(np.percentile(np.abs(exact), 95)))
    rows = []
    arrays = dict(points=points, fine_values=exact)
    for block in (8, 4, 2, 1):
        legs = []
        for leg in range(2):
            field, _ = reference.bygg_multires(
                wp, vertices, faces, coarse_pitch, origin, "cpu", block=block,
                sd_grov=coarse["sd"], gmin=coarse["gmin"], shape_l=coarse["shape_l"])
            values = reference.avstand(field, points)
            arrays[f"block_{block}_{leg}"] = values
            continuity = reference.kontinuitetscheck(field, max_punkter_per_face=2000)
            legs.append(dict(fingerprint=reference._fingerprint(field),
                             query_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                             samples=sum(tile.size for tile in field.tiles.values()),
                             cells=reference.aktiva_voxlar(field),
                             median=float(np.median(np.abs(values))),
                             p95=float(np.percentile(np.abs(values), 95)),
                             continuity_pass=bool(continuity["GRON"]),
                             continuity_jump=float(continuity["max_hopp_mm"])))
        gates = dict(repeat=legs[0] == legs[1],
                     memory=all(r["cells"] <= .5 * fine["aktiva_voxlar"] for r in legs),
                     accuracy=all(r["median"] <= limits["median"] and
                                  r["p95"] <= limits["p95"] for r in legs),
                     continuity=all(r["continuity_pass"] for r in legs))
        row = dict(block=block, legs=legs, gates=gates,
                   cell_ratio=legs[0]["cells"] / fine["aktiva_voxlar"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    gates = dict(repeat=all(r["gates"]["repeat"] for r in rows),
                 bracket=any(all(r["gates"].values()) for r in rows))
    report = dict(status="VERIFIED-FRESH" if all(gates.values()) else "OWN-GATE-FAIL",
                  scope="CPU bracket block-size sweep; original fine comparator and query set.",
                  fine_cells=fine["aktiva_voxlar"], limits=limits, rows=rows, gates=gates)
    dest = ROOT / "reports"
    dest.mkdir(exist_ok=True)
    (dest / "block_size.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(dest / "block_size_arrays.npz", **arrays)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
