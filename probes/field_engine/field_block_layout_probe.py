"""Measure block-size storage and sign preservation before adaptive design."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import tempfile

# Configure visibility before importing Warp; all launches explicitly use CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import faltkarna_v1_multires as baseline

ROOT = Path(__file__).resolve().parents[2]


def measure(wp):
    with tempfile.TemporaryDirectory() as tmp:
        vertices, triangles = baseline._bracket_mesh(str(Path(tmp)/"bracket.stl"))
    pitch, _ = baseline.M2S.valj_pitch_for_feature(vertices, triangles, 2.0, "auto")
    origin = vertices.min(axis=0)-12*pitch
    fine = baseline._global_fall(wp, "cpu", vertices, triangles, pitch, origin, "fine")
    dense = fine["sd"]
    truth = dense < 0
    rows = []
    for block in (1, 2, 4, 8, 16):
        field = baseline.M2S.klassificera_och_evaluera_fran_tatt_falt(wp, dense, pitch, block, "cpu")
        solid = baseline.M2S.reconstruct_solid_from_sparse(field, *dense.shape, block)
        rows.append({"block": block, "active_blocks": field["n_active"],
                     "total_blocks": field["n_blocks"],
                     "active_voxels": field["n_active"]*block**3,
                     "ratio_vs_frozen_fine": field["n_active"]*block**3/fine["aktiva_voxlar"],
                     "stored_bytes": sum(field[k].nbytes for k in ("kind", "active_ids", "tiles")),
                     "sign_mismatches": int(np.count_nonzero(solid != truth)),
                     "solid_sha256": hashlib.sha256(solid.tobytes()).hexdigest(),
                     "layout_sha256": hashlib.sha256(b"".join(field[k].tobytes() for k in ("kind", "active_ids", "tiles"))).hexdigest()})
    return {"pitch": float(pitch), "dense_shape": list(dense.shape),
            "dense_bytes": dense.nbytes, "fine_active_voxels": fine["aktiva_voxlar"],
            "dense_sha256": hashlib.sha256(dense.tobytes()).hexdigest(), "rows": rows}


def main():
    import warp as wp
    wp.init()
    legs = [measure(wp), measure(wp)]
    gates = {"bit_identical": legs[0] == legs[1],
             "published_fine_voxels": all(r["fine_active_voxels"] == 718848 for r in legs),
             "sign_preserved": all(q["sign_mismatches"] == 0 for r in legs for q in r["rows"])}
    result = {"legs": legs, "gates": gates, "timing_measured": False}
    destination = ROOT/"artifacts"/"field_block_layout_probe.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
