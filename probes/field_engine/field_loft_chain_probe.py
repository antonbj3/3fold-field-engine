"""Regenerate the optimizer-to-loft chain twice in fresh CPU processes."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = ROOT / "src/field_engine"
from field_paths import field_path as _field_path
FILES = ("field_recipe_sections_probe.npy", "field_recipe_sections_probe.json",
         "field_to_recipe_loft_v1_recipe.json", "field_to_recipe_loft_v1.json")


def main():
    destination = ROOT / "reports" / "loft_chain"
    destination.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1")
    legs = []
    for leg in range(2):
        folder = destination / str(leg)
        folder.mkdir(exist_ok=True)
        for script in ("field_recipe_sections_probe.py", "field_to_recipe_loft_v1.py"):
            with (folder / (script + ".log")).open("w") as log:
                subprocess.run([sys.executable, str(_field_path(ROOT, script))], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        hashes = {}
        for name in FILES:
            data = (ENGINE / "artifacts" / name).read_bytes()
            (folder / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        sections = json.loads((folder / FILES[1]).read_text())
        loft = json.loads((folder / FILES[3]).read_text())
        legs.append(dict(hashes=hashes, section_gates=sections["gates"],
                         loft_gates=loft["gates"], volume_error=loft["relative_volume_error"],
                         rasterisation_bound=loft["rasterisation_bound"]))
    gates = dict(sections=all(all(r["section_gates"].values()) for r in legs),
                 loft=all(all(r["loft_gates"].values()) for r in legs),
                 independent_chain_repeat=legs[0]["hashes"] == legs[1]["hashes"])
    report = dict(status="VERIFIED-FRESH" if all(gates.values()) else "OWN-GATE-FAIL",
                  scope="Synthetic topology-optimized fixture; ruled section slabs; CPU only.",
                  legs=legs, gates=gates)
    (destination / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
