"""Stress the frozen native RT winding candidate at rotations and translations."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
from scipy import ndimage
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline

ROOT = Path(__file__).resolve().parents[2]


def guard():
    result = subprocess.run(['journalctl', '-k', '-b', '--no-pager'], capture_output=True, text=True)
    if result.returncode or 'not seeing messages' in result.stderr:
        raise RuntimeError('current-boot fault log unavailable')
    if re.search(r'NVRM:.*Xid', result.stdout):
        raise RuntimeError('current-boot Xid: stop GPU work')


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    binary, program = os.environ['RT_WINDING_BINARY'], os.environ['RT_WINDING_PTX']
    rotated = trimesh.creation.box(extents=[4, 3, 2])
    rotated.apply_transform(trimesh.transformations.rotation_matrix(0.37, [1, 2, 3]))
    rotated.apply_translation([0.13, -0.21, 0.37])
    cases = {}
    for name, shift in [('rotated', 0.), ('translated_100', 100.), ('translated_10000', 10000.), ('translated_1000000', 1000000.)]:
        mesh = rotated.copy()
        mesh.apply_translation([shift, shift, shift])
        cases[name] = mesh
    thin = trimesh.creation.box(extents=[4, 3, 0.3])
    thin.apply_transform(trimesh.transformations.rotation_matrix(0.37, [1, 2, 3]))
    cases['thin_rotated'] = thin
    rows, arrays = [], {}
    guard()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, mesh in cases.items():
            pitch, origin = 0.25, np.asarray(mesh.bounds).min(axis=0)-1.0
            gmin, shape, _, solid, distance = baseline.surface_raster_and_flood(mesh.vertices, mesh.faces, pitch, origin)
            indices = np.indices(shape).reshape(3, -1).T
            points = origin + gmin*pitch + indices*pitch
            points[:, 0] += pitch*baseline.JITTER_FRAC*baseline._GYLLENE[0]
            points[:, 1] += pitch*baseline.JITTER_FRAC*baseline._GYLLENE[1]
            corners = np.asarray(mesh.vertices[mesh.faces], dtype='<f4')
            payload = np.array([len(mesh.faces), len(points)], dtype='<u4').tobytes()+corners.tobytes()+np.asarray(points, dtype='<f4').tobytes()
            infile = tmp/(name+'.bin'); infile.write_bytes(payload)
            row = dict(case=name, voxels=int(solid.size), legs=[])
            for leg in range(2):
                guard()
                output = tmp/f'{name}_{leg}.bin'
                result = subprocess.run([binary, program, str(infile), str(output)], capture_output=True, timeout=30)
                guard()
                item = dict(exit=result.returncode, stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
                if result.returncode == 0:
                    counts = np.fromfile(output, dtype='<i4').reshape(shape)
                    mask = counts != 0
                    d_out = ndimage.distance_transform_edt(~mask, sampling=(pitch,)*3).astype(np.float32)
                    d_in = ndimage.distance_transform_edt(mask, sampling=(pitch,)*3).astype(np.float32)
                    sd = (d_out-d_in).astype(np.float32)
                    sd = (sd-np.sign(sd)*np.float32(0.5*pitch)).astype(np.float32)
                    item.update(winding_sha256=digest(counts), distance_sha256=digest(sd), baseline_occupied=int(solid.sum()), candidate_occupied=int(mask.sum()),
                                occupancy_mismatches=int(np.count_nonzero(mask != solid)),
                                distance_mismatches=int(np.count_nonzero(sd != distance)),
                                max_distance_difference=float(np.max(np.abs(sd-distance))))
                    arrays[f'{name}_counts_{leg}'] = counts
                    arrays[f'{name}_distance_{leg}'] = sd
                row['legs'].append(item)
                if result.returncode:
                    # A runtime failure stops all further context creation in this probe.
                    rows.append(row)
                    failure = dict(rows=rows, runtime_success=False, accepted_pairs=sum(len(r['legs'])==2 for r in rows), timing_measured=False, stderr=result.stderr.decode(errors='replace'))
                    (ROOT/'artifacts'/'field_rt_winding_precision_v1_failure.json').write_text(json.dumps(failure, indent=2)+'\n')
                    raise RuntimeError(f'native runtime exit{result.returncode}; no retry')
            rows.append(row)
    gates = dict(runtime_success=all(q['exit']==0 for r in rows for q in r['legs']),
                 full_winding_repeat=all(r['legs'][0]['winding_sha256']==r['legs'][1]['winding_sha256'] for r in rows),
                 exact_occupancy=all(q['occupancy_mismatches']==0 for r in rows for q in r['legs']),
                 exact_distance=all(q['distance_mismatches']==0 for r in rows for q in r['legs']),
                 no_observed_current_boot_fault=True)
    report = dict(rows=rows, gates=gates, timing_measured=False,
                  binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                  ptx_sha256=hashlib.sha256(Path(program).read_bytes()).hexdigest())
    folder=ROOT/'artifacts'; folder.mkdir(exist_ok=True)
    (folder/'field_rt_winding_precision_v1.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(folder/'field_rt_winding_precision_v1_arrays.npz', **arrays)
    print(json.dumps(report), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
