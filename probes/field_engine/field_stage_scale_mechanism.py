"""Measure the frozen CPU mesh stage on the original declared plate scales."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--mesh',required=True);args=p.parse_args()
    path=Path(args.mesh);source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    mesh=trimesh.load(path,process=False);mesh.merge_vertices()
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces);rows=[];arrays={}
    for pitch in (2.,1.,.5):
        lo=v.min(axis=0)-3*pitch;legs=[]
        for leg in range(2):
            start=time.perf_counter();result=baseline.surface_raster_and_flood(v,f,pitch,lo);elapsed=time.perf_counter()-start
            gmin,shape,surface,solid,sd=result
            hashes={k:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for k,a in dict(gmin=gmin,surface=surface,solid=solid,distance=sd).items()}
            legs.append(dict(milliseconds=elapsed*1000,hashes=hashes))
            for k,a in dict(gmin=gmin,surface=surface,solid=solid,distance=sd).items():arrays[f'{pitch}_{leg}_{k}']=a
        rows.append(dict(pitch=pitch,shape=list(shape),cells=int(solid.size),legacy_native_supported=max(shape)<=128 and solid.size<=262144,legs=legs))
    gates=dict(full_repeat=all(r['legs'][0]['hashes']==r['legs'][1]['hashes'] for r in rows),
               source_unchanged=source_hash==hashlib.sha256(path.read_bytes()).hexdigest(),source_closed=bool(mesh.is_watertight))
    report=dict(rows=rows,gates=gates,source_sha256=source_hash,vertices=len(v),triangles=len(f),scope='Frozen CPU stage only, no device classification or accelerated comparison.')
    folder=ROOT/'reports';folder.mkdir(exist_ok=True)
    (folder/'stage_scale_mechanism.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(folder/'stage_scale_mechanism_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 2


if __name__=='__main__':raise SystemExit(main())
