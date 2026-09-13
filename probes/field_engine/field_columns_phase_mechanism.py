"""Measured prepared-stage component costs before another optimization."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import trimesh
from scipy import ndimage
from mesh_field_columns_v1 import PreparedMeshField
from rt_columns_handle_v1 import occupancy
from field_mesh_columns_stage_probe import digest


def main():
    root=Path(__file__).resolve().parents[2]
    a=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    b=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    idle=a.returncode==b.returncode==0 and a.stdout.strip()=='0' and not b.stdout.strip()
    if not idle:
        (root/'reports/columns_phase.json').write_text(json.dumps(dict(samples=0,idle=False,util=a.stdout,contexts=b.stdout),indent=2)+'\n');return 2
    mesh=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);mesh.merge_vertices();pitch=.5
    rows=[];arrays={}
    def timed(fn):
        start=time.perf_counter_ns();value=fn();return value,(time.perf_counter_ns()-start)*1e-6
    with PreparedMeshField(mesh.vertices,mesh.faces,pitch,mesh.vertices.min(0)-3*pitch,winding_library=os.environ['RT_COLUMNS_LIBRARY'],ptx=os.environ['RT_COLUMNS_PTX'],edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY']) as field:
        reference=field.evaluate()
        for leg in range(2):
            hits,query_ms=timed(lambda:field._rt.query(field._xy))
            solid,keys_ms=timed(lambda:occupancy(*hits,pitch,field._anchor,field.shape))
            sd,edt_ms=timed(lambda:field._edt.signed_distance(solid,pitch))
            surface,surface_ms=timed(lambda:solid & ~ndimage.binary_erosion(solid,ndimage.generate_binary_structure(3,1)))
            outputs=(*hits,solid,sd,surface);hashes=[digest(x) for x in outputs]
            diffs=[int(np.count_nonzero(a!=b)) for a,b in zip((solid,sd,surface),(reference[3],reference[4],reference[2]))]
            rows.append(dict(leg=leg,query_ms=query_ms,keys_ms=keys_ms,edt_ms=edt_ms,surface_ms=surface_ms,hashes=hashes,differences=diffs))
            for name,x in zip(('columns','z','sign','solid','distance','surface'),outputs):arrays[f'{leg}_{name}']=x
        capacity=field._rt.hit_capacity;columns=len(field._xy)
    report=dict(rows=rows,gates=dict(exact_composed=all(r['differences']==[0,0,0] for r in rows),full_repeat=rows[0]['hashes']==rows[1]['hashes'],idle=idle),columns=columns,hit_capacity=capacity,depth_sign_readback_bytes=columns*capacity*12,scope='Synthetic full component calls; no isolated kernel or speed acceptance claim.')
    (root/'reports/columns_phase.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(root/'reports/columns_phase_arrays.npz',**arrays);print(json.dumps(report));return 0 if all(report['gates'].values()) else 2

if __name__=='__main__':raise SystemExit(main())
