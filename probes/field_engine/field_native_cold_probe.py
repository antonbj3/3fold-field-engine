"""Two independent cold native stage processes with complete output equality."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def child(leg):
    import trimesh
    import faltkarna_v1_mesh_to_sdf as baseline
    from mesh_field_native_mask_v1 import surface_raster_and_flood_rt
    from field_mesh_columns_stage_probe import digest
    mesh=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);mesh.merge_vertices();v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces);pitch=.5;lo=v.min(0)-3*pitch
    start=time.perf_counter_ns();ref=baseline.surface_raster_and_flood(v,f,pitch,lo);cpu_ms=(time.perf_counter_ns()-start)*1e-6
    a=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    b=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    idle=a.returncode==b.returncode==0 and a.stdout.strip()=='0' and not b.stdout.strip()
    if not idle:
        report=dict(leg=leg,idle=False,samples=0,util=a.stdout,contexts=b.stdout)
        (ROOT/f'reports/native_cold_{leg}.json').write_text(json.dumps(report,indent=2)+'\n');return 2
    start=time.perf_counter_ns()
    actual=surface_raster_and_flood_rt(v,f,pitch,lo,winding_library=os.environ['RT_COLUMNS_LIBRARY'],ptx=os.environ['RT_COLUMNS_PTX'],edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'],column_library=os.environ['COLUMN_MASK_LIBRARY'])
    native_ms=(time.perf_counter_ns()-start)*1e-6
    arrays={};diffs={};hashes={}
    for i,k in ((0,'gmin'),(2,'surface'),(3,'solid'),(4,'distance')):
        arrays[k]=actual[i];hashes[k]=digest(actual[i]);diffs[k]=int(np.count_nonzero(actual[i]!=ref[i]))
    np.savez_compressed(ROOT/f'reports/native_cold_{leg}_arrays.npz',**arrays)
    report=dict(leg=leg,idle=idle,cpu_ms=cpu_ms,native_ms=native_ms,differences=diffs,hashes=hashes,shape_equal=actual[1]==ref[1])
    (ROOT/f'reports/native_cold_{leg}.json').write_text(json.dumps(report,indent=2)+'\n');return 0


def main():
    rows=[]
    for leg in range(2):
        start=time.perf_counter_ns();result=subprocess.run([sys.executable,__file__,'--child',str(leg)],capture_output=True,text=True,timeout=90);wall=(time.perf_counter_ns()-start)*1e-6
        p=ROOT/f'reports/native_cold_{leg}.json';row=json.loads(p.read_text()) if p.exists() else dict(samples=0,idle=False)
        row.update(exit=result.returncode,stderr=result.stderr,process_ms=wall);rows.append(row)
        if result.returncode:break
    valid=len(rows)==2 and all(r['exit']==0 for r in rows)
    gates=dict(exact=valid and all(r['shape_equal'] and not any(r['differences'].values()) for r in rows),full_repeat=valid and rows[0]['hashes']==rows[1]['hashes'],idle=valid and all(r['idle'] for r in rows),cold_faster=valid and all(r['native_ms']<r['cpu_ms'] for r in rows))
    report=dict(rows=rows,gates=gates,scope='Two fresh child processes; first native call includes setup and teardown. Process wall includes imports/reference/checks/report writes.')
    (ROOT/'reports/native_cold.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if all(gates.values()) else 2

if __name__=='__main__':raise SystemExit(child(int(sys.argv[2])) if '--child' in sys.argv else main())
