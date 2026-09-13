"""Paired host-wall timings with full exact outputs and explicit cold/setup cost."""

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
import faltkarna_v1_mesh_to_sdf as baseline
from mesh_field_native_mask_v1 import PreparedMeshField, surface_raster_and_flood_rt
from mesh_field_compact_v1 import PreparedMeshField as FullPrepared
from field_mesh_columns_stage_probe import digest


def main():
    root=Path(__file__).resolve().parents[2]
    util=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    contexts=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    idle=util.returncode==0 and contexts.returncode==0 and util.stdout.strip()=='0' and not contexts.stdout.strip()
    if not idle:
        report=dict(samples=0,idle=False,utilization=util.stdout,contexts=contexts.stdout)
        (root/'reports/native_mask_timing.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 2
    mesh=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);mesh.merge_vertices();v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces)
    kwargs=dict(winding_library=os.environ['RT_COLUMNS_LIBRARY'],ptx=os.environ['RT_COLUMNS_PTX'],edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'],column_library=os.environ['COLUMN_MASK_LIBRARY'])
    rows=[];arrays={}
    def timed(fn):
        start=time.perf_counter_ns();value=fn();duration=(time.perf_counter_ns()-start)*1e-6;return value,duration
    cold,cold_ms=timed(lambda:surface_raster_and_flood_rt(v,f,.5,v.min(0)-1.5,**kwargs))
    for i,k in ((0,'gmin'),(2,'surface'),(3,'solid'),(4,'distance')):arrays['cold_'+k]=cold[i]
    for pitch in (2.,1.,.5):
        lo=v.min(0)-3*pitch
        field,setup_ms=timed(lambda:PreparedMeshField(v,f,pitch,lo,**kwargs))
        full_kwargs=dict(kwargs);full_kwargs.pop('column_library')
        full_field=FullPrepared(v,f,pitch,lo,**full_kwargs)
        try:
            for leg in range(2):
                ref,cpu_ms=timed(lambda:baseline.surface_raster_and_flood(v,f,pitch,lo))
                one,one_ms=timed(lambda:surface_raster_and_flood_rt(v,f,pitch,lo,**kwargs))
                full,full_ms=timed(full_field.evaluate)
                prepared,prepared_ms=timed(field.evaluate)
                hashes={};differences={}
                for method,result in (('cpu',ref),('one',one),('prepared',prepared),('full',full)):
                    hashes[method]={};differences[method]={}
                    for i,k in ((0,'gmin'),(2,'surface'),(3,'solid'),(4,'distance')):
                        hashes[method][k]=digest(result[i]);differences[method][k]=int(np.count_nonzero(result[i]!=ref[i]))
                        arrays[f'{pitch}_{leg}_{method}_{k}']=result[i]
                rows.append(dict(pitch=pitch,leg=leg,shape=list(ref[1]),cpu_ms=cpu_ms,one_shot_ms=one_ms,prepared_ms=prepared_ms,setup_ms=setup_ms,full_prepared_ms=full_ms,hashes=hashes,differences=differences))
        finally:
            field.close();full_field.close()
    gates=dict(exact=all(not any(d.values()) for r in rows for d in r['differences'].values()) and all(np.array_equal(cold[i],arrays['0.5_0_cpu_'+k]) for i,k in ((0,'gmin'),(2,'surface'),(3,'solid'),(4,'distance'))),full_repeat=all(rows[i]['hashes']==rows[i+1]['hashes'] for i in range(0,len(rows),2)),idle=idle,prepared_L_faster=all(r['prepared_ms']<r['cpu_ms'] for r in rows if r['pitch']==.5),one_shot_L_faster=all(r['one_shot_ms']<r['cpu_ms'] for r in rows if r['pitch']==.5))
    gates['native_L_faster_than_compact']=all(r['prepared_ms']<r['full_prepared_ms'] for r in rows if r['pitch']==.5)
    report=dict(rows=rows,cold_one_shot_L_ms=cold_ms,gates=gates,scope='Synthetic cloud stage, synchronized wall time; cold setup included separately.')
    (root/'reports/native_mask_timing.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(root/'reports/native_mask_timing_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 2

if __name__=='__main__':raise SystemExit(main())
