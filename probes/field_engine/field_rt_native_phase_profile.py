"""Native compact RT phase clocks with full-field exactness controls."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import ctypes as ct
import json
import os
from pathlib import Path
import time
import numpy as np
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline
from mesh_field_persistent_edt_v1 import PreparedMeshField

ROOT=Path(__file__).resolve().parents[2]


class TimedCall:
    def __init__(self,target,method,label,times):
        self.target=target;self.method=method;self.label=label;self.times=times
    def __getattr__(self,name):
        value=getattr(self.target,name)
        if name!=self.method:return value
        def call(*args,**kwargs):
            start=time.perf_counter_ns();result=value(*args,**kwargs)
            self.times[self.label]=(time.perf_counter_ns()-start)*1e-6
            return result
        return call


def main():
    mesh=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);mesh.merge_vertices()
    vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces)
    kwargs=dict(winding_library=os.environ['RT_COLUMNS_LIBRARY'],ptx=os.environ['RT_COLUMNS_PTX'],
                edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'],column_library=os.environ['COLUMN_MASK_LIBRARY'],
                pair_library=os.environ['NATIVE_EDT_PAIR_LIBRARY'])
    rows,arrays=[],{}
    for leg in range(2):
        for pitch in ((2.,1.,.5) if leg==0 else (.5,1.,2.)):
            origin=vertices.min(0)-3*pitch
            expected=baseline.surface_raster_and_flood(vertices,faces,pitch,origin)
            with PreparedMeshField(vertices,faces,pitch,origin,**kwargs) as field:
                field.evaluate()
                parts={}
                native_call=field._rt._lib.columns_query_compact
                phase_reader=field._rt._lib.columns_query_phases
                phase_reader.argtypes=[ct.c_void_p];phase_reader.restype=None
                def timed_native(*args):
                    start=time.perf_counter_ns();value=native_call(*args)
                    parts['native_rt']=(time.perf_counter_ns()-start)*1e-6
                    phases=np.empty(5,dtype=np.float64);phase_reader(phases.ctypes.data)
                    for name,elapsed in zip(('rays','upload','trace','counts','readback'),phases):parts['native_'+name]=float(elapsed)
                    return value
                field._rt._lib.columns_query_compact=timed_native
                field._rt=TimedCall(field._rt,'query','rt_query',parts)
                field._mask=TimedCall(field._mask,'evaluate','mask',parts)
                field._edt=TimedCall(field._edt,'signed_distance','edt',parts)
                for repeat in range(4):
                    start=time.perf_counter_ns();actual=field.evaluate();elapsed=(time.perf_counter_ns()-start)*1e-6
                    exact=all(np.asarray(a).dtype==np.asarray(b).dtype and np.asarray(a).shape==np.asarray(b).shape and np.asarray(a).tobytes()==np.asarray(b).tobytes() for a,b in zip(actual,expected))
                    hashes=[hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for a in actual]
                    rows.append(dict(leg=leg,pitch=pitch,repeat=repeat,total_ms=elapsed,parts_ms=dict(parts),exact=exact,hashes=hashes))
                    for i,a in enumerate(actual):arrays[f'{leg}_{pitch}_{repeat}_{i}']=np.asarray(a)
    gates=dict(coverage=len(rows)==24,exact=all(r['exact'] for r in rows),
               repeat=all(len({json.dumps(r['hashes']) for r in rows if r['pitch']==p})==1 for p in (2.,1.,.5)),
               native_accounting=all(0<=sum(r['parts_ms']['native_'+k] for k in ('rays','upload','trace','counts','readback'))<=r['parts_ms']['native_rt'] for r in rows),
               accounting=all(0<=sum(v for k,v in r['parts_ms'].items() if not k.startswith('native_'))<=r['total_ms'] for r in rows))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,
                scope='Direct prepared stage with one untimed warmup, two reversed size orders and four timed evaluations each. Component methods are already synchronous; no extra GPU fence. All native_* values are nested inside rt_query; rays/upload/trace/counts/readback subdivide native_rt. No IPC and no 2 ms acceptance claim.')
    (ROOT/'reports/field_rt_native_phase_profile.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_rt_native_phase_profile_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
