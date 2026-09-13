"""CPU rank reduction on saved exact synthetic RT hits, no CUDA initialization."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
import os
import time
from pathlib import Path
import numpy as np
import trimesh
import faltkarna_v1_mesh_to_sdf as baseline
from rt_columns_handle_v1 import occupancy as reference
from column_mask_native_v1 import NativeColumnMask
from scipy import ndimage
from field_mesh_columns_stage_probe import digest


def main():
    root=Path(__file__).resolve().parents[2];saved=np.load(root/'reports/rt_columns_arrays.npz')
    plate=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);plate.merge_vertices();cases=[(f'plate_{p}',plate,p) for p in (2.,1.,.5)]
    for shift in (0.,1e6):
        box=trimesh.creation.box(extents=[4,3,2]);box.apply_transform(trimesh.transformations.rotation_matrix(.31,[1,2,3]));box.apply_translation([shift+.13,shift-.17,shift+.23]);cases.append((f'box_{shift}',box,.5))
    rows=[];arrays={};native=NativeColumnMask(os.environ["COLUMN_MASK_LIBRARY"])
    for name,mesh,pitch in cases:
        lo=mesh.vertices.min(0)-2*pitch;gmin,shape=baseline._fonster(mesh.vertices,pitch,lo);origin=lo+gmin*pitch
        hits=[saved[name+'_0_'+k] for k in ('columns','z','sign')]
        for leg in range(2):
            start=time.perf_counter_ns();ref=reference(*hits,pitch,origin,shape);ref_ms=(time.perf_counter_ns()-start)*1e-6
            start=time.perf_counter_ns();actual,surface=native.evaluate(*hits,pitch,origin,shape);ranked_ms=(time.perf_counter_ns()-start)*1e-6
            rows.append(dict(case=name,leg=leg,reference_ms=ref_ms,ranked_ms=ranked_ms,differences=int(np.count_nonzero(actual!=ref)),surface_differences=int(np.count_nonzero(surface!=(ref & ~ndimage.binary_erosion(ref,ndimage.generate_binary_structure(3,1))))),hash=digest(actual),surface_hash=digest(surface)))
            arrays[f'{name}_{leg}']=actual;arrays[f'{name}_{leg}_surface']=surface
    empty,empty_surface=native.evaluate(np.array([],np.int64),np.array([],np.float64),np.array([],np.int32),1.,np.zeros(3),(3,3,3))
    gates=dict(exact=all(r['differences']==0 for r in rows),exact_surface=all(r['surface_differences']==0 for r in rows),full_repeat=all(rows[i]['hash']==rows[i+1]['hash'] and rows[i]['surface_hash']==rows[i+1]['surface_hash'] for i in range(0,len(rows),2)),empty=not empty.any() and not empty_surface.any())
    report=dict(rows=rows,gates=gates,scope='Saved synthetic RT hits, native mask and surface; no full-stage speed claim.')
    (root/'reports/column_mask_native.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(root/'reports/column_mask_native_arrays.npz',**arrays);print(json.dumps(report));return 0 if all(gates.values()) else 2

if __name__=='__main__':raise SystemExit(main())
