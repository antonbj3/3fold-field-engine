"""Rowwise ordering against the frozen global sort and full mesh-stage fields."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import trimesh
from rt_columns_rowwise_handle_v1 import sort_rows
from mesh_field_rowwise_v1 import PreparedMeshField
from mesh_field_persistent_edt_v1 import PreparedMeshField as ReferenceField
from field_mesh_native_mask_stage_probe import cases,expected_open,baseline
from field_linear_edt_probe import timed

ROOT=Path(__file__).resolve().parents[2]


def same(a,b):return all(np.asarray(x).dtype==np.asarray(y).dtype and np.asarray(x).shape==np.asarray(y).shape and np.asarray(x).tobytes()==np.asarray(y).tobytes() for x,y in zip(a,b))


def sorting_controls():
    rng=np.random.default_rng(967);result=[]
    for rows,width in ((1,0),(17,2),(31,7),(129,64),(4099,4)):
        counts=rng.integers(0,width+1,rows,dtype=np.int32)
        z=rng.choice(np.array([-2.,-0.,0.,1.,3.]),size=(rows,width));s=rng.choice(np.array([-1,1],np.int32),size=z.shape)
        valid=np.arange(width)[None,:]<counts[:,None]
        z[~valid]=np.nan
        c=np.broadcast_to(np.arange(rows)[:,None],valid.shape)[valid];depths=z[valid];signs=s[valid]
        order=np.lexsort((signs,depths,c));expected=(c[order],depths[order],signs[order])
        result.append(same(sort_rows(counts,z,s),expected))
    for counts,z,s in ((np.array([3]),np.zeros((1,2)),np.ones((1,2))), (np.array([-1]),np.zeros((1,2)),np.ones((1,2)))):
        try:sort_rows(counts,z,s)
        except ValueError:result.append(True)
        else:result.append(False)
    return result


def main():
    controls=sorting_controls();rows=[];arrays={}
    kwargs=dict(winding_library=os.environ['RT_COLUMNS_LIBRARY'],ptx=os.environ['RT_COLUMNS_PTX'],
                edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'],column_library=os.environ['COLUMN_MASK_LIBRARY'],pair_library=os.environ['NATIVE_EDT_PAIR_LIBRARY'])
    for name,(mesh,pitch,origin,closed) in cases().items():
        vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces)
        expected=baseline.surface_raster_and_flood(vertices,faces,pitch,origin) if closed else expected_open(vertices,faces,pitch,origin)
        with ReferenceField(vertices,faces,pitch,origin,**kwargs) as old,PreparedMeshField(vertices,faces,pitch,origin,**kwargs) as new:
            old.evaluate();new.evaluate()
            for leg in range(2):
                actual,times={},{}
                for label in (('global','rowwise') if leg==0 else ('rowwise','global')):
                    actual[label],times[label]=timed((old if label=='global' else new).evaluate)
                hashes=[hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for a in actual['rowwise']]
                rows.append(dict(case=name,leg=leg,pitch=pitch,closed=closed,times_ms=times,exact=same(actual['global'],actual['rowwise']) and same(actual['rowwise'],expected),hashes=hashes))
                for i,a in enumerate(actual['rowwise']):arrays[f'{name}_{leg}_{i}']=np.asarray(a)
            if closed:
                hit=new._rt.query(new._xy)
                # Reject deliberately decreasing columns or within-column depths.
                for c,z,s in ((np.array([1,0]),np.array([0.,0.]),np.array([1,-1])),(np.array([0,0]),np.array([1.,0.]),np.array([1,-1]))):
                    try:new._mask.evaluate(c,z,s,pitch,new._anchor,new.shape)
                    except ValueError:controls.append(True)
                    else:controls.append(False)
    # Plate L uses pitch 0.5 in the frozen seven-fixture generator.
    large=[r for r in rows if r['case']=='plate_0.5']
    gates=dict(coverage=len(rows)==14,exact=all(r['exact'] for r in rows),
               repeat=all(len({json.dumps(r['hashes']) for r in rows if r['case']==name})==1 for name in {r['case'] for r in rows}),
               controls=len(controls)==17 and all(controls),L_speedup_1_5=len(large)==2 and all(r['times_ms']['global']/r['times_ms']['rowwise']>=1.5 for r in large))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,controls=controls,
                scope='Frozen RT compact library and persistent paired EDT on both paths. Per-row lexicographic sorting plus checked ordered mask eliminates global sorts; numerical operands and GPU kernels unchanged. Seven fixtures including translated/rotated/open/reversed cases, two reversed timing orders, preparation/warmup excluded.')
    (ROOT/'reports/field_rowwise_stage.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_rowwise_stage_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
