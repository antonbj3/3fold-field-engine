"""Bounded EDT templates versus fixed-512 scratch in the full native field."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import hashlib
import json
import os
from pathlib import Path
import threading
import numpy as np
import trimesh
from mesh_field_fused_v1 import PreparedMeshField
from mesh_field_fused_v1 import PreparedMeshField as ReferenceField
from field_mesh_native_mask_stage_probe import cases,expected_open,baseline
from field_linear_edt_probe import timed

ROOT=Path(__file__).resolve().parents[2]


def same(a,b):return all(np.asarray(x).dtype==np.asarray(y).dtype and np.asarray(x).shape==np.asarray(y).shape and np.asarray(x).tobytes()==np.asarray(y).tobytes() for x,y in zip(a,b))


def main():
    common=dict(winding_library=os.environ['RT_COLUMNS_LIBRARY'],ptx=os.environ['RT_COLUMNS_PTX'],
                edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'],column_library=os.environ['COLUMN_MASK_LIBRARY'])
    candidate=dict(common,fused_library=os.environ['MESH_FUSED_TEMPLATED_LIBRARY'])
    reference=dict(common,fused_library=os.environ['MESH_FUSED_LIBRARY'])
    rows,arrays,controls=[],{},[]
    for name,(mesh,pitch,origin,closed) in cases().items():
        vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces)
        expected=baseline.surface_raster_and_flood(vertices,faces,pitch,origin) if closed else expected_open(vertices,faces,pitch,origin)
        with ReferenceField(vertices,faces,pitch,origin,**reference) as old,PreparedMeshField(vertices,faces,pitch,origin,**candidate) as new:
            old.evaluate();new.evaluate()
            for leg in range(2):
                actual,times={},{}
                for mode in (('fixed512','templated') if leg==0 else ('templated','fixed512')):
                    actual[mode],times[mode]=timed((old if mode=='fixed512' else new).evaluate)
                rows.append(dict(case=name,leg=leg,times_ms=times,exact=same(actual['fixed512'],actual['templated']) and same(actual['templated'],expected),
                                 hashes=[hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for a in actual['templated']]))
                for i,a in enumerate(actual['templated']):arrays[f'{name}_{leg}_{i}']=np.asarray(a)
            def foreign():
                try:new.evaluate()
                except RuntimeError:controls.append(True)
                else:controls.append(False)
            t=threading.Thread(target=foreign);t.start();t.join()
        try:new.evaluate()
        except RuntimeError:controls.append(True)
        else:controls.append(False)
    for leg in range(2):
        for count in (32,33):
            mesh=trimesh.util.concatenate([trimesh.creation.box(extents=[4,4,4]).apply_translation([0,0,10*i]) for i in range(count)])
            vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces);origin=vertices.min(0)-3
            with PreparedMeshField(vertices,faces,1.,origin,**candidate) as field:
                if count==32:
                    expected=baseline.surface_raster_and_flood(vertices,faces,1.,origin)
                    actual=field.evaluate();controls.append(same(actual,expected))
                    for i,a in enumerate(actual):arrays[f'capacity_{leg}_{i}']=np.asarray(a)
                else:
                    for _ in range(2):
                        try:field.evaluate()
                        except RuntimeError:controls.append(True)
                        else:controls.append(False)
    gates=dict(coverage=len(rows)==14,exact=all(r['exact'] for r in rows),
               repeat=all(len({json.dumps(r['hashes']) for r in rows if r['case']==name})==1 for name in {r['case'] for r in rows}),
               capacity_lifecycle=len(controls)==20 and all(controls),
               L_speedup_1_25=all(r['times_ms']['fixed512']/r['times_ms']['templated']>=1.25 for r in rows if r['case']=='plate_0.5'))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,controls=controls,
                source_sha256=hashlib.sha256((ROOT/'src/field_engine/mesh_fused_templated_v1/field.cu').read_bytes()).hexdigest(),
                scope='Prepared immutable query grid. Every evaluation retraces geometry and recomputes mask/surface/signed EDT. Both closed paths keep hits/mask on device; only EDT array bounds differ; open path preserves GWN/stateless EDT. Preparation and warmup excluded, two reversed orders. No full-service 2 ms claim.')
    (ROOT/'reports/field_mesh_templated.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_mesh_templated_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
