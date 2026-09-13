"""Exact device-packed RT hit rows versus narrow-row copies and CPU hits."""

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
import faltkarna_v1_mesh_to_sdf as baseline
from rt_columns_compact_handle_v1 import ColumnsHandle
from field_linear_edt_probe import timed

ROOT=Path(__file__).resolve().parents[2]


def same(a,b):
    return all(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes() for x,y in zip(a,b))


def main():
    mesh=trimesh.load(os.environ['FIELD_PLATE_STL'],process=False);mesh.merge_vertices()
    vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces);corners=vertices[faces]
    libraries=dict(strided=os.environ['RT_COLUMNS_LIBRARY'],packed=os.environ['RT_COLUMNS_PACKED_LIBRARY'])
    ptx=os.environ['RT_COLUMNS_PTX'];rows=[];arrays={}
    for leg in range(2):
        for pitch in ((2.,1.,.5) if leg==0 else (.5,1.,2.)):
            origin=vertices.min(0)-3*pitch;gmin,shape=baseline._fonster(vertices,pitch,origin,None,2);anchor=origin+gmin*pitch
            ij=np.indices(shape[:2]).reshape(2,-1).T
            xy=anchor[:2]+(ij+baseline.VOXEL_PROVPUNKT)*pitch
            xy[:,0]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[0];xy[:,1]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[1]
            expected=baseline._kolumntraffar(vertices,faces,pitch,anchor,*shape[:2]);order=np.lexsort((expected[2],expected[1],expected[0]));expected=tuple(x[order] for x in expected)
            # CPU orientations are int8; the frozen native hit ABI is int32.
            expected=(expected[0].astype(np.int64),expected[1].astype(np.float64),expected[2].astype(np.int32))
            handles={};setup={}
            try:
                for name,library in libraries.items():
                    handles[name],setup[name]=timed(lambda:ColumnsHandle(library,ptx,corners,anchor,len(xy)))
                    handles[name].query(xy)
                for repeat in range(4):
                    outputs,times={},{}
                    for name in (('strided','packed') if (leg+repeat)%2==0 else ('packed','strided')):
                        outputs[name],times[name]=timed(lambda:handles[name].query(xy))
                    hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in outputs['packed']]
                    rows.append(dict(leg=leg,pitch=pitch,repeat=repeat,setup_ms=setup,query_ms=times,exact=same(outputs['packed'],outputs['strided']) and same(outputs['packed'],expected),hashes=hashes))
                    for i,a in enumerate(outputs['packed']):arrays[f'{leg}_{pitch}_{repeat}_{i}']=a
            finally:
                for h in handles.values():h.close()
    box=trimesh.creation.box(extents=[4,4,4]);triangle=np.asarray(box.vertices[box.faces]);xy=np.array([[.123,.234]])
    controls=[]
    for leg in range(2):
        for count in (32,33):
            corners=np.concatenate([triangle+[0,0,10*i] for i in range(count)])
            h=ColumnsHandle(libraries['packed'],ptx,corners,np.zeros(3),1)
            try:
                def foreign():
                    try:h.query(xy)
                    except RuntimeError:controls.append(True)
                    else:controls.append(False)
                t=threading.Thread(target=foreign);t.start();t.join()
                for bad in (np.empty((0,2)),np.zeros((2,2)),np.array([[np.nan,0]])):
                    try:h.query(bad)
                    except ValueError:controls.append(True)
                    else:controls.append(False)
                if count==32:
                    with ColumnsHandle(libraries['strided'],ptx,corners,np.zeros(3),1) as reference:
                        controls.append(same(h.query(xy),reference.query(xy)))
                    controls.append(all(len(a)==0 for a in h.query(np.array([[10.,10.]]))))
                else:
                    for _ in range(2):
                        try:h.query(xy)
                        except RuntimeError:controls.append(True)
                        else:controls.append(False)
            finally:h.close()
            try:h.query(xy)
            except RuntimeError:controls.append(True)
            else:controls.append(False)
    gates=dict(coverage=len(rows)==24,exact=all(r['exact'] for r in rows),repeat=all(len({json.dumps(r['hashes']) for r in rows if r['pitch']==p})==1 for p in (2.,1.,.5)),
               capacity_lifecycle=len(controls)==28 and all(controls),L_speedup_2=all(r['query_ms']['strided']/r['query_ms']['packed']>=2 for r in rows if r['pitch']==.5))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,controls=controls,
                source_sha256=hashlib.sha256((ROOT/'src/field_engine/rt_columns_packed_v1/api.cu').read_bytes()).hexdigest(),
                scope='Frozen RT geometry and trace programs, same compact host ABI. Persistent device scratch plus contiguous readbacks replaces strided cudaMemcpy2D. Warm query includes Python validation/filter/sort. Setup separate, one warmup; no complete field/service claim.')
    (ROOT/'reports/field_rt_packed.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'reports/field_rt_packed_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
