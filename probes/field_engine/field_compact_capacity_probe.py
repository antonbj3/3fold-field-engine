"""Compact readback at full capacity, overflow and lifecycle controls."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
import os
import threading
from pathlib import Path
import numpy as np
import trimesh
from rt_columns_handle_v1 import ColumnsHandle as Full
from rt_columns_compact_handle_v1 import ColumnsHandle as Compact
from field_mesh_columns_stage_probe import digest


def main():
    root=Path(__file__).resolve().parents[2];box=trimesh.creation.box(extents=[4,4,4]);tri=np.asarray(box.vertices[box.faces]);xy=np.array([[.123,.234]])
    rows=[];arrays={};overflow=retry=foreign=0
    for leg in range(2):
        for count in (32,33):
            corners=np.concatenate([tri+[0,0,10*i] for i in range(count)])
            for name,cls,library in (('full',Full,os.environ['RT_FULL_LIBRARY']),('compact',Compact,os.environ['RT_COLUMNS_LIBRARY'])):
                with cls(library,os.environ['RT_COLUMNS_PTX'],corners,np.zeros(3),1) as h:
                    errors=[]
                    def wrong_thread():
                        try:h.query(xy)
                        except RuntimeError:errors.append(True)
                    t=threading.Thread(target=wrong_thread);t.start();t.join();foreign+=len(errors)
                    if count==32:
                        out=h.query(xy);rows.append(dict(leg=leg,method=name,hits=len(out[0]),hashes=[digest(x) for x in out]))
                        for k,x in zip(('columns','z','sign'),out):arrays[f'{leg}_{name}_{k}']=x
                    else:
                        try:h.query(xy)
                        except RuntimeError as e:overflow+=int('status 4' in str(e))
                        try:h.query(xy)
                        except RuntimeError as e:retry+=int('no retry' in str(e))
    gates=dict(full_width_exact=all(r['hits']==64 and r['hashes']==rows[0]['hashes'] for r in rows),full_repeat=rows[0]['hashes']==rows[2]['hashes'] and rows[1]['hashes']==rows[3]['hashes'],overflow_and_latch=overflow==retry==4,thread_refusal=foreign==8)
    report=dict(rows=rows,gates=gates,overflow_rejections=overflow,retry_refusals=retry,thread_refusals=foreign,scope='Synthetic column resource and lifecycle control, no timing claim.')
    (root/'reports/compact_capacity.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(root/'reports/compact_capacity_arrays.npz',**arrays);print(json.dumps(report));return 0 if all(gates.values()) else 2

if __name__=='__main__':raise SystemExit(main())
