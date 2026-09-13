"""Quantify full-array external winding repeat differences without hiding them."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
import numpy as np
from field_open_mesh_mechanism import leg,ROOT


def main():
    a,aa=leg();b,bb=leg();rows=[];saved={}
    for key in aa:
        x,y=aa[key],bb[key]
        rows.append(dict(array=key,differing_elements=int(np.count_nonzero(x!=y)),
                         max_absolute_difference=float(np.max(np.abs(x.astype(float)-y.astype(float)))),
                         bytes_identical=x.tobytes()==y.tobytes()))
        saved[key+'_0']=x;saved[key+'_1']=y
    report=dict(rows=rows,gates=dict(full_repeat=all(r['bytes_identical'] for r in rows)),reference_tables=[a,b])
    folder=ROOT/'reports';folder.mkdir(exist_ok=True)
    (folder/'open_mesh_repeat.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(folder/'open_mesh_repeat_arrays.npz',**saved)
    print(json.dumps(report));return 0 if all(report['gates'].values()) else 2


if __name__=='__main__':raise SystemExit(main())
