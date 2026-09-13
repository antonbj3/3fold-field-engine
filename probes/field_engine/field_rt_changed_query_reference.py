"""Measure changed ray-grid queries before a mutable persistent RT interface."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import json
from pathlib import Path
import numpy as np
from field_rt_mesh_probe import fixtures, digest
from field_rt_lifecycle_probe import edt
import faltkarna_v1_mesh_to_sdf as baseline

ROOT=Path(__file__).resolve().parents[2]


def measure():
    rows,arrays=[],{}
    for name,mesh in fixtures().items():
        pitch=0.125
        anchor=np.asarray(mesh.bounds).min(axis=0)-0.5
        _,shape=baseline._fonster(mesh.vertices,pitch,anchor)
        # Fixed local frame and capacity; only sample origin changes between calls.
        grid=np.indices(shape).reshape(3,-1).T*pitch
        corners=np.asarray(mesh.vertices[mesh.faces]-anchor,dtype='<f4')
        arrays[name+'_corners']=corners
        first=None
        for phase,offset in enumerate((np.zeros(3),pitch*np.array([0.25,0.5,0.75]),pitch*np.array([-0.5,0.25,-0.25]))):
            origin=anchor+offset
            mask,diag=baseline.solid_via_stralvindning(mesh.vertices,mesh.faces,pitch,origin,shape)
            distance=edt(mask,pitch)
            points=origin+grid
            points[:,0]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[0]
            points[:,1]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[1]
            rays=np.asarray(points-anchor,dtype='<f4')
            if first is None:first=mask.copy()
            key=f'{name}_{phase}'
            arrays[key+'_rays']=rays;arrays[key+'_mask']=mask;arrays[key+'_distance']=distance
            rows.append(dict(case=name,phase=phase,offset=offset.tolist(),shape=list(shape),triangles=len(mesh.faces),rays=len(rays),
                             occupied=int(mask.sum()),changed_vs_first=int(np.count_nonzero(mask!=first)),
                             hits=diag['n_traffar'],mask_sha256=digest(mask),distance_sha256=digest(distance),rays_sha256=digest(rays)))
    return rows,arrays


def main():
    first,a=measure();second,b=measure()
    gates=dict(full_arrays_repeat=all(np.array_equal(a[k],b[k]) for k in a),full_tables_repeat=first==second,
               finite=all(np.isfinite(v).all() for v in a.values()),changed_queries_observable=all(r['changed_vs_first']>0 for r in first if r['phase']>0))
    report=dict(rows=first,gates=gates,timing_measured=False)
    out=ROOT/'artifacts'
    (out/'field_rt_changed_query_reference.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'field_rt_changed_query_reference_arrays.npz',**a)
    print(json.dumps(report));return 0 if all(gates.values()) else 1

if __name__=='__main__':raise SystemExit(main())
