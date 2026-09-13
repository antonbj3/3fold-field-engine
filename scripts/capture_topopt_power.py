"""Capture or replay every power call of the frozen topology fixture.

Usage: python scripts/capture_topopt_power.py SOURCE_ROOT NEW_OUTPUT capture
       python scripts/capture_topopt_power.py SOURCE_ROOT NEW_OUTPUT replay REFERENCE
Only fresh source copies execute; all numerical artifacts stay under NEW_OUTPUT.
"""
import os,sys,json,hashlib,shutil,tempfile,argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('source_root',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('mode',choices=['capture','replay'])
parser.add_argument('reference',type=Path,nargs='?')
args=parser.parse_args()
if (args.mode=='replay') != (args.reference is not None):parser.error('Replay requires a reference; capture must not supply one')
root=args.source_root.resolve();out=args.output.resolve();mode=args.mode;out.mkdir(parents=True,exist_ok=False)
source_sha={n:hashlib.sha256((root/'src/field_engine'/n).read_bytes()).hexdigest() for n in ['lastfalt_v1_topopt.py','lastfalt_v1_fem.py']}
reference=None
try:
 if mode=='replay':
  reference_root=args.reference.resolve();meta=json.loads((reference_root/'capture.json').read_text());data=reference_root/'power_calls.npz'
  if hashlib.sha256(data.read_bytes()).hexdigest()!=meta['npz_sha256']:raise RuntimeError('Power archive integrity mismatch before execution')
  if source_sha!=meta['source_sha256']:raise RuntimeError('Numerical source mismatch before execution')
  reference=np.load(data,allow_pickle=False)
 scratch_owner=tempfile.TemporaryDirectory(prefix='field-power-capture-')
 scratch=Path(scratch_owner.name)/'src/field_engine';scratch.mkdir(parents=True)
 for name in source_sha:shutil.copy2(root/'src/field_engine'/name,scratch/name)
 sys.path.insert(0,str(scratch));import lastfalt_v1_topopt as TO
 TO.OUT_DIR=str(out/'numerical_artifacts');Path(TO.OUT_DIR).mkdir()
 original_power=np.power;arrays={};calls=[]
 def power(x,y,*args,**kwargs):
  if not isinstance(x,np.ndarray) or x.shape!=(5120,):return original_power(x,y,*args,**kwargs)
  i=len(calls);prefix=f'call{i:02}';exponent=np.asarray(y)
  if reference is not None:
   a=reference[prefix+'_input'];b=reference[prefix+'_exponent']
   if x.dtype!=a.dtype or x.shape!=a.shape or x.tobytes()!=a.tobytes() or exponent.dtype!=b.dtype or exponent.tobytes()!=b.tobytes():
    raise RuntimeError('Captured operand mismatch at power call '+str(i))
  computed=original_power(x,y,*args,**kwargs);result=computed if reference is None else reference[prefix+'_result'].copy()
  arrays[prefix+'_input']=x.copy();arrays[prefix+'_exponent']=exponent.copy();arrays[prefix+'_result']=result.copy()
  if reference is not None:arrays[prefix+'_computed']=computed.copy()
  calls.append(dict(index=i,exponent=float(y),strides=list(x.strides),computed_returned_differences=int(np.count_nonzero(computed!=result))))
  return result
 np.power=power
 report=TO.run(n_iter=12,tag='field_to_recipe_v1')
 if len(calls)!=24 or [c['exponent'] for c in calls]!=[3.,2.]*12:raise RuntimeError('Unexpected power-call coverage')
 np.savez_compressed(out/'power_calls.npz',**arrays);shutil.copy2(Path(TO.OUT_DIR)/'rho_field_field_to_recipe_v1.npy',out/'density.npy')
 (out/'topopt.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 runtimes=[]
 for info in threadpool_info():
  path=Path(info.pop('filepath'));info.update(binary_name=path.name,binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest());runtimes.append(info)
 capture=dict(status='VERIFIED-FRESH',mode=mode,source_sha256=source_sha,calls=calls,npz_sha256=hashlib.sha256((out/'power_calls.npz').read_bytes()).hexdigest(),density_sha256=hashlib.sha256((out/'density.npy').read_bytes()).hexdigest(),numpy_core_sha256=hashlib.sha256(Path(np._core._multiarray_umath.__file__).read_bytes()).hexdigest(),requested_numpy_disabled=os.environ.get('NPY_DISABLE_CPU_FEATURES',''),requested_openblas_core=os.environ.get('OPENBLAS_CORETYPE','native'),runtimes=runtimes,array_bytes=sum(a.nbytes for a in arrays.values()),numpy_dispatch_enabled={k:bool(np._core._multiarray_umath.__cpu_features__.get(k)) for k in np._core._multiarray_umath.__cpu_dispatch__})
 (out/'capture.json').write_text(json.dumps(capture,indent=2)+'\n');print(json.dumps(dict(mode=mode,calls=len(calls),density_sha256=capture['density_sha256'])))
except (RuntimeError,ValueError,KeyError,OSError) as error:
 (out/'failure.json').write_text(json.dumps(dict(status='OWN-GATE-FAIL',mode=mode,error=str(error)),indent=2)+'\n');print(str(error),file=sys.stderr);sys.exit(2)
