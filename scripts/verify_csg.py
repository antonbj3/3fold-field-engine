"""Re-run the documented CPU CSG slice in an isolated disposable tree.

This is a subset verifier, not all-repository or GPU verification. Numerical
contracts come from RUNNING.md. Complete arrays are compared to committed
captures; timing samples are retained but not compared for exact equality.
"""
import argparse
from field_paths import field_path
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def contract(text):
    marker=r'<!-- csg-verify-contract:v1 -->\s*```json\s*(.*?)\s*```'
    matches=re.findall(marker,text,re.S)
    if len(matches)!=1:
        raise ValueError('Exactly one documented CSG verification contract required')
    value=json.loads(matches[0])
    if value.get('version')!=1 or not value.get('probes'):
        raise ValueError('Unsupported or empty verification contract')
    return value


def compare_arrays(expected,actual):
    with np.load(expected,allow_pickle=False) as a,np.load(actual,allow_pickle=False) as b:
        if set(a.files)!=set(b.files):
            return False,0
        total=0
        for key in sorted(a.files):
            x,y=a[key],b[key]
            if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():
                return False,total
            total+=x.nbytes
        return True,total


def check_decisive(report,expected):
    for item in expected:
        value=report
        for part in item['path']:
            try:
                value=value[part]
            except (KeyError, IndexError, TypeError):
                return False
        if isinstance(item['value'],bool) and value is not item['value']:
            return False
        if value!=item['value']:
            return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='New evidence directory retaining every fresh report and capture')
    args=parser.parse_args()
    evidence=args.output.resolve() if args.output else None
    if evidence is not None:
        evidence.mkdir(parents=True,exist_ok=True)
        if any(evidence.iterdir()):parser.error('Evidence directory must be empty')
    specification=contract((ROOT/'docs/RUNNING.md').read_text())
    rows=[]
    with tempfile.TemporaryDirectory(prefix='field-csg-verify-') as directory:
        scratch=Path(directory)
        shutil.copytree(ROOT/'src',scratch/'src',ignore=shutil.ignore_patterns('__pycache__','artifacts'))
        for extra in ('probes', 'scripts'):
            if (ROOT/extra).is_dir():
                shutil.copytree(ROOT/extra, scratch/extra, ignore=shutil.ignore_patterns('__pycache__', 'artifacts'))
        (scratch/'reports').mkdir()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 PYTHONPATH=str(scratch/'src')+os.pathsep+str(scratch/'src/field_engine'))
        for item in specification['probes']:
            name=item['module']; report_name=item['report']
            # Contract entries may identify files only within the fixed source/report folders.
            if Path(name).name!=name or Path(report_name).name!=report_name or not name.endswith('.py'):
                raise ValueError('Invalid documented probe path')
            start=time.perf_counter()
            run=subprocess.run([sys.executable,str(field_path(scratch, name))],cwd=scratch,env=env,
                               capture_output=True,text=True,timeout=180)
            result_path=scratch/'reports'/f'{report_name}.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else {}
            saved=json.loads((ROOT/'reports'/f'{report_name}.json').read_text())
            checks=result.get('gates',{})
            passed=(run.returncode==0 and isinstance(checks,dict) and set(checks)==set(saved['gates'])
                    and len(checks)==item['gate_count'] and all(value is True for value in checks.values()))
            numeric=check_decisive(result,item.get('decisive',[])) if result else False
            output=scratch/'reports'/f'{report_name}_arrays.npz'
            exact,bytes_compared=compare_arrays(ROOT/'reports'/f'{report_name}_arrays.npz',output) if output.exists() else (False,0)
            # Native source hashes, if present, must still match the captured source.
            sources=all(result.get(k)==v for k,v in saved.items() if k.endswith('source_sha256'))
            row=dict(module=name,exit=run.returncode,gates=passed,decisive=numeric,arrays_exact=exact,
                     source_hashes=sources,bytes_compared=bytes_compared,elapsed_s=time.perf_counter()-start,
                     observed_gates=result.get('gates',{}))
            if run.stderr:
                row['stderr']=run.stderr[-3000:].replace(str(scratch),'<isolated-root>').replace(str(ROOT),'<repo>').replace(sys.prefix,'<python-env>')
            if evidence is not None:
                for path in (result_path,output):
                    if path.exists():shutil.copy2(path,evidence/path.name)
                log=(run.stdout+'\n'+run.stderr).replace(str(scratch),'<isolated-root>').replace(str(ROOT),'<repo>').replace(sys.prefix,'<python-env>')
                (evidence/f'{report_name}.log').write_text(log)
            rows.append(row)
            print(json.dumps(row),flush=True)
            if not all((passed,numeric,exact,sources)):
                break
    # Independent negative controls: stale documented number and changed array.
    rejects=not check_decisive({'value':1},[dict(path=['value'],value=2)])
    with tempfile.TemporaryDirectory(prefix='field-csg-controls-') as directory:
        a=Path(directory)/'a.npz';b=Path(directory)/'b.npz'
        np.savez(a,x=np.array([1,2],np.float32));np.savez(b,x=np.array([1,3],np.float32))
        rejects &= not compare_arrays(a,b)[0]
    gates=dict(coverage=len(rows)==len(specification['probes']),fresh=all(r['gates'] for r in rows),
               documented_numbers=all(r['decisive'] for r in rows),arrays=all(r['arrays_exact'] for r in rows),
               source_hashes=all(r['source_hashes'] for r in rows),negative_controls=bool(rejects))
    report=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,
                contract_sha256=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest(),
                scope='CPU CSG subset only; seven fresh probes in isolated tree. Full archived arrays and documented decisive values exact. Live speed gates are required, timing values are not equality targets. No GPU or all-repository verification claim.')
    ((evidence/'verification.json') if evidence is not None else ROOT/'reports/csg_verify.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],gates=gates)),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
