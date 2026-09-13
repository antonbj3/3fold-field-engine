"""Reproduce declared geometry gates, complete reports and frozen field captures."""
import argparse
from field_paths import field_path
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from verify import ROOT, contract
from verify_csg import compare_arrays, check_decisive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', action='append', help='Explicit module selection; all omitted rows remain reported')
    args = parser.parse_args()
    specification = contract((ROOT/'docs/RUNNING.md').read_text(), 'geometry-verify-contract:v1')
    items = specification['probes']
    if args.module:
        if not set(args.module) <= {item['module'] for item in items}:
            parser.error('Unknown selected geometry module')
        items = [item for item in items if item['module'] in args.module]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('Evidence output must be empty')
    for name, expected in specification['source_sha256'].items():
        path = (ROOT/name).resolve(); path.relative_to(ROOT)
        if digest(path) != expected:
            raise ValueError('Documented source hash mismatch: '+name)
    for name, expected in specification.get('resource_sha256', {}).items():
        path = (ROOT/name).resolve(); path.relative_to(ROOT)
        if digest(path) != expected:
            raise ValueError('Documented resource hash mismatch: '+name)
    rows, executed, command_roots = [], {}, {}
    with tempfile.TemporaryDirectory(prefix='field-geometry-verify-') as directory:
        scratch = Path(directory)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', FIELD_ENGINE_DEVICE='cpu',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        for item in items:
            command = item['command']
            if Path(command).name != command or not command.endswith('.py'):
                raise ValueError('Invalid geometry command')
            if command not in executed:
                work = scratch/command
                shutil.copytree(ROOT/'src', work/'src', ignore=shutil.ignore_patterns('__pycache__', 'artifacts'))
                for extra in ('probes', 'scripts'):
                    if (ROOT/extra).is_dir():
                        shutil.copytree(ROOT/extra, work/extra, ignore=shutil.ignore_patterns('__pycache__', 'artifacts'))
                shutil.copytree(ROOT/'examples', work/'examples', ignore=shutil.ignore_patterns('__pycache__', '*.stl', '*.step'))
                (work/'reports').mkdir()
                for name in specification.get('resource_sha256', {}):
                    target=work/name; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT/name, target)
                command_roots[command] = work
                env['PYTHONPATH'] = str(work/'src')+os.pathsep+str(work/'src/field_engine')
                child = subprocess.run([sys.executable, str(field_path(work, command))], cwd=work,
                                       env=env, capture_output=True, text=True, timeout=600)
                log = (child.stdout+'\n'+child.stderr).replace(str(scratch), '<isolated-root>').replace(str(ROOT), '<repo>').replace(sys.prefix, '<python-env>')
                (output/(command+'.log')).write_text(log)
                executed[command] = child.returncode
            work = command_roots[command]
            result_path = (work/item['report']).resolve(); result_path.relative_to(work)
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            checks = result.get('gates', {})
            gates = isinstance(checks, dict) and set(checks)==set(item['gates']) and all(v is True for v in checks.values())
            report_exact = digest(result_path)==item['report_sha256'] if result_path.exists() else False
            decisive = check_decisive(result, item['decisive']) if result else False
            captures_exact, size = True, 0
            for capture in item['captures']:
                actual = (work/capture['actual']).resolve(); actual.relative_to(work)
                expected = (ROOT/capture['expected']).resolve(); expected.relative_to(ROOT)
                if not actual.exists():
                    exact, count = False, 0
                elif actual.suffix=='.npz':
                    exact, count = compare_arrays(expected, actual)
                else:
                    exact, count = expected.read_bytes()==actual.read_bytes(), actual.stat().st_size
                captures_exact &= exact; size += count
                if actual.exists():
                    destination = output/item['module']/Path(capture['actual']).name
                    destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(actual, destination)
            if result_path.exists():
                destination = output/item['module']/'report.json'
                destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(result_path, destination)
            rows.append(dict(module=item['module'], command=command, exit=executed[command],
                             gates=gates, decisive=decisive, report_exact=report_exact, captures_exact=captures_exact,
                             bytes_compared=size, observed_gates=checks))
            print(json.dumps(rows[-1]), flush=True)
            if not all((executed[command]==0, gates, decisive, report_exact, captures_exact)):
                break
    gates = dict(coverage=len(rows)==len(items), fresh=all(r['exit']==0 and r['gates'] and r['decisive'] for r in rows),
                 reports=all(r['report_exact'] for r in rows), arrays=all(r['captures_exact'] for r in rows),
                 source_hashes=all(digest(ROOT/name)==value for name, value in specification['source_sha256'].items()))
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates, rows=rows,
                  source_sha256=specification['source_sha256'], executions=executed, selected_modules=[item['module'] for item in items],
                  omitted_modules=[item['module'] for item in specification['probes'] if item not in items],
                  scope='Geometry slice only. Frozen complete report bytes, full captures and documented source pins. '
                        'Loft module is explicitly executed by the chain, with its own six gates and full report checked separately; imports do not count.')
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
