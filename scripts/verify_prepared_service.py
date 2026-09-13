"""Fresh isolated verification of the prepared-service slice; GPU use is explicit."""
import argparse
from field_paths import field_path
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from build_prepared_service import ARTIFACTS, ROOT, digest
from verify_csg import compare_arrays


def load_contract():
    matches = re.findall(r'<!-- prepared-verify-contract:v1 -->\s*```json\s*(.*?)\s*```',
                         (ROOT/'docs/RUNNING.md').read_text(), re.S)
    if len(matches) != 1:
        raise ValueError('Exactly one prepared verification contract required')
    value = json.loads(matches[0])
    if value['version'] != 1:
        raise ValueError('Unsupported contract')
    return value


def build_environment(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['version'] != 1 or manifest['artifacts'] != ARTIFACTS:
        raise ValueError('Unexpected build artifact contract')
    for name in ARTIFACTS.values():
        if digest(directory/name) != manifest['artifact_sha256'][name]:
            raise ValueError('Build artifact hash mismatch: '+name)
    if not manifest['source_sha256']:
        raise ValueError('Empty source manifest')
    for name, expected in manifest['source_sha256'].items():
        path = (ROOT/name).resolve()
        if not any(path.is_relative_to(ROOT/base) for base in ('src/field_engine', 'probes/field_engine')):
            raise ValueError('Build source outside field source folders')
        if digest(path) != expected:
            raise ValueError('Build source hash mismatch: '+name)
    return {key: str(directory/name) for key, name in ARTIFACTS.items()}


def check_report(actual, saved, item):
    key = item['checks']
    checks = actual.get(key, {})
    if not isinstance(checks, dict) or set(checks) != set(saved[key]) or len(checks) != item['count']:
        return False
    if not all(value is True for value in checks.values()):
        return False
    for key, value in item.get('decisive', {}).items():
        if actual.get(key) != value:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Run GPU probes under caller-owned GPU coordination')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--before-probe', type=Path, help='Optional caller-provided executable guard, run before each GPU probe')
    parser.add_argument('--plate', type=Path, help='Generated examples/parts/holed_plate_v1.stl')
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory; never overwrite captures')
    args = parser.parse_args()
    if args.gpu and (not args.build or not args.plate):
        parser.error('--gpu requires --build and --plate')
    specification = load_contract()
    items = specification['gpu' if args.gpu else 'cpu']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if args.gpu:
        env.update(build_environment(args.build.resolve()))
        env['FIELD_PLATE_STL'] = str(args.plate.resolve(strict=True))
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('Evidence directory must be empty')
    rows = []
    cleanup = json.loads((ROOT/'reports/prepared_source_cleanup.json').read_text())
    cleanup_path = ROOT/cleanup['file']
    if digest(cleanup_path) != cleanup['current_sha256'] or not cleanup['nonwhitespace_exact']:
        raise ValueError('Recorded whitespace-only source transition no longer applies')
    relocation = json.loads((ROOT/'reports/probe_relocation.json').read_text())
    transitions = {}
    for row in relocation['rows']:
        path = (ROOT/row['new']).resolve(); path.relative_to(ROOT)
        if digest(path) != row['new_sha256'] or not (row.get('numerical_ast_exact') is True or row.get('only_include_paths_changed') is True):
            raise ValueError('Recorded source relocation no longer applies: '+row['new'])
        previous = transitions.setdefault(row['new_sha256'], row['old_sha256'])
        if previous != row['old_sha256']:
            raise ValueError('Ambiguous source relocation digest')
    def normalized(value):
        if isinstance(value, dict):
            return {k: normalized(v) for k, v in value.items()}
        value = cleanup['current_sha256'] if value == cleanup['captured_sha256'] else value
        return transitions.get(value, value) if isinstance(value, str) else value
    with tempfile.TemporaryDirectory(prefix='field-prepared-verify-') as directory:
        scratch = Path(directory)
        shutil.copytree(ROOT/'src', scratch/'src', ignore=shutil.ignore_patterns('__pycache__', 'artifacts'))
        for extra in ('probes', 'scripts'):
            if (ROOT/extra).is_dir():
                shutil.copytree(ROOT/extra, scratch/extra, ignore=shutil.ignore_patterns('__pycache__', 'artifacts'))
        (scratch/'reports').mkdir()
        env['PYTHONPATH'] = str(scratch/'src')+os.pathsep+str(scratch/'src/field_engine')
        for item in items:
            module, name = item['module'], item['report']
            if Path(module).name != module or Path(name).name != name:
                raise ValueError('Invalid probe path')
            if args.gpu and args.before_probe:
                subprocess.run([str(args.before_probe.resolve())], check=True, timeout=10)
            start = time.perf_counter()
            timed_out = False
            try:
                child = subprocess.run([sys.executable, str(field_path(scratch, module))],
                                       cwd=scratch, env=env, capture_output=True, text=True, timeout=150)
                code, log = child.returncode, child.stdout+'\n'+child.stderr
            except subprocess.TimeoutExpired:
                code, log, timed_out = -1, 'Probe timed out', True
            saved = json.loads((ROOT/'reports'/f'{name}.json').read_text())
            path = scratch/'reports'/f'{name}.json'
            actual = json.loads(path.read_text()) if path.exists() else {}
            checks = code == 0 and check_report(actual, saved, item)
            sources = normalized(actual.get('source_sha256')) == normalized(saved.get('source_sha256'))
            exact, size = True, 0
            if item['arrays']:
                capture = scratch/'reports'/f'{name}_arrays.npz'
                exact, size = compare_arrays(ROOT/'reports'/f'{name}_arrays.npz', capture) if capture.exists() else (False, 0)
            for evidence in (scratch/'reports').glob(name+'*'):
                shutil.copy2(evidence, output/evidence.name)
            for prefix, replacement in ((str(scratch), '<isolated-root>'), (str(ROOT), '<repo>'),
                                        (sys.prefix, '<python-env>')):
                log = log.replace(prefix, replacement)
            (output/f'{name}.log').write_text(log)
            row = dict(module=module, exit=code, checks=checks, sources=sources, arrays_exact=exact,
                       bytes_compared=size, timed_out=timed_out, elapsed_s=time.perf_counter()-start)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if not all((checks, sources, exact)):
                break
    gates = dict(coverage=len(rows)==len(items), fresh=all(r['checks'] for r in rows),
                 source_hashes=all(r['sources'] for r in rows), arrays=all(r['arrays_exact'] for r in rows))
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates, rows=rows,
                  scope=('GPU helper/raw-hit/full owned-service slice; unchanged 2 ms warm L gate. '
                         if args.gpu else 'CPU interval and service-ownership slice; no GPU execution. ')+
                        'Complete captures retained; not all-module verification.',
                  contract_sha256=__import__('hashlib').sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest())
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
