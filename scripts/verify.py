"""Run declared numerical recipes and report every unmapped fresh-status row."""
import argparse
from field_paths import field_path
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def fresh_rows(text):
    rows = {}
    for line in text.splitlines():
        cells = [x.strip().strip('`') for x in line.split('|')]
        if len(cells) < 4 or cells[2] != 'VERIFIED-FRESH':
            continue
        name = cells[1]
        if name in rows:
            raise ValueError('Duplicate current VERIFIED-FRESH row: '+name)
        path = (ROOT/name).resolve()
        path.relative_to(ROOT)
        if not path.is_file():
            raise ValueError('Missing declared source: '+name)
        rows[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not rows:
        raise ValueError('No current VERIFIED-FRESH rows')
    return rows


def contract(text, marker):
    matches = re.findall(r'<!-- '+re.escape(marker)+r' -->\s*```json\s*(.*?)\s*```', text, re.S)
    if len(matches) != 1:
        raise ValueError('Exactly one '+marker+' contract required')
    result = json.loads(matches[0])
    if result.get('version') != 1:
        raise ValueError('Unsupported contract version')
    return result


def declarations(text):
    csg = contract(text, 'csg-verify-contract:v1')['probes']
    prepared = contract(text, 'prepared-verify-contract:v1')
    groups = {'csg_cpu': csg, 'prepared_cpu': prepared['cpu'], 'prepared_gpu': prepared['gpu'],
              'geometry_cpu': contract(text, 'geometry-verify-contract:v1')['probes'],
              'recipe_cpu': contract(text, 'recipe-verify-contract:v1')['probes']}
    seen = set()
    for items in groups.values():
        if not items:
            raise ValueError('Empty recipe group')
        for item in items:
            name = item['module']
            if Path(name).is_absolute() or '..' in Path(name).parts or not name.endswith('.py') or name in seen:
                raise ValueError('Invalid or duplicate declared probe')
            seen.add(name)
    return groups



AGGREGATE_GATES = {
    'csg_cpu': {'coverage', 'fresh', 'documented_numbers', 'arrays', 'source_hashes', 'negative_controls'},
    'prepared_cpu': {'coverage', 'fresh', 'source_hashes', 'arrays'},
    'geometry_cpu': {'coverage', 'fresh', 'reports', 'arrays', 'source_hashes'},
    'recipe_cpu': {'coverage', 'fresh', 'decisive', 'reports', 'source_hashes'},
}


def accepted_result(code, result, group, items):
    checks = result.get('gates', {})
    rows = result.get('rows', [])
    expected = [item['module'] for item in items]
    return (code == 0 and result.get('status') == 'VERIFIED-FRESH'
            and isinstance(checks, dict) and set(checks) == AGGREGATE_GATES[group]
            and all(value is True for value in checks.values())
            and [row.get('module') for row in rows] == expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declared', action='store_true', help='Verify only explicitly mapped CPU rows; coverage remains reported')
    parser.add_argument('--group', choices=tuple(AGGREGATE_GATES), help='Select one CPU group, only with --declared')
    parser.add_argument('--inventory', action='store_true', help='Read-only mapping audit without numerical execution')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.group and not args.declared:
        parser.error('--group requires --declared')
    selected_groups = [args.group] if args.group else list(AGGREGATE_GATES)
    text = (ROOT/'docs/RUNNING.md').read_text()
    sources = fresh_rows(text)
    groups = declarations(text)
    mapping = {str(field_path(ROOT,item['module']).relative_to(ROOT)): group for group, items in groups.items() for item in items}
    unexpected = sorted(set(mapping)-set(sources))
    if unexpected:
        raise ValueError('Mapped probe lacks a current fresh row: '+', '.join(unexpected))
    unmapped = sorted(set(sources)-set(mapping))
    output = (args.output or ROOT/'.verify-runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('Evidence output must be empty')
    report = dict(status='OWN-GATE-FAIL', source_sha256=sources, mapping=mapping, unmapped=unmapped,
                  runtime_deferred=sorted(k for k, v in mapping.items() if v=='prepared_gpu'), executions=[],
                  selected_groups=selected_groups,
                  scope='CPU declared numerical recipes only. Native/imported helpers are not counted from parent imports. '
                        'GPU rows require the separate explicit prepared GPU command. Full coverage requires every current fresh row.',
                  running_sha256=hashlib.sha256(text.encode()).hexdigest())
    if not args.inventory:
        with tempfile.TemporaryDirectory(prefix='field-declared-verify-') as directory:
            scratch = Path(directory)
            for folder in ('src', 'scripts', 'probes'):
                shutil.copytree(ROOT/folder, scratch/folder, ignore=shutil.ignore_patterns('__pycache__', 'artifacts'))
            shutil.copytree(ROOT/'examples', scratch/'examples',
                            ignore=shutil.ignore_patterns('__pycache__', '*.stl', '*.step'))
            (scratch/'docs').mkdir(); (scratch/'reports').mkdir()
            (scratch/'docs/RUNNING.md').write_text(text)
            names = {item['report'] for group, items in groups.items() if group not in ('geometry_cpu', 'recipe_cpu') for item in items}
            for name in names:
                for suffix in ('.json', '_arrays.npz'):
                    path = ROOT/'reports'/(name+suffix)
                    if path.exists():
                        shutil.copy2(path, scratch/'reports'/path.name)
            shutil.copy2(ROOT/'reports/prepared_source_cleanup.json', scratch/'reports/prepared_source_cleanup.json')
            shutil.copy2(ROOT/'reports/probe_relocation.json', scratch/'reports/probe_relocation.json')
            geometry_files = {item['baseline'] for item in groups['geometry_cpu']}
            geometry_files.update(contract(text, 'geometry-verify-contract:v1').get('resource_sha256', {}))
            geometry_files.update(c['expected'] for item in groups['geometry_cpu'] for c in item['captures'])
            for name in geometry_files:
                original = (ROOT/name).resolve(); original.relative_to(ROOT)
                target = scratch/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', FIELD_ENGINE_DEVICE='cpu',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            for group, script, parameters, result_path in (
                ('csg_cpu', 'verify_csg.py', ['--output', str(output/'csg_cpu')], output/'csg_cpu/verification.json'),
                ('prepared_cpu', 'verify_prepared_service.py', ['--output', str(output/'prepared_cpu')], output/'prepared_cpu/verification.json'),
                ('geometry_cpu', 'verify_geometry.py', ['--output', str(output/'geometry_cpu')], output/'geometry_cpu/verification.json'),
                ('recipe_cpu', 'verify_recipe_selftests.py', ['--output', str(output/'recipe_cpu')], output/'recipe_cpu/verification.json'),
            ):
                if group not in selected_groups:
                    continue
                child = subprocess.run([sys.executable, str(scratch/'scripts'/script), *parameters],
                                       cwd=scratch, env=env, capture_output=True, text=True, timeout=600)
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                passed = accepted_result(child.returncode, result, group, groups[group])
                report['executions'].append(dict(group=group, exit=child.returncode, passed=passed,
                                                 result=result, modules=[x['module'] for x in groups[group]]))
                log = (child.stdout+'\n'+child.stderr).replace(str(scratch), '<isolated-root>').replace(str(ROOT), '<repo>').replace(sys.prefix, '<python-env>')
                (output/(group+'.log')).write_text(log)
                if not passed:
                    break
    gates = dict(mapped_cpu=len(report['executions'])==len(selected_groups) and all(r['passed'] for r in report['executions']),
                 full_coverage=not unmapped and not report['runtime_deferred'])
    report['gates'] = gates
    report['selected_status'] = 'VERIFIED-FRESH' if gates['mapped_cpu'] else 'OWN-GATE-FAIL'
    if all(gates.values()):
        report['status'] = 'VERIFIED-FRESH'
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], selected_status=report['selected_status'],
                         fresh_rows=len(sources), mapped_rows=len(mapping), unmapped=len(unmapped),
                         gpu_deferred=len(report['runtime_deferred']), gates=gates)), flush=True)
    if args.inventory:
        return 0 if gates['full_coverage'] else 2
    return 0 if gates['mapped_cpu'] and (args.declared or gates['full_coverage']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
