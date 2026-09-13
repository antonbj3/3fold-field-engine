"""Run explicitly declared legacy recipe selftests without changing their gates."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from verify import ROOT, contract
from verify_csg import check_decisive


def final_report(stdout, overall_key="all_pass"):
    """Accept one terminal JSON object with the selftest's explicit overall gate."""
    candidates = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char != '{':
            continue
        try:
            value, end = decoder.raw_decode(stdout[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and overall_key in value and not stdout[index+end:].strip():
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError('Expected one terminal selftest JSON object')
    return candidates[0]


def read_report(stdout, item, recipe_root):
    """Read a declared fresh artifact or the explicitly gated terminal report."""
    if 'report_artifact' not in item:
        return final_report(stdout, item.get('overall_key', 'all_pass'))
    relative = Path(item['report_artifact'])
    if not relative.parts or relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'artifacts' or relative.suffix != '.json':
        raise ValueError('Expected relative JSON artifact inside the recipe output directory')
    path = recipe_root/relative
    path.resolve().relative_to((recipe_root/'artifacts').resolve())
    if path.is_symlink():
        raise ValueError('Expected newly written regular JSON artifact')
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError('Expected JSON artifact object')
    for keys in item.get('self_artifact_paths', []):
        value = report
        for key in keys:
            value = value[key]
        if not isinstance(value, str) or Path(value).resolve() != path.resolve():
            raise ValueError('Self-reference must address this fresh report artifact')
    return report


def own_checks(report, item):
    declared_checks = item.get('checks', [])
    if not declared_checks or any(not isinstance(x, str) for x in declared_checks) or len(declared_checks)!=len(set(declared_checks)):
        return False
    overall = report.get(item.get('overall_key', 'all_pass'))
    expected = item.get('overall_value', True)
    if type(overall) is not type(expected) or overall != expected:
        return False
    if item['kind'] == 'sketch_atoms':
        atoms = [a for a in report['ATOMS']['atoms'] if a['type']=='value-in-artifact']
        names = [a['key'] for a in atoms]
        checks = item['checks']
        declared = ['.'.join(map(str, a['path'])) for a in item['artifact_assertions']+item['decisive']]
        if (len(names)!=len(set(names)) or set(names)!=set(checks) or
                len(declared)!=len(set(declared)) or set(declared)!=set(checks)):
            return False
        return (artifact_assertions(report, item['artifact_assertions'])
                and check_decisive(report, item['decisive'])
                and report['failures']==[]
                and type(report['facits']['tangentcirkelkedja']['dof']) is int
                and report['facits']['tangentcirkelkedja']['dof']==0
                and len(report['diagnosis_demo']['overbestamd']['top_blame'])>=2
                and type(report['timing_ms']['tangentcirkelkedja']['median']) in (int, float)
                and math.isfinite(report['timing_ms']['tangentcirkelkedja']['median'])
                and report['timing_ms']['tangentcirkelkedja']['median']>0
                and report['diagnosis_demo']['geometrisk_giltighetsgrind']['status'] in
                    ('GEOMETRISKT_OGILTIG', 'OVERBESTAMD_KONFLIKT'))
    case_gate = item.get('case_gate', 'pass')
    if item['kind'] == 'boolean_checks':
        checks = report.get('checks', {})
        return isinstance(checks, dict) and set(checks)==set(item['checks']) and all(v is True for v in checks.values())
    if item['kind'] == 'dict':
        container = report
        for key in item.get('checks_path', []):
            container = container[key]
        checks = {key: value for key, value in container.items() if isinstance(value, dict) and case_gate in value}
    elif item['kind'] in ('results', 'checks', 'case_list'):
        if item['kind']=='case_list':
            values = report
            for key in item['checks_path']:
                values = values[key]
        else:
            values = report.get(item['kind'], [])
        if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
            return False
        checks = {value.get(item.get('case_id', 'id')): value for value in values}
        if len(checks) != len(values):
            return False
    else:
        raise ValueError('Unknown selftest schema')
    expected_case = item.get('case_value', True)
    return set(checks)==set(item['checks']) and all(
        type(value.get(case_gate)) is type(expected_case) and value.get(case_gate)==expected_case
        for value in checks.values())



def artifact_assertions(report, assertions):
    """Evaluate frozen analytic value bands, never commands from report data."""
    for assertion in assertions:
        value = report
        try:
            for key in assertion['path']:
                value = value[key]
            expected, tolerance = assertion['value'], assertion['atol']
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in (value, expected, tolerance)):
                return False
            if tolerance < 0 or abs(value-expected) > tolerance:
                return False
        except (KeyError, IndexError, TypeError):
            return False
    return True


def normalized_report(report, item):
    """Project only named clock observations and generated STEP directory names."""
    result = copy.deepcopy(report)
    observations = {}
    for path in item.get('timing_paths', []):
        sketch_clock = (len(path)==3 and path[0]=='timing_ms' and
                        path[1] in {'tangentcirkelkedja', 'dimensionerad_fyrkant_fasning', 'spline_tangenthandtag'} and
                        path[2] in {'median', 'p95', 'all_ms'})
        if not sketch_clock and (not path or path[-1] not in {'t_cold_s', 't_warm_s', 't_mem_s', 'speedup_x', 'wall_ms', 'wall_ms_miss_run', 'wall_ms_hit_run'}):
            raise ValueError('Only declared clock observations may be omitted')
        parent = result
        for key in path[:-1]:
            parent = parent[key]
        observations['.'.join(map(str, path))] = parent.pop(path[-1])
    for path in item.get('basename_paths', []):
        if not path or path[-1] != 'step':
            raise ValueError('Only generated STEP locations may use basename projection')
        parent = result
        for key in path[:-1]:
            parent = parent[key]
        value = parent[path[-1]]
        if not isinstance(value, str) or Path(value).suffix != '.step':
            raise ValueError('Expected generated STEP location')
        parent[path[-1]] = Path(value).name
    for path in item.get('self_artifact_paths', []):
        if len(path)!=4 or path[:2]!=['ATOMS', 'atoms'] or type(path[2]) is not int or path[3]!='artifact':
            raise ValueError('Only explicit ATOMS self-reference paths may be projected')
        parent = result['ATOMS']['atoms'][path[2]]
        value = parent['artifact']
        if not isinstance(value, str) or Path(value).name != Path(item['report_artifact']).name:
            raise ValueError('Self-reference filename must match the declared artifact')
        parent['artifact'] = item['report_artifact']
    return result, observations


def retain_captures(scratch, item, output):
    """Retain every declared complete fresh array file, including mismatches."""
    rows = []
    for capture in item.get('captures', []):
        relative = Path(capture['path'])
        path = scratch/relative
        row = {'path': str(relative), 'exact': False, 'bytes': 0, 'sha256': None, 'error': None}
        try:
            if relative.is_absolute() or '..' in relative.parts or path.suffix not in {'.npy', '.npz'}:
                raise ValueError('Expected relative complete array capture')
            path.resolve().relative_to((scratch/'src/field_engine/artifacts').resolve())
            if path.is_symlink():
                raise ValueError('Expected newly written regular array capture')
            data = path.read_bytes()
            (output/(Path(item['module']).name+'.'+path.name)).write_bytes(data)
            row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            row['exact'] = row['sha256']==capture['sha256'] and row['bytes']==capture['bytes']
        except (ValueError, OSError) as failure:
            row['error'] = str(failure)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specification = contract((ROOT/'docs/RUNNING.md').read_text(), 'recipe-verify-contract:v1')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('Evidence output must be empty')
    for name, expected in specification['source_sha256'].items():
        path = (ROOT/name).resolve(); path.relative_to(ROOT)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Documented recipe source mismatch: '+name)
    rows = []
    for item in specification['probes']:
        with tempfile.TemporaryDirectory(prefix='field-recipe-verify-') as directory:
            scratch = Path(directory)
            shutil.copytree(ROOT/'src', scratch/'src', ignore=shutil.ignore_patterns('__pycache__', 'artifacts'))
            shutil.copytree(ROOT/'examples', scratch/'examples', ignore=shutil.ignore_patterns('__pycache__', '*.stl', '*.step'))
            (scratch/'src/field_engine/recipe/artifacts').mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', FIELD_ENGINE_DEVICE='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       PYTHONPATH=str(scratch/'src/field_engine/recipe')+os.pathsep+str(scratch/'src/field_engine'))
            module = (scratch/'src/field_engine'/item['module']).resolve(); module.relative_to(scratch/'src/field_engine')
            child = subprocess.run([sys.executable, str(module), *item['args']], cwd=scratch, env=env,
                                   capture_output=True, text=True, timeout=180)
            name = Path(item['module']).name
            (output/(name+'.stdout')).write_text(child.stdout)
            (output/(name+'.stderr')).write_text(child.stderr)
            capture_rows = retain_captures(scratch, item, output)
            captures_exact = all(r['exact'] for r in capture_rows)
            try:
                report = read_report(child.stdout, item, module.parent)
                raw = json.dumps(report, sort_keys=True, indent=2)+'\n'
                (output/(name+'.json')).write_text(raw)
                projected, observations = normalized_report(report, item)
                canonical = json.dumps(projected, sort_keys=True, indent=2)+'\n'
                digest = hashlib.sha256(canonical.encode()).hexdigest()
                (output/(name+'.normalized.json')).write_text(canonical)
                passed = own_checks(report, item)
                decisive = check_decisive(report, item['decisive']) and artifact_assertions(report, item.get('artifact_assertions', []))
                error = None
            except (ValueError, TypeError, KeyError, OSError) as failure:
                digest, passed, decisive, error, observations = None, False, False, str(failure), {}
            row = dict(module=item['module'], exit=child.returncode, gates=passed, decisive=decisive,
                       declared_report_exact=digest==item['report_sha256'], observational_timing=observations, report_sha256=digest,
                       own_check_count=len(item['checks']), captures=capture_rows, captures_exact=captures_exact, error=error)
            rows.append(row); print(json.dumps(row), flush=True)
    gates = dict(coverage=len(rows)==len(specification['probes']), fresh=all(r['exit']==0 and r['gates'] for r in rows),
                 decisive=all(r['decisive'] for r in rows), reports=all(r['declared_report_exact'] and r['captures_exact'] for r in rows),
                 source_hashes=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in specification['source_sha256'].items()))
    result = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates, rows=rows,
                  requested_openblas_core=os.environ.get('OPENBLAS_CORETYPE', 'native'),
                  requested_numpy_cpu_features_disabled=os.environ.get('NPY_DISABLE_CPU_FEATURES', ''),
                  source_sha256=specification['source_sha256'], scope='Explicit executable recipe selftests; every own gate and all report fields except named clock observations/generated STEP directories/self-references. Raw reports retained; no imported-library coverage inferred.')
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
