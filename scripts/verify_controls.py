"""Negative controls for current-row coverage and nested numerical receipts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from verify import AGGREGATE_GATES, ROOT, accepted_result, contract, declarations, fresh_rows
from verify_csg import check_decisive, compare_arrays
from verify_recipe_selftests import final_report, own_checks, normalized_report, read_report, artifact_assertions, retain_captures


def main():
    source = 'scripts/verify.py'
    row = '| '+source+' | VERIFIED-FRESH | explicit control |\n'
    checks = {'valid_row': list(fresh_rows(row)) == [source]}
    for name, text in [('duplicate_row', row+row), ('empty_rows', ''),
                       ('missing_source', '| missing-controlled-file.py | VERIFIED-FRESH | |'),
                       ('escaped_source', '| ../../../../etc/passwd | VERIFIED-FRESH | |')]:
        try: fresh_rows(text)
        except ValueError: checks[name] = True
        else: checks[name] = False
    marker = '<!-- test:v1 -->\n```json\n{"version":1}\n```\n'
    checks['valid_contract'] = contract(marker, 'test:v1') == {'version': 1}
    for name, text in [('duplicate_contract', marker+marker), ('unsupported_contract', marker.replace(':1', ':2'))]:
        try: contract(text, 'test:v1')
        except ValueError: checks[name] = True
        else: checks[name] = False
    items = [{'module': 'one.py'}, {'module': 'two.py'}]
    value = dict(status='VERIFIED-FRESH', gates={k: True for k in AGGREGATE_GATES['prepared_cpu']},
                 rows=[{'module': x['module']} for x in items])
    checks['valid_receipt'] = accepted_result(0, value, 'prepared_cpu', items)
    checks['child_failure'] = not accepted_result(1, value, 'prepared_cpu', items)
    variants = [('empty_gates', {'gates': {}}), ('invented_gate', {'gates': {'anything': True}}),
                ('omitted_execution', {'rows': value['rows'][:1]}), ('false_status', {'status': 'OWN-GATE-FAIL'})]
    for name, change in variants:
        altered = copy.deepcopy(value); altered.update(change)
        checks[name] = not accepted_result(0, altered, 'prepared_cpu', items)
    altered = copy.deepcopy(value); altered['gates']['fresh'] = False
    checks['false_nested_gate'] = not accepted_result(0, altered, 'prepared_cpu', items)
    text = (ROOT/'docs/RUNNING.md').read_text()
    groups = declarations(text)
    checks['explicit_probe_only'] = all(not Path(item['module']).is_absolute() and '..' not in Path(item['module']).parts and item['module'].endswith('.py')
                                        for items in groups.values() for item in items)
    for name, stdout in [('recipe_trailing_text', '{"all_pass":true} trailing'),
                         ('recipe_missing_overall', '{"pass":true}')]:
        try: final_report(stdout)
        except ValueError: checks[name] = True
        else: checks[name] = False
    recipe_item = {'kind': 'dict', 'checks': ['one', 'two']}
    checks['recipe_omitted_case'] = not own_checks({'all_pass': True, 'one': {'pass': True}}, recipe_item)
    checks['recipe_false_case'] = not own_checks({'all_pass': True, 'one': {'pass': True}, 'two': {'pass': False}}, recipe_item)
    checks['recipe_duplicate_case'] = not own_checks({'all_pass': True, 'results': [{'id': 'one', 'pass': True}, {'id': 'one', 'pass': True}]}, {'kind': 'results', 'checks': ['one']})
    boolean_item = {'kind': 'boolean_checks', 'checks': ['one', 'two']}
    checks['boolean_map_valid'] = own_checks({'all_pass': True, 'checks': {'one': True, 'two': True}}, boolean_item)
    for label, values in [('omitted', {'one': True}), ('false', {'one': True, 'two': False}),
                           ('integer', {'one': True, 'two': 1}), ('extra', {'one': True, 'two': True, 'three': True})]:
        checks['boolean_map_'+label+'_refused'] = not own_checks({'all_pass': True, 'checks': values}, boolean_item)
    checks['recipe_empty_case_contract_refused'] = not own_checks({'all_pass': True}, {'kind': 'dict', 'checks': []})
    checks['recipe_duplicate_case_contract_refused'] = not own_checks({'all_pass': True, 'one': {'pass': True}}, {'kind': 'dict', 'checks': ['one', 'one']})
    list_item = {'kind': 'case_list', 'checks_path': ['cases'], 'case_id': 'name', 'checks': ['one']}
    checks['explicit_case_list_valid'] = own_checks({'all_pass': True, 'cases': [{'name': 'one', 'pass': True}]}, list_item)
    for label, values in [('omitted', []), ('duplicate', [{'name': 'one', 'pass': True}]*2),
                          ('false', [{'name': 'one', 'pass': False}]), ('not_records', [1]), ('not_list', {})]:
        checks['explicit_case_list_'+label+'_refused'] = not own_checks({'all_pass': True, 'cases': values}, list_item)
    verdict_item = {'kind': 'results', 'case_id': 'name', 'case_gate': 'verdict', 'case_value': 'PASS', 'checks': ['one']}
    verdict_report = {'all_pass': True, 'results': [{'name': 'one', 'verdict': 'PASS'}]}
    checks['named_string_verdict_valid'] = own_checks(verdict_report, verdict_item)
    for label, records in [('false', [{'name': 'one', 'verdict': 'FAIL'}]),
                           ('boolean', [{'name': 'one', 'verdict': True}]),
                           ('duplicate', verdict_report['results']*2),
                           ('omitted', [])]:
        checks['named_string_verdict_'+label] = not own_checks({'all_pass': True, 'results': records}, verdict_item)
    for name, item in [('omit_geometry_refused', {'timing_paths': [['volume']]}),
                       ('omit_gate_refused', {'timing_paths': [['all_pass']]}),
                       ('basename_nonartifact_refused', {'basename_paths': [['volume']]})]:
        try: normalized_report({'volume': 1, 'all_pass': True}, item)
        except ValueError: checks[name] = True
        else: checks[name] = False
    projection = {'timing_paths': [['wall_ms']], 'basename_paths': [['step']]}
    original = {'volume': 1, 'wall_ms': 3, 'step': '/tmp/first/part.step', 'all_pass': True}
    a, _ = normalized_report(original, projection)
    b, _ = normalized_report(dict(original, wall_ms=9, step='/tmp/second/part.step'), projection)
    checks['clock_and_directory_only'] = a == b and original['wall_ms'] == 3
    b, _ = normalized_report(dict(original, volume=2), projection)
    checks['projected_geometry_change_detected'] = a != b
    b, _ = normalized_report(dict(original, all_pass=False), projection)
    checks['projected_gate_change_detected'] = a != b
    b, _ = normalized_report(dict(original, step='/tmp/first/other.step'), projection)
    checks['projected_filename_change_detected'] = a != b
    checks['boolean_not_integer'] = not check_decisive({'gate': 1}, [{'path': ['gate'], 'value': True}])
    checks['uppercase_case_failure'] = not own_checks({'PASS': True, 'one': {'PASS': False}},
                                                     {'kind': 'dict', 'overall_key': 'PASS', 'case_gate': 'PASS', 'checks': ['one']})
    checks['missing_decisive_path'] = not check_decisive({}, [{'path': ['missing'], 'value': 1}])
    checks['missing_decisive_index'] = not check_decisive({'rows': []}, [{'path': ['rows', 0], 'value': 1}])
    band = [{'path': ['volume'], 'value': 4.0, 'atol': 0.25}]
    checks['analytic_band_boundary'] = artifact_assertions({'volume': 4.25}, band)
    for label, value in [('outside', 4.2501), ('nan', float('nan')), ('infinite', float('inf')), ('boolean', True)]:
        checks['analytic_band_'+label] = not artifact_assertions({'volume': value}, band)
    checks['analytic_band_missing'] = not artifact_assertions({}, band)
    with tempfile.TemporaryDirectory(prefix='field-recipe-artifact-control-') as directory:
        root = Path(directory); (root/'artifacts').mkdir()
        item = {'report_artifact': 'artifacts/result.json'}
        for label, location in [('empty', ''), ('missing', 'artifacts/result.json'), ('escape', '../result.json'), ('outside', '/tmp/result.json')]:
            try: read_report('{"all_pass":true}', {'report_artifact': location}, root)
            except (ValueError, OSError): checks['artifact_'+label+'_refused'] = True
            else: checks['artifact_'+label+'_refused'] = False
        (root/'artifacts/result.json').write_text('{"all_pass":false}')
        checks['artifact_overrides_stdout_summary'] = read_report('{"all_pass":true}', item, root)['all_pass'] is False
        (root/'artifacts/link.json').symlink_to(root/'artifacts/result.json')
        try: read_report('', {'report_artifact': 'artifacts/link.json'}, root)
        except ValueError: checks['artifact_symlink_refused'] = True
        else: checks['artifact_symlink_refused'] = False
    sketch_item = {'kind': 'sketch_atoms', 'checks': ['volume'], 'decisive': [], 'artifact_assertions': band}
    sketch_report = {'all_pass': True, 'volume': 4.0, 'ATOMS': {'atoms': [{'type': 'value-in-artifact', 'key': 'volume'}]},
                     'failures': [], 'facits': {'tangentcirkelkedja': {'dof': 0}},
                     'timing_ms': {'tangentcirkelkedja': {'median': 1.0, 'n_reps': 15}},
                     'diagnosis_demo': {'overbestamd': {'top_blame': [1, 2]},
                                        'geometrisk_giltighetsgrind': {'status': 'OVERBESTAMD_KONFLIKT'}}}
    checks['sketch_predicates_valid'] = own_checks(sketch_report, sketch_item)
    missing_assertion = dict(sketch_item, artifact_assertions=[])
    checks['sketch_missing_declared_assertion_refused'] = not own_checks(sketch_report, missing_assertion)
    for name, path, value in [('coordinate', ['volume'], 5.0),
                              ('clock_zero', ['timing_ms', 'tangentcirkelkedja', 'median'], 0),
                              ('clock_nan', ['timing_ms', 'tangentcirkelkedja', 'median'], float('nan')),
                              ('dof', ['facits', 'tangentcirkelkedja', 'dof'], 1),
                              ('blame_omitted', ['diagnosis_demo', 'overbestamd', 'top_blame'], [1]),
                              ('invalid_geometry_green', ['diagnosis_demo', 'geometrisk_giltighetsgrind', 'status'], 'PASS'),
                              ('reported_failure', ['failures'], ['failure']),
                              ('atom_omitted', ['ATOMS', 'atoms'], []),
                              ('atom_duplicate', ['ATOMS', 'atoms'], sketch_report['ATOMS']['atoms']*2)]:
        altered = copy.deepcopy(sketch_report); parent = altered
        for key in path[:-1]: parent = parent[key]
        parent[path[-1]] = value
        checks['sketch_'+name+'_refused'] = not own_checks(altered, sketch_item)
    projection = {'timing_paths': [['timing_ms', 'tangentcirkelkedja', 'median']]}
    projected, _ = normalized_report(sketch_report, projection)
    checks['sketch_clock_projection_preserves_repetitions'] = projected['timing_ms']['tangentcirkelkedja']=={'n_reps': 15}
    try: normalized_report({'geometry': {'median': 1}}, {'timing_paths': [['geometry', 'median']]})
    except ValueError: checks['sketch_geometry_median_not_clock'] = True
    else: checks['sketch_geometry_median_not_clock'] = False
    with tempfile.TemporaryDirectory(prefix='field-self-reference-control-') as directory:
        root = Path(directory); (root/'artifacts').mkdir(); path = root/'artifacts/report.json'
        item = {'report_artifact': 'artifacts/report.json', 'self_artifact_paths': [['ATOMS', 'atoms', 0, 'artifact']]}
        report = {'ATOMS': {'atoms': [{'artifact': str(path)}]}}
        path.write_text(json.dumps(report))
        checks['artifact_self_reference_valid'] = read_report('', item, root)==report
        report['ATOMS']['atoms'][0]['artifact'] = str(root/'elsewhere/report.json')
        path.write_text(json.dumps(report))
        try: read_report('', item, root)
        except ValueError: checks['artifact_wrong_same_basename_refused'] = True
        else: checks['artifact_wrong_same_basename_refused'] = False
    nested_item = {'kind': 'dict', 'checks_path': ['parts'], 'checks': ['one'], 'case_gate': 'PASS'}
    checks['nested_case_valid'] = own_checks({'all_pass': True, 'parts': {'one': {'PASS': True}}}, nested_item)
    checks['nested_case_false_refused'] = not own_checks({'all_pass': True, 'parts': {'one': {'PASS': False}}}, nested_item)
    with tempfile.TemporaryDirectory(prefix='field-recipe-array-control-') as directory:
        root = Path(directory); folder = root/'src/field_engine/artifacts'; folder.mkdir(parents=True)
        output = root/'out'; output.mkdir(); path = folder/'density.npy'
        import numpy as np
        np.save(path, np.array([1., 2., 3.], dtype=np.float64))
        data = path.read_bytes()
        item = {'module': 'probe.py', 'captures': [{'path': 'src/field_engine/artifacts/density.npy',
                  'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}]}
        rows = retain_captures(root, item, output)
        checks['complete_array_valid'] = len(rows)==1 and rows[0]['exact'] and (output/'probe.py.density.npy').read_bytes()==data
        np.save(path, np.array([1., 2., 4.], dtype=np.float64))
        rows = retain_captures(root, item, output)
        checks['complete_array_wrong_value_refused'] = len(rows)==1 and not rows[0]['exact']
        checks['complete_array_failure_retained'] = (output/'probe.py.density.npy').read_bytes()==path.read_bytes()
        path.unlink(); rows = retain_captures(root, item, output)
        checks['complete_array_missing_refused'] = not rows[0]['exact'] and bool(rows[0]['error'])
    # Execute a real probe against an intentionally wrong documented number.
    # This is a verifier falsifier; the numerical probe and its arrays are untouched.
    with tempfile.TemporaryDirectory(prefix='field-retention-control-') as directory:
        scratch = Path(directory)
        shutil.copytree(ROOT/'src', scratch/'src', ignore=shutil.ignore_patterns('__pycache__', 'artifacts'))
        shutil.copytree(ROOT/'probes', scratch/'probes', ignore=shutil.ignore_patterns('__pycache__', 'artifacts'))
        (scratch/'scripts').mkdir(); (scratch/'docs').mkdir(); (scratch/'reports').mkdir()
        shutil.copy2(ROOT/'scripts/verify_csg.py', scratch/'scripts/verify_csg.py')
        shutil.copy2(ROOT/'scripts/field_paths.py', scratch/'scripts/field_paths.py')
        specification = contract(text, 'csg-verify-contract:v1')
        specification['probes'][0]['decisive'][0]['value'] += 1
        marker = r'(<!-- csg-verify-contract:v1 -->\s*```json\s*)(.*?)(\s*```)'
        altered = re.sub(marker, lambda m: m[1]+json.dumps(specification)+m[3], text, flags=re.S)
        (scratch/'docs/RUNNING.md').write_text(altered)
        for suffix in ('.json', '_arrays.npz'):
            shutil.copy2(ROOT/'reports'/('field_csg'+suffix), scratch/'reports'/('field_csg'+suffix))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        child = subprocess.run([sys.executable, str(scratch/'scripts/verify_csg.py'), '--output', str(scratch/'fresh')],
                               cwd=scratch, env=env, capture_output=True, text=True, timeout=60)
        receipt_path = scratch/'fresh/verification.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        checks['wrong_number_child_fails'] = child.returncode == 1 and receipt.get('gates', {}).get('documented_numbers') is False
        actual = scratch/'fresh/field_csg_arrays.npz'
        checks['failed_capture_retained'] = actual.exists() and compare_arrays(ROOT/'reports/field_csg_arrays.npz', actual) == (True, 1801776)
        checks['failed_report_retained'] = (scratch/'fresh/field_csg.json').exists() and (scratch/'fresh/field_csg.log').exists()
    # A first command overwrites its copy of a pinned reference. A second command
    # must still receive the original resource, including in selected execution.
    with tempfile.TemporaryDirectory(prefix='field-command-isolation-control-') as directory:
        scratch = Path(directory)
        for folder in ('src/field_engine', 'scripts', 'docs', 'reports', 'examples'):
            (scratch/folder).mkdir(parents=True, exist_ok=True)
        for name in ('verify_geometry.py', 'verify.py', 'verify_csg.py', 'field_paths.py'):
            shutil.copy2(ROOT/'scripts'/name, scratch/'scripts'/name)
        resource = scratch/'reports/resource.json'; resource.write_text('0')
        specification = {'version': 1, 'source_sha256': {},
                         'resource_sha256': {'reports/resource.json': hashlib.sha256(b'0').hexdigest()}, 'probes': []}
        for name, mutation in [('first.py', True), ('second.py', False)]:
            source = "import json\nfrom pathlib import Path\nroot=Path(__file__).resolve().parents[2]\nvalue=int((root/'reports/resource.json').read_text())\n"
            if mutation:
                source += "(root/'reports/resource.json').write_text('19')\n"
            source += "result={'gates':{'ok':value==0},'resource_before':value}\n"
            source += "(root/'reports/"+name+".json').write_text(json.dumps(result,sort_keys=True)+'\\n')\n"
            path = scratch/'src/field_engine'/name; path.write_text(source)
            specification['source_sha256'][str(path.relative_to(scratch))] = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = json.dumps({'gates': {'ok': True}, 'resource_before': 0}, sort_keys=True)+'\n'
            specification['probes'].append({'module': name, 'command': name, 'report': 'reports/'+name+'.json',
                'gates': ['ok'], 'decisive': [{'path': ['resource_before'], 'value': 0}], 'captures': [],
                'report_sha256': hashlib.sha256(expected.encode()).hexdigest()})
        (scratch/'docs/RUNNING.md').write_text('<!-- geometry-verify-contract:v1 -->\n```json\n'+json.dumps(specification)+'\n```\n')
        for selected in (False, True):
            output = scratch/('selected' if selected else 'all')
            command = [sys.executable, str(scratch/'scripts/verify_geometry.py'), '--output', str(output)]
            if selected: command += ['--module', 'second.py']
            child = subprocess.run(command, cwd=scratch, capture_output=True, text=True, timeout=30)
            path = output/'verification.json'; receipt = json.loads(path.read_text()) if path.exists() else {}
            passed = child.returncode==0 and receipt.get('status')=='VERIFIED-FRESH'
            if selected:
                checks['selected_omissions_explicit'] = passed and receipt.get('selected_modules')==['second.py'] and receipt.get('omitted_modules')==['first.py']
            else:
                checks['per_command_reference_isolation'] = passed and len(receipt.get('rows', []))==2
        checks['original_resource_unchanged'] = resource.read_bytes()==b'0'
    with tempfile.TemporaryDirectory(prefix='field-recipe-isolation-control-') as directory:
        scratch = Path(directory)
        for folder in ('src/field_engine/recipe/artifacts', 'scripts', 'docs', 'examples'):
            (scratch/folder).mkdir(parents=True, exist_ok=True)
        for name in ('verify_recipe_selftests.py', 'verify.py', 'verify_csg.py', 'field_paths.py'):
            shutil.copy2(ROOT/'scripts'/name, scratch/'scripts'/name)
        (scratch/'src/field_engine/recipe/artifacts/stale').write_text('seeded')
        specification = {'version': 1, 'source_sha256': {}, 'probes': []}
        expected = json.dumps({'all_pass': True, 'clean': {'pass': True}}, sort_keys=True, indent=2)+'\n'
        for name in ('first.py', 'second.py'):
            source = "from pathlib import Path\nimport json\np=Path(__file__).parent/'artifacts'\nclean=p.is_dir() and not any(p.iterdir())\n(p/'stale').write_text('mutated')\nprint(json.dumps({'all_pass':clean,'clean':{'pass':clean}}))\n"
            path = scratch/'src/field_engine/recipe'/name; path.write_text(source)
            specification['source_sha256'][str(path.relative_to(scratch))] = hashlib.sha256(path.read_bytes()).hexdigest()
            specification['probes'].append({'module': 'recipe/'+name, 'args': [], 'kind': 'dict', 'checks': ['clean'],
                'decisive': [], 'report_sha256': hashlib.sha256(expected.encode()).hexdigest()})
        (scratch/'docs/RUNNING.md').write_text('<!-- recipe-verify-contract:v1 -->\n```json\n'+json.dumps(specification)+'\n```\n')
        child = subprocess.run([sys.executable, str(scratch/'scripts/verify_recipe_selftests.py'), '--output', str(scratch/'out')],
                               cwd=scratch, capture_output=True, text=True, timeout=30)
        receipt = json.loads((scratch/'out/verification.json').read_text())
        checks['recipe_fresh_empty_artifacts_per_command'] = child.returncode==0 and len(receipt['rows'])==2 and all(receipt['gates'].values())
        checks['recipe_parent_artifact_unchanged'] = (scratch/'src/field_engine/recipe/artifacts/stale').read_text()=='seeded'
        first = scratch/'src/field_engine/recipe/first.py'
        first.write_text(first.read_text().replace('clean=p.is_dir() and not any(p.iterdir())', 'clean=False'))
        specification['source_sha256']['src/field_engine/recipe/first.py'] = hashlib.sha256(first.read_bytes()).hexdigest()
        (scratch/'docs/RUNNING.md').write_text('<!-- recipe-verify-contract:v1 -->\n```json\n'+json.dumps(specification)+'\n```\n')
        child = subprocess.run([sys.executable, str(scratch/'scripts/verify_recipe_selftests.py'), '--output', str(scratch/'failed_first')],
                               cwd=scratch, capture_output=True, text=True, timeout=30)
        receipt = json.loads((scratch/'failed_first/verification.json').read_text())
        checks['recipe_failure_does_not_skip_independent_command'] = (child.returncode==1 and len(receipt['rows'])==2
            and receipt['gates']['coverage'] and not receipt['rows'][0]['gates'] and receipt['rows'][1]['gates']
            and receipt['status']=='OWN-GATE-FAIL')

    print(json.dumps(dict(status='VERIFIED-FRESH' if all(checks.values()) else 'OWN-GATE-FAIL',
                          controls=checks, scope='Coverage/parser/child receipt controls plus real wrong-number falsifier and retained complete outputs; no GPU initialization.'), indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
