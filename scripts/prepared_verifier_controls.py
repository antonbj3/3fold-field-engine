"""Exercise verifier refusal of incomplete gates and corrupted build evidence."""
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
from build_prepared_service import ARTIFACTS, ROOT, digest
from verify_prepared_service import build_environment, check_report
from verify_csg import compare_arrays


def main():
    item = dict(checks='gates', count=2, decisive={'cases': 3})
    saved = dict(gates={'exact': True, 'repeat': True}, cases=3)
    controls = {'valid_report': check_report(saved, saved, item)}
    for name, change in [('false_gate', {'gates': {'exact': False, 'repeat': True}}),
                         ('missing_gate', {'gates': {'exact': True}}),
                         ('empty_gates', {'gates': {}}), ('nonboolean_gate', {'gates': {'exact': 1, 'repeat': True}}),
                         ('wrong_number', {'cases': 4})]:
        actual = copy.deepcopy(saved); actual.update(change)
        controls[name] = not check_report(actual, saved, item)
    with tempfile.TemporaryDirectory(prefix='field-prepared-controls-') as folder:
        folder = Path(folder)
        a, b = folder/'a.npz', folder/'b.npz'
        np.savez(a, x=np.array([0., -0.], np.float32))
        np.savez(b, x=np.array([0., 0.], np.float32))
        controls['signed_zero_bytes'] = not compare_arrays(a, b)[0]
        np.savez(b, x=np.array([0., -0.], np.float64))
        controls['wrong_dtype'] = not compare_arrays(a, b)[0]
        source = ROOT/'src/field_engine/rt_columns_prepared_v1/params.h'
        manifest = dict(version=1, artifacts=ARTIFACTS,
                        source_sha256={str(source.relative_to(ROOT)): digest(source)}, artifact_sha256={})
        for name in ARTIFACTS.values():
            (folder/name).write_bytes(b'controlled artifact')
            manifest['artifact_sha256'][name] = digest(folder/name)
        def write(value):
            (folder/'manifest.json').write_text(json.dumps(value))
        write(manifest)
        controls['valid_manifest'] = set(build_environment(folder)) == set(ARTIFACTS)
        for name, change in [('stale_source', {'source_sha256': {str(source.relative_to(ROOT)): '0'*64}}),
                             ('empty_sources', {'source_sha256': {}}),
                             ('missing_artifact', {'artifacts': {}})]:
            altered = copy.deepcopy(manifest); altered.update(change); write(altered)
            try: build_environment(folder)
            except ValueError: controls[name] = True
            else: controls[name] = False
        write(manifest)
        (folder/'prepared.ptx').write_bytes(b'changed')
        try: build_environment(folder)
        except ValueError: controls['corrupt_artifact'] = True
        else: controls['corrupt_artifact'] = False
    result = dict(status='VERIFIED-FRESH' if all(controls.values()) else 'OWN-GATE-FAIL', controls=controls,
                  scope='Verifier controls only; synthetic artifacts are never loaded or executed.')
    print(json.dumps(result, indent=2))
    return 0 if all(controls.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
