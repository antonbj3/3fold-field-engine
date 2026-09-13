"""Reproduce the captured float64 power mismatch without running the optimizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('Evidence directory must be empty')
    manifest = json.loads((ROOT/'reports/power_dispatch_fixture.json').read_text())
    fixture = ROOT/'reports/power_dispatch_fixture.npz'
    fixture_valid = hashlib.sha256(fixture.read_bytes()).hexdigest()==manifest['fixture_sha256']
    names = {'lastfalt_v1_topopt.py', 'lastfalt_v1_fem.py'}
    source_valid = set(manifest['source_sha256'])==names and all(
        hashlib.sha256((ROOT/'src/field_engine'/name).read_bytes()).hexdigest()==expected
        for name, expected in manifest['source_sha256'].items())
    if not fixture_valid or not source_valid:
        raise ValueError('Frozen fixture or numerical source hash mismatch')
    with np.load(fixture, allow_pickle=False) as archive:
        if set(archive.files)!={'density', 'exponent', 'reference', 'native'}:
            raise ValueError('Unexpected fixture arrays')
        arrays = {name: archive[name].copy() for name in archive.files}
    if any(a.dtype!=np.float64 or not np.all(np.isfinite(a)) for a in arrays.values()):
        raise ValueError('Expected finite float64 fixture')
    if arrays['exponent'].shape!=() or float(arrays['exponent'])!=3.0 or any(
            arrays[name].shape!=(5120,) for name in ('density', 'reference', 'native')):
        raise ValueError('Unexpected fixture shape or exponent')
    density = arrays['density']
    before = density.tobytes()
    actual = np.power(density, float(arrays['exponent']))
    np.save(output/'actual.npy', actual)
    reference_exact = actual.tobytes()==arrays['reference'].tobytes()
    native_exact = actual.tobytes()==arrays['native'].tobytes()
    gates = dict(fixture_integrity=fixture_valid, source_integrity=source_valid,
                 input_preserved=density.tobytes()==before,
                 known_result=reference_exact or native_exact, reference_exact=reference_exact)
    result = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL', gates=gates,
                  differing_values=int(np.count_nonzero(actual!=arrays['reference'])),
                  max_abs_difference=float(np.max(np.abs(actual-arrays['reference']))),
                  matches_known_native=native_exact, actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                  numpy_version=np.__version__, numpy_core_sha256=hashlib.sha256(Path(np._core._multiarray_umath.__file__).read_bytes()).hexdigest(),
                  requested_numpy_disabled=os.environ.get('NPY_DISABLE_CPU_FEATURES', ''),
                  enabled_dispatch={name: bool(np._core._multiarray_umath.__cpu_features__.get(name))
                                    for name in np._core._multiarray_umath.__cpu_dispatch__},
                  scope='Complete captured second-iteration power operation only. Native mismatch remains a failed reference comparison; no optimizer, geometry or accuracy claim inferred.')
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
