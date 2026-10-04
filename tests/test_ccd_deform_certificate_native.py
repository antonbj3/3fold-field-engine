"""Compile and exercise the optional CPU geometric certificate (no CUDA required)."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest

@pytest.mark.parametrize('source', ['test_ccd_deform_certificate.cpp', 'test_ccd_deform_transport.cpp', 'test_ccd_deform_reserve.cpp'])
def test_native_certificate(tmp_path, source):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('optional native certificate requires g++')
    eigen = Path(os.environ.get('EIGEN3_INCLUDE_DIR', '/usr/include/eigen3'))
    if not (eigen / 'Eigen/Core').is_file():
        pytest.skip('optional native certificate requires Eigen3 headers')
    root = Path(__file__).resolve().parents[1]
    target = tmp_path / 'regression'
    subprocess.run([compiler, '-O2', '-std=c++17', '-ffp-contract=off', f'-I{eigen}',
                    f'-I{root / "src/field_engine/experimental/native"}',
                    str(root / 'tests/native' / source), '-o', str(target)], check=True, timeout=60)
    result = subprocess.run([str(target)], check=True, capture_output=True, text=True, timeout=20)
    assert 'regression groups passed' in result.stdout
