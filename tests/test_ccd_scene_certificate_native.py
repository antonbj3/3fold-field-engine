"""Compile and run exact scene guard regressions; requires C++17 + Eigen3.

Optional native prototype. Decline is never interpreted as a free-step decision.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class NativeSceneCertificate(unittest.TestCase):
    def test_exact_guard_acceptance_and_refusals(self):
        root = Path(__file__).resolve().parents[1]
        compiler = shutil.which('g++')
        eigen = Path(os.environ.get('EIGEN3_INCLUDE_DIR', '/usr/include/eigen3'))
        if compiler is None or not (eigen / 'Eigen/Core').is_file():
            self.skipTest('optional native guard needs C++17 compiler and Eigen3')
        # Explicit TMPDIR keeps build outputs in the invoking lane during review.
        with tempfile.TemporaryDirectory(prefix='ccd-scene-') as tmp:
            binary = Path(tmp) / 'guard-test'
            subprocess.run([compiler, '-O2', '-std=c++17', '-I'+str(eigen),
                            '-I'+str(root / 'src'),
                            str(root / 'tests/native/test_ccd_scene_certificate.cpp'),
                            '-o', str(binary)], check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn('checks 19', result.stdout)
