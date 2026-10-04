"""Malformed imported records must be refused without weakening exact replay."""
from fractions import Fraction as Q
import pytest
from field_engine.experimental.hyperelastic_energy import scalar_certificate, verify as verify_energy
from field_engine.experimental.continuum_stability import rod_certificate, verify as verify_rod


@pytest.mark.parametrize('record', [None, [], 'CERTIFIED', 1, True])
def test_energy_rejects_non_dictionary_records(record):
    assert verify_energy(record) is False


@pytest.mark.parametrize('key', [1, None, (1, 0, 0)])
def test_rod_rejects_non_string_polynomial_keys(key):
    inputs = dict(t='4/5', length='10', field=[{key: '1'}, {}, {}])
    with pytest.raises(ValueError):
        rod_certificate(**inputs)
    assert verify_rod({'inputs': inputs}) is False


def test_public_two_bar_scope_and_positive_energy_certificate():
    cert = scalar_certificate('arch', '0.18553911', '1/100000', '1/1000')
    assert cert['scope'] == 'TWO_BAR_CONSTRAINED_APEX'
    assert cert['status'] == 'CERTIFIED'
    assert Q(cert['energy_gap_upper']) > 0
    assert verify_energy(cert)
