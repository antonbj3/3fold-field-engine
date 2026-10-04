from fractions import Fraction as Q
import itertools
import pytest
from field_engine.conic_witness_v1 import check_conic_witness as conic, check_strict_support as support

@pytest.fixture(autouse=True)
def require_live_bidirectional_verifiers():
    # Pair acceptance and rejection for each checker in every regression.
    # A negative-only test must never pass when the verifier is disabled.
    assert conic([[0]], [-1], [('nonnegative', 1)], [1]).status == 'INFEASIBLE'
    assert conic([[1]], [1], [('nonnegative', 1)], [1]).status == 'UNKNOWN'
    assert support([[1], [-1]], [0, 0], ['1/2', '1/2']).status == 'NO_STRICT_FIT'
    assert support([[1], [-1]], [1, 1], ['1/2', '1/2']).status == 'UNKNOWN'

I3 = [[1,0,0],[0,1,0],[0,0,1]]

def test_friction_separator_and_changed_rhs():
    bad = conic(I3,[1,2,0],[('soc',3)],[1,-1,0])
    feasible = conic(I3,[1,'1/2',0],[('soc',3)],[1,-1,0])
    assert bad.status == 'INFEASIBLE' and bad.margin == 1
    assert feasible.status == 'UNKNOWN' and bad.model_sha256 != feasible.model_sha256

@pytest.mark.parametrize('y', [[-1,0,0],[1,2,0],[0,0,0]])
def test_soc_sign_norm_and_zero_guard(y):
    assert conic(I3,[1,2,0],[('soc',3)],y).status == 'UNKNOWN'

def test_lattice_moment_source():
    velocities = [v for v in itertools.product([-1,0,1],repeat=3) if sum(x*x for x in v)<=2]
    assert len(velocities)==19
    A = [[1]*19,[v[2]**2 for v in velocities]]
    c = conic(A,['35/2','301/12'],[('nonnegative',19)],[1,-1])
    assert c.status == 'INFEASIBLE' and c.margin == Q(91,12)
    assert conic(A,['35/2','35/2'],[('nonnegative',19)],[1,-1]).status=='UNKNOWN'

def test_free_dual_requires_exact_zero():
    assert conic([[1,1],[1,-1]],[-1,-1],[('free',1),('nonnegative',1)],[1,1]).status=='UNKNOWN'
    assert conic([[1,1],[-1,1]],[-1,-1],[('free',1),('nonnegative',1)],[1,1]).status=='INFEASIBLE'

def test_weak_infeasibility_has_no_single_strict_witness():
    # t=z, u=1 in SOC(t,u,z), equivalent to rotated cone a=0,b=1.
    A = [[1,0,-1],[0,1,0]]
    for y in itertools.product(range(-4,5),repeat=2):
        assert conic(A,[0,1],[('soc',3)],y).status=='UNKNOWN'


def test_equal_projections_are_not_closed_infeasibility():
    r = support([[1,0],[-1,0]],[0,0],['1/2','1/2'])
    assert r.status == 'NO_STRICT_FIT' and r.margin == 0
    assert support([[1,0],[-1,0]],[1,1],['1/2','1/2']).status=='UNKNOWN'
    assert support([[1,0],[-1,0]],[-1,-1],[0,0]).status=='UNKNOWN'


def test_three_supports_and_invalid_balance():
    assert support([[1,0],[0,1],[-1,-1]],[-1,-1,-1],['1/3']*3).status=='NO_STRICT_FIT'
    assert support([[1,0],[0,1],[-1,-1]],[-1,-1,-1],[1,0,0]).status=='UNKNOWN'

@pytest.mark.parametrize('bad', [0.0,True,float('nan')])
def test_reject_inexact_and_boolean_inputs(bad):
    with pytest.raises(TypeError):conic([[1]],[bad],[('nonnegative',1)],[1])

@pytest.mark.parametrize('A,b,blocks,y', [([],[],[],[]),([[1],[1,2]],[1,2],[('free',1)],[1,1]),([[1]],[1],[('soc',1)],[1]),([[1]],[1],[('nonnegative',2)],[1]),([[1]],[1],[('free',1)],[]),([[1]],[1],[('unknown',1)],[1])])
def test_malformed_models(A,b,blocks,y):
    with pytest.raises(ValueError):conic(A,b,blocks,y)


def test_exhaustive_constructed_feasible_orthant_models():
    A = [[1,2,-1],[0,1,2]]
    for x in itertools.product(range(3),repeat=3):
        b=[sum(a*z for a,z in zip(row,x)) for row in A]
        for y in itertools.product(range(-2,3),repeat=2):
            assert conic(A,b,[('nonnegative',3)],y).status!='INFEASIBLE'


def test_review_sub_float_soc_gap_and_forged_membership():
    tiny = Q(1, 10**30)
    valid = conic(I3, [1, 1 + tiny, 0], [('soc', 3)], [1, -1, 0])
    assert valid.status == 'INFEASIBLE' and valid.margin == tiny
    assert conic(I3, [1, 0, 0], [('soc', 3)], [1, 1 + tiny, 0]).status == 'UNKNOWN'


def test_review_degenerate_zero_image_and_empty_slice():
    assert conic([[0]], [-1], [('nonnegative', 1)], [1]).status == 'INFEASIBLE'
    assert conic([[0, 0, 0]], [-1], [('soc', 3)], [1]).status == 'INFEASIBLE'
    assert conic([[0, 0, 0]], [0], [('soc', 3)], [1]).status == 'UNKNOWN'


def test_review_boolean_block_dimension_is_rejected():
    with pytest.raises(ValueError):
        conic([[1]], [-1], [('nonnegative', True)], [1])


def test_review_zero_support_normal_proves_only_declared_strict_bound():
    assert support([[0]], [-1], [1]).status == 'NO_STRICT_FIT'
    assert support([[0]], [1], [1]).status == 'UNKNOWN'
