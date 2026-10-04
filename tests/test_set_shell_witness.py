from fractions import Fraction as Q
import pytest
from field_engine.experimental import set_shell_witness as s


T = [([-1,-1], [1,1])]
P = [([0,0],[0,0])]


def test_shape_independent_obstruction():
    r = s.verify_shell_witness(T,P,'4/5','2/5',['4/5',0],['11/10',0],unit='mm')
    assert r.status == 'INFEASIBLE'
    assert r.protected_distance_squared == Q(16,25)
    assert r.shell_distance_squared == Q(9,100)
    assert r.unit == 'mm'


def test_source_boundary_and_near_outside():
    r = s.verify_shell_witness(T,P,1,1,[1,0],[1,0],unit='m')
    assert r.status == 'UNKNOWN'
    r = s.verify_shell_witness(T,P,1,'1/1000000000000',[1,0],['1.000000000001',0],unit='m')
    assert r.status == 'INFEASIBLE'


def test_other_source_box_covers_outside_point():
    r = s.verify_shell_witness(T+[([1,-1],[2,1])],P,1,1,[1,0],['1.1',0],unit='mm')
    assert r.status == 'UNKNOWN'


def test_empty_protected_and_invalid_membership():
    assert s.verify_shell_witness(T,[],1,1,[1,0],[2,0],unit='m').status == 'UNKNOWN'
    assert s.verify_shell_witness(T,P,0,1,[1,0],[2,0],unit='m').status == 'UNKNOWN'
    assert s.verify_shell_witness(T,P,1,'1/100',[1,0],[2,0],unit='m').status == 'UNKNOWN'


def test_zero_shell_direct_obstruction():
    assert s.verify_shell_witness([],P,0,0,[0,0],[0,0],unit='m').status == 'INFEASIBLE'


def test_sdf_margin_is_arithmetic_without_a_proof_flag():
    assert s.sdf_shell_margin(['-1/10','1/10'],0) == (Q(-1,10),Q(1,10))
    assert s.sdf_shell_margin(['-1/10','-1/20'],'1/10') == (Q(0),Q(1,20))
    assert s.sdf_shell_margin([-2,-1],1) == (Q(-1),Q(0))


def test_box_shell_recomputes_full_set_condition():
    result=s.verify_box_shell(T[0],([-Q(3,4),-Q(1,2)],[Q(3,4),Q(1,2)]),'1/4',unit='mm')
    assert result.status == 'CONTAINED'
    assert result.extent_margin == 0
    result=s.verify_box_shell(T[0],([-Q(3,4),-Q(1,2)],[Q(3,4),Q(1,2)]),'1/2',unit='mm')
    assert result.status == 'INFEASIBLE'
    assert result.extent_margin == Q(1,4)


@pytest.mark.parametrize('radius,width', [(-1,0),(0,-1)])
def test_negative_radii(radius,width):
    with pytest.raises(ValueError):
        s.verify_shell_witness(T,P,radius,width,[0,0],[2,0],unit='mm')
