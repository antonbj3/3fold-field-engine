from fractions import Fraction as Q
import math
import pytest
from field_engine.experimental import projected_gap as g


U = [[0, 0, -5], [3, 0, -2], [0, 3, 1]]
L = [[1, -1, 0], [4, 2, 0], [1, 2, 0]]


def test_edge_intersection_extremum():
    r = g.triangle_gap(U, L, unit='mm')
    assert r.status == 'CERTIFIED'
    assert r.exact == Q(-4)
    assert r.witness_xy == (Q(1), Q(0))
    assert r.lower == -4.0
    assert r.upper == -4.0


def test_reversed_orientation_and_units():
    r = g.triangle_gap(list(reversed(U)), list(reversed(L)), unit='m')
    assert r.exact == Q(-4)
    assert r.unit == 'm'
    r = g.triangle_gap([[Q(x,1000) for x in p] for p in U],
                       [[Q(x,1000) for x in p] for p in L], unit='m')
    assert r.exact == Q(-1,250)


def test_tiny_separation_is_empty():
    u = [[0,0,0],[1,0,0],[0,1,0]]
    d = Q(1,10**12)
    l = [[1+d,0,1],[2+d,0,1],[1+d,1,1]]
    assert g.triangle_gap(u,l,unit='mm').status == 'EMPTY'


def test_near_parallel_projection_is_resolved():
    u = [[0,0,0],[1,0,0],[1,Q(1,10**30),0]]
    l = [[p[0],p[1],1] for p in u]
    assert g.triangle_gap(u,l,unit='mm').exact == -1


def test_closed_line_contact():
    u = [[0,0,2],[1,0,2],[0,1,2]]
    l = [[0,0,1],[1,0,1],[0,-1,1]]
    assert g.triangle_gap(u,l,unit='m').exact == 1


def test_degenerate_projection_and_surface_unknown():
    deg = [[0,0,0],[1,0,1],[2,0,2]]
    assert g.triangle_gap(deg,L,unit='m').status == 'UNKNOWN'
    assert g.surface_gap([U,deg],[L],unit='m').status == 'UNKNOWN'


def test_empty_surface_and_global_minimum():
    assert g.surface_gap([], [L], unit='mm').status == 'EMPTY'
    assert g.surface_gap([U], [L], unit='mm').exact == -4


def test_binary_and_decimal_semantics():
    assert g._q(.1) == Q(3602879701896397,36028797018963968)
    assert g._q('0.1') == Q(1,10)


def test_outward_rounding_nonrepresentable():
    lo,hi = g.outward_float(Q(1,10))
    assert Q(lo) < Q(1,10)
    assert Q(hi) > Q(1,10)
    assert math.nextafter(lo, math.inf) == hi


def test_outward_rounding_underflow_overflow():
    lo,hi = g.outward_float(Q(1,10**400))
    assert lo == 0.0
    assert hi == math.nextafter(0.,math.inf)
    lo,hi = g.outward_float(Q(10**400))
    assert lo == math.nextafter(math.inf,-math.inf)
    assert hi == math.inf
    lo,hi = g.outward_float(-Q(10**400))
    assert lo == -math.inf
    assert hi == math.nextafter(-math.inf,math.inf)


@pytest.mark.parametrize('unit', ['N','',None])
def test_invalid_unit(unit):
    with pytest.raises(ValueError):
        g.triangle_gap(U,L,unit=unit)


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),True])
def test_invalid_coordinate(bad):
    with pytest.raises(ValueError):
        g.triangle_gap([[bad,0,0],[1,0,0],[0,1,0]],L,unit='m')
