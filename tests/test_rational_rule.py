from copy import deepcopy
from fractions import Fraction as Q
import pytest
from field_engine.experimental import rational_rule as r


def matrix(constant):
    return [[(constant,constant),(0,0)],[(0,0),(0,0)]]


def fixture():
    models = {'a':{0:{'components':[matrix(1)],'guards':[]}},
              'b':{0:{'components':[matrix(4)],'guards':[]}}}
    cell = {'box':[[0,1]],'potential_masks':{'a':[0],'b':[0]},
            'pairs':[{'selected_mask':0,'other_mask':0,'other_anchor':0,
                      'inequalities':[{'selected_component':0,'selected_norm_upper':1,
                                       'difference_squared_lower':3,'rhs_upper':'3/4'}]}]}
    return models, [cell]


def replay(models,cells):
    return r.replay_rule([[0,1]],cells,models,selected='a',other='b',margin='1/4')


def test_full_rule_and_exact_counts():
    models,cells=fixture();out=replay(models,cells)
    assert out.status == 'CERTIFIED_SURROGATE_RULE'
    assert out.cells == 1
    assert out.inequalities == 1


def test_partition_gaps_overlaps_and_duplicates():
    assert r.exact_partition([[0,1]],[[[0,'1/2']],[['1/2',1]]]) == True
    assert r.exact_partition([[0,1]],[[[0,'1/2']],[['3/4',1]]]) == False
    assert r.exact_partition([[0,1]],[[[0,'3/4']],[['1/4','1/2']]]) == False
    assert r.exact_partition([[0,1]],[[[0,1]],[[0,1]]]) == False
    assert r.exact_partition([[0,1]],[]) == False


def test_missing_branch_pair_and_empty_family():
    models,cells=fixture();models['a'][1]=deepcopy(models['a'][0])
    assert replay(models,cells).status == 'UNKNOWN'
    models,cells=fixture();cells[0]['pairs']=[]
    assert replay(models,cells).status == 'UNKNOWN'
    models,cells=fixture();models['b']={}
    assert replay(models,cells).status == 'UNKNOWN'


def test_omitted_component_and_forged_bound():
    models,cells=fixture();models['a'][0]['components'].append(matrix(1))
    assert replay(models,cells).status == 'UNKNOWN'
    models,cells=fixture();cells[0]['pairs'][0]['inequalities'][0]['difference_squared_lower']=4
    assert replay(models,cells).status == 'UNKNOWN'
    models,cells=fixture();cells[0]['pairs'][0]['inequalities'][0]['selected_norm_upper']='1/2'
    assert replay(models,cells).status == 'UNKNOWN'
    models,cells=fixture();cells[0]['pairs'][0]['inequalities'][0]['rhs_upper']='1/4'
    assert replay(models,cells).status == 'UNKNOWN'


def test_signed_anchor_guard():
    models,cells=fixture();models['b'][0]['anchor_guards']={0:[[(-1,-1),(0,0)]]}
    assert replay(models,cells).status == 'UNKNOWN'


def test_branch_exclusion_is_strict():
    models,cells=fixture();models['a'][1]={'components':[matrix(1)],'guards':[[(1,1),(0,0)]]}
    assert replay(models,cells).status == 'CERTIFIED_SURROGATE_RULE'
    models['a'][1]['guards']=[[(0,0),(0,0)]]
    assert replay(models,cells).status == 'UNKNOWN'


def test_quadratic_interior_extremum_invalidates_corners():
    # q=(x-1/2)^2 has zero interior value but both corner values 1/4.
    q=[[(Q(1,4),Q(1,4)),(-Q(1,2),-Q(1,2))],
       [(-Q(1,2),-Q(1,2)),(1,1)]]
    low,high=r.quadratic_range(q,r.quadratic_weights([[0,1]]))
    assert low <= 0
    assert high >= Q(1,4)
    assert r.affine_range([('1/3','1/3'),(2,2)],[[0,1]]) == (Q(1,3),Q(7,3))


def test_invalid_boxes():
    with pytest.raises(ValueError):
        r.quadratic_weights([[1,0]])


def test_negative_squared_surrogates_cannot_create_preference():
    models,cells=fixture()
    models['a'][0]['components']=[matrix(-100)]
    models['b'][0]['components']=[matrix(-99)]
    cells[0]['pairs'][0]['inequalities']=[{'selected_component':0,'selected_norm_upper':0,
                                         'difference_squared_lower':1,'rhs_upper':'1/16'}]
    assert replay(models,cells).status == 'UNKNOWN'


def test_disjoint_branch_domains_only_certify_conditional_inequalities():
    models,cells=fixture()
    models['a'][0]['guards']=[[(Q(3,4),Q(3,4)),(-1,-1)]]
    models['b'][0]['guards']=[[(Q(-1,4),Q(-1,4)),(1,1)]]
    out=replay(models,cells)
    assert out.status == 'CERTIFIED_BRANCH_CONDITIONAL_RULE'
    assert out.domain_coverage == 'UNKNOWN'
    assert out.covered_cells == 0
    assert out.inequalities == 1


def test_verified_branch_cover_includes_closed_boundary():
    models,cells=fixture()
    models['a'][0]['guards']=[[(-1,-1),(1,1)]]
    out=replay(models,cells)
    assert out.status == 'CERTIFIED_SURROGATE_RULE'
    assert out.domain_coverage == 'CERTIFIED'
    assert out.covered_cells == 1
    # Coefficient uncertainty crossing the boundary cannot establish coverage.
    models['a'][0]['guards']=[[(-1,-1),(1,2)]]
    out=replay(models,cells)
    assert out.status == 'CERTIFIED_BRANCH_CONDITIONAL_RULE'
    assert out.domain_coverage == 'UNKNOWN'


def test_missing_branch_region_is_not_a_whole_box_rule():
    models,cells=fixture()
    models['a'][0]['guards']=[[(Q(-1,2),Q(-1,2)),(1,1)]]
    out=replay(models,cells)
    assert out.status == 'CERTIFIED_BRANCH_CONDITIONAL_RULE'
    assert out.domain_coverage == 'UNKNOWN'
