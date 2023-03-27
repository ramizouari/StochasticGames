import pytest
from ..games import mpg
from . import wrapper
from ..graph import random_graph

@pytest.fixture(scope='function', params=[[(0,0,0,-1)],
                                    [(0,0,0,0)],
                                    [(0,0,0,1)],
                                    [(0,1,1,-1),(1,0,0,-1)],])
def ternary_max_constraints(request):
    system=wrapper.TernaryMaxAtomSystem()
    for x,y,z,w in request.param:
        system.add_constraint(x,y,z,w)
    return system

def test_solve_max_atom(ternary_max_constraints):
    result=wrapper.solve_ternary_max_atom_system(ternary_max_constraints,as_dict=True)
    for x,y,z,w in ternary_max_constraints.constraints:
        assert result[x]<=result[y]+result[z]+w


def test_solve_max_atom_maximality(ternary_max_constraints):
    result=wrapper.solve_ternary_max_atom_system(ternary_max_constraints,as_dict=True)
    fixedVariable=next(iter(ternary_max_constraints.variables))
    for x in ternary_max_constraints.variables:
        if x==fixedVariable:
            continue
        oldX=result[x]
        result[x]+=1
        if result[x]==oldX:
            continue
        satisfied=True
        for x,y,z,w in ternary_max_constraints.constraints:
            if result[x]>result[y]+result[z]+w:
                satisfied=False
                break
        assert not satisfied
        result[x]-=1

# This test case should be guaranteed to have a unique optimal strategy pair
@pytest.fixture(scope='function', params=range(5))
def mpg_instance(request):
    k=request.param
    if k==0:
        game=wrapper.MeanPayoffGraph()
        game.add_weighted_edges_from([(1, 2, 5),
                                      (2, 3, -7),
                                      (3, 7, 0),
                                      (3, 6, 5),
                                      (6, 1, -3),
                                      (1, 4, 4),
                                      (4, 5, -3),
                                      (5, 6, 3),
                                      (5, 7, 0),
                                      (7, 1, 0),
                                      (0, 1, 5)])
    else:
        game= random_graph.gnm_random_mpg(n=10,m=20,seed=k,method="fast",loops=True,distribution="integers",low=-7,high=8)
    return game
def test_wrapper_winners(mpg_instance):
    result=wrapper.winners(mpg_instance,as_dict=True)
    S1,S2=mpg.optimal_strategy_pair(mpg_instance)
    intermediateResult=mpg.winners(mpg_instance,S1,S2)
    expectedResult=({},{})
    for u,turn in intermediateResult:
        expectedResult[turn][u]=not intermediateResult[(u,turn)]
    assert result==expectedResult


def test_wrapper_winners_list(mpg_instance):
    result=wrapper.winners(mpg_instance,as_dict=False)
    S1,S2=mpg.optimal_strategy_pair(mpg_instance)
    intermediateResult=mpg.winners(mpg_instance,S1,S2)
    n=len(mpg_instance)
    expectedResult=([0]*n,[0]*n)
    for u,turn in intermediateResult:
        expectedResult[turn][u]=not intermediateResult[(u,turn)]
    assert result==expectedResult

def test_wrapepr_optimal_strategy_pair(mpg_instance):
    result=wrapper.optimal_strategy_pair(mpg_instance,as_dict=True)
    expectedResult=mpg.optimal_strategy_pair(mpg_instance)

    assert result==expectedResult

def test_wrapper_optimal_strategy_pair_list(mpg_instance):
    result=wrapper.optimal_strategy_pair(mpg_instance,as_dict=False)
    intermediateResult=mpg.optimal_strategy_pair(mpg_instance)
    n=len(mpg_instance)
    expectedResult=([0]*n,[0]*n)
    for E,I in zip(expectedResult,intermediateResult):
        for u in I:
            E[u]=I[u]
    assert result==expectedResult

def test_wrapper_mean_payoffs(mpg_instance):
    S1,S2=mpg.optimal_strategy_pair(mpg_instance)
    result=wrapper.mean_payoffs(mpg_instance,as_dict=True)
    intermediate=mpg.mean_payoffs(mpg_instance,S1,S2)
    expectedResult=({},{})
    for u,turn in intermediate:
        expectedResult[turn][u]=intermediate[(u,turn)]
    assert result==pytest.approx(expectedResult)

def test_wrapper_mean_payoffs_list(mpg_instance):
    S1,S2=mpg.optimal_strategy_pair(mpg_instance)
    result=wrapper.mean_payoffs(mpg_instance,as_dict=False)
    intermediate=mpg.mean_payoffs(mpg_instance,S1,S2)
    n=len(mpg_instance)
    expectedResult=([0]*n,[0]*n)
    for u,turn in intermediate:
        expectedResult[turn][u]=intermediate[(u,turn)]
    assert result==pytest.approx(expectedResult)