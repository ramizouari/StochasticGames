from typing import Dict, Any, Union, List, Tuple

import numpy as np

from. import mpgcpp
from ..csp.max_atom import MaxAtomSystem
from ..games import MeanPayoffGraph
from ..csp.max_atom import TernaryMaxAtomSystem


def _type_cxx_intermediate(edgetype):
    if edgetype is not None:
        match edgetype():
            case int() | np.int64() | np.int32():
                edgetype_ = int
            case float() | np.float64() | np.float32():
                edgetype_ = float
            case _:
                raise ValueError("Unknown edge type")
    return edgetype_

def solve_max_atom_csp(system: MaxAtomSystem,as_dict=True) -> Union[List[Any],Dict[str, Any]]:
    """
    Solves a max atom csp using the mpgcpp wrapper
    :param system: The max atom system to solve
    :param timeout: The timeout in seconds
    :return: A dictionary containing the solution
    """
    mapper = dict(zip(system.variables, range(len(system.variables))))
    system_cxx : List[Tuple[int,int,int,int]] = []
    for x,y,z,w in system.constraints:
        system_cxx.append((mapper[x],mapper[y],mapper[z],w))
    assignment=mpgcpp.arc_consistency_cxx(system_cxx,as_dict)
    if as_dict:
        return {system.variables[i]:assignment[i] for i in range(len(system.variables))}
    else:
        return assignment


def optimal_strategy_pair(game: Union[MeanPayoffGraph,str], as_dict=None,edgetype=None) -> Union[List[Any], Dict[str, Any]]:
    """
    Computes an optimal strategy pair for a mean payoff game
    :param game: The mean payoff game, either as a MeanPayoffGraph object or as a path to a file
    :param timeout: The timeout in seconds
    :return: A dictionary containing the solution
    """
    n:int
    match game:
        case MeanPayoffGraph():
            if as_dict is None:
                as_dict = True
            mapper = dict(zip(game.nodes, range(len(game.nodes))))
            game_cxx: List[Tuple[int, int, int]] = []
            n=len(game.nodes)
            edgetype_ = None
            if edgetype is not None:
                edgetype_ = _type_cxx_intermediate(edgetype)
            for u, v in game.edges:
                if edgetype is None:
                    edgetype = type(game.edges[u, v]['weight'])
                    edgetype_ = _type_cxx_intermediate(edgetype)
                game_cxx.append((mapper[u], mapper[v], edgetype_(game.edges[u, v]['weight'])))
            match edgetype():
                case int() | np.int64() | np.int32():
                    S_max, S_min = mpgcpp.optimal_strategy_pair_edges_cxx(game_cxx)
                case float() | np.float64():
                    S_max, S_min = mpgcpp.optimal_strategy_pair_double_edges_cxx(game_cxx)
                case np.float32():
                    S_max, S_min = mpgcpp.optimal_strategy_pair_float_edges_cxx(game_cxx)
                case _:
                    raise ValueError("Unknown edge type")
            inverse_mapper = {mapper[k]: k for k in mapper}
            S_max = [inverse_mapper[i] for i in S_max]
            S_min = [inverse_mapper[i] for i in S_min]
            if as_dict:
                S_max = {u: S_max[i] for i, u in enumerate(game.nodes)}
                S_min = {u: S_min[i] for i, u in enumerate(game.nodes)}
            pass
        case _:
            if as_dict is None:
                as_dict = False
            if edgetype is None:
                edgetype = int
            match edgetype():
                case int() | np.int64() | np.int32():
                    S_max, S_min = mpgcpp.optimal_strategy_pair_file_cxx(game)
                case float() | np.float64():
                    S_max, S_min = mpgcpp.optimal_strategy_pair_double_file_cxx(game)
                case np.float32():
                    S_max, S_min = mpgcpp.optimal_strategy_pair_float_file_cxx(game)
            if as_dict:
                raise RuntimeError("Argument as_dict is not supported for file input")

    return S_max,S_min

def winners(game: Union[MeanPayoffGraph,str], as_dict=True,edgetype=None) -> Union[List[Any], Dict[str, Any]]:
    """
    Computes the winners of a mean payoff game
    :param game: The mean payoff game, either as a MeanPayoffGraph object or as a path to a file
    :param timeout: The timeout in seconds
    :return: A dictionary containing the solution
    """
    n:int
    match game:
        case MeanPayoffGraph():
            mapper = dict(zip(game.nodes, range(len(game.nodes))))
            game_cxx: List[Tuple[int, int, int]] = []
            n=len(game.nodes)
            edgetype_ = None
            if edgetype is not None:
                edgetype_ = _type_cxx_intermediate(edgetype)
            for u, v in game.edges:
                if edgetype is None:
                    edgetype = type(game.edges[u, v]['weight'])
                    edgetype_ = _type_cxx_intermediate(edgetype)
                game_cxx.append((mapper[u], mapper[v], edgetype_(game.edges[u, v]['weight'])))
            match edgetype():
                case int() | np.int64() | np.int32():
                    W_max, W_min = mpgcpp.winners_edges_cxx(game_cxx)
                case float() | np.float64():
                    W_max, W_min = mpgcpp.winners_double_edges_cxx(game_cxx)
                case np.float32():
                    W_max, W_min = mpgcpp.winners_float_edges_cxx(game_cxx)
            if as_dict:
                W_max = {u: W_max[i] for i, u in enumerate(game.nodes)}
                W_min = {u: W_min[i] for i, u in enumerate(game.nodes)}
            pass
        case _:
            W_max,W_min=mpgcpp.winners_file_cxx(game)
            if as_dict:
                raise RuntimeError("Argument as_dict is not supported for file input")

    return W_max,W_min

def mean_payoffs(game: Union[MeanPayoffGraph,str], as_dict=True,edgetype=None) -> Union[List[Any], Dict[str, Any]]:
    """
    Computes the mean payoffs of a mean payoff game
    :param game: The mean payoff game, either as a MeanPayoffGraph object or as a path to a file
    :param timeout: The timeout in seconds
    :return: A dictionary containing the solution
    """
    n:int
    match game:
        case MeanPayoffGraph():
            mapper = dict(zip(game.nodes, range(len(game.nodes))))
            game_cxx: List[Tuple[int, int, int]] = []
            n=len(game.nodes)
            if edgetype is not None:
                edgetype_ = _type_cxx_intermediate(edgetype)
            for u, v in game.edges:
                if edgetype is None:
                    edgetype = type(game.edges[u, v]['weight'])
                    edgetype_ = _type_cxx_intermediate(edgetype)
                game_cxx.append((mapper[u], mapper[v], edgetype_(game.edges[u, v]['weight'])))
            match edgetype():
                case int() | np.int64() | np.int32():
                    M_max, M_min = mpgcpp.mean_payoffs_edges_cxx(game_cxx)
                case float() | np.float64():
                    M_max, M_min = mpgcpp.mean_payoffs_double_edges_cxx(game_cxx)
                case np.float32():
                    M_max, M_min = mpgcpp.mean_payoffs_float_edges_cxx(game_cxx)
            if as_dict:
                M_max = {u: M_max[i] for i, u in enumerate(game.nodes)}
                M_min = {u: M_min[i] for i, u in enumerate(game.nodes)}
            pass
        case _:
            M_max,M_min=mpgcpp.mean_payoffs_file_cxx(game)
            if as_dict:
                raise RuntimeError("Argument as_dict is not supported for file input")

    return M_max,M_min

def _convert_assignment(x):
    match x:
        case str():
            return -np.inf
        case _:
            return x
def arc_consistency(system:TernaryMaxAtomSystem,weighttype=None):
    """
    Solve the ternary max atom system using arc consistency
    :param system: The ternary max atom system
    :param weighttype: The type of the weights
    :return: A dictionary containing the solution
    """
    weighttype_=None
    if weighttype is not None:
        weighttype_=_type_cxx_intermediate(weighttype)
    system_cxx:List[Tuple[int,int,int,Any]]=[]
    mapper=dict(zip(system.variables,range(len(system.variables))))
    for x,y,z,w in system.constraints:
        if weighttype is None:
            weighttype=type(w)
            weighttype_=_type_cxx_intermediate(weighttype)
        system_cxx.append((mapper[x],mapper[y],mapper[z],weighttype_(w)))
    match weighttype():
        case int() | np.int64() | np.int32():
            assignment= mpgcpp.arc_consistency_cxx(system_cxx)
        case float() | np.float64():
            assignment= mpgcpp.arc_consistency_double_cxx(system_cxx)
        case np.float32():
            assignment= mpgcpp.arc_consistency_float_cxx(system_cxx)
        case _:
            raise RuntimeError("Unsupported weight type")
    return {x:_convert_assignment(assignment[i]) for i,x in enumerate(system.variables)}

if __name__== '__main__':
    game=MeanPayoffGraph()
    game.add_edge('a','b',weight=1)
    game.add_edge('a','c',weight=2)
    game.add_edge('b','c',weight=3)
    game.add_edge('c','a',weight=4)
    print(mean_payoffs(game))