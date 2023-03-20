from typing import Dict, Any, Union, List, Tuple

import mpgcpp
from csp.max_atom import MaxAtomSystem
from games import MeanPayoffGraph

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


def optimal_strategy_pair(game: Union[MeanPayoffGraph,str], as_dict=None) -> Union[List[Any], Dict[str, Any]]:
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
                dict = True
            mapper = dict(zip(game.nodes, range(len(game.nodes))))
            game_cxx: List[Tuple[int, int, int]] = []
            n=len(game.nodes)
            for u, v in game.edges:
                game_cxx.append((mapper[u], mapper[v], game.edges[u, v]['weight']))
            S_max, S_min = mpgcpp.optimal_strategy_pair_edges_cxx(game_cxx)
            inverse_mapper = {mapper[k]: k for k in mapper}
            S_max = [inverse_mapper[i] for i in S_max]
            S_min = [inverse_mapper[i] for i in S_min]
            if as_dict:
                S_max = {u: S_max[i] for i, u in enumerate(game.nodes)}
                S_min = {u: S_min[i] for i, u in enumerate(game.nodes)}
            pass
        case _:
            if as_dict is None:
                dict = False
            S_max,S_min=mpgcpp.optimal_strategy_pair_file_cxx(game)
            if as_dict:
                raise RuntimeError("Argument as_dict is not supported for file input")

    return S_max,S_min

def winners(game: Union[MeanPayoffGraph,str], as_dict=True) -> Union[List[Any], Dict[str, Any]]:
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
            for u, v in game.edges:
                game_cxx.append((mapper[u], mapper[v], game.edges[u, v]['weight']))
            W_max, W_min = mpgcpp.winners_edges_cxx(game_cxx)
            inverse_mapper = {mapper[k]: k for k in mapper}
            if as_dict:
                W_max = {u: W_max[i] for i, u in enumerate(game.nodes)}
                W_min = {u: W_min[i] for i, u in enumerate(game.nodes)}
            pass
        case _:
            W_max,W_min=mpgcpp.winners_file_cxx(game)
            if as_dict:
                raise RuntimeError("Argument as_dict is not supported for file input")

    return W_max,W_min

def mean_payoffs(game: Union[MeanPayoffGraph,str], as_dict=True) -> Union[List[Any], Dict[str, Any]]:
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
            for u, v in game.edges:
                game_cxx.append((mapper[u], mapper[v], game.edges[u, v]['weight']))
            M_max, M_min = mpgcpp.mean_payoffs_edges_cxx(game_cxx)
            if as_dict:
                M_max = {u: M_max[i] for i, u in enumerate(game.nodes)}
                M_min = {u: M_min[i] for i, u in enumerate(game.nodes)}
            pass
        case _:
            M_max,M_min=mpgcpp.mean_payoffs_file_cxx(game)
            if as_dict:
                raise RuntimeError("Argument as_dict is not supported for file input")

    return M_max,M_min

if __name__== '__main__':
    game=MeanPayoffGraph()
    game.add_edge('a','b',weight=1)
    game.add_edge('a','c',weight=2)
    game.add_edge('b','c',weight=3)
    game.add_edge('c','a',weight=4)
    print(mean_payoffs(game))