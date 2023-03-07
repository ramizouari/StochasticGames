import queue
from typing import Dict, Any, Union, Tuple

import networkx as nx
import numpy as np

from csp import max_atom as ma
from csp.max_atom import MinMaxSystem
from graph import algorithms as graph_algorithms
from algbera import TropicalInteger


class VertexVariable(ma.Variable):
    """
    This class represents a vertex variable in the MPG CSP.
    """

    def __init__(self, vertex: Any, turn):
        """
        Initialize a new vertex variable
        :param name: The name of the variable
        :param domain: The domain of the variable
        """
        name = f"V_{turn}({vertex})"
        super().__init__(id=(vertex, turn), name=name)


class MeanPayoffGraph(nx.DiGraph):
    """
    This class represents a Mean Payoff Graph (MPG). It is a directed graph with weighted edges.

    An MPG is a directed graph G = (V, E) with a weight function L: E -> R, and a set of players P = {0, 1}.

    - The first player is called the Max player, his goal is to maximize the mean-payoff.

    - The second player is called the Min player, his goal is to minimize the mean-payoff.
    """
    player0: int = 0
    """
The first player is called the Max player, his goal is to maximize the mean-payoff.
    """

    player1: int = 1
    """
The second player is called the Min player, his goal is to minimize the mean-payoff.
    """

    def __init__(self):
        """
        Initialize a new MPG
        """
        super().__init__()
        self.bipartite = False

    # This method is used to close an MPG
    # It adds a self-loop to each node that has no outgoing edge with zero weight
    def closure(self) -> None:
        """
        Close the MPG by adding a self-loop to each node that has no outgoing edge with zero weight
        :return: None
        """
        for u in self.nodes:
            if not self.out_edges(u):
                self.add_edge(u, u, weight=0)

    # This method is used to add an edge to the MPG
    def add_edge(self, u, v, weight, **attr) -> None:
        """
        Add an edge to the MPG
        :param u: The source node
        :param v: The target node
        :param weight: The weight of the edge L(u,v)
        :param attr: Extra attributes of the edge
        :return: None
        """
        # The label attribute is used to display the weight of the edge in the visualisation
        super().add_edge(u, v, weight=weight, label=weight, **attr)

    # This method is used to convert an MPG into a bipartite MPG
    def as_bipartite(self) -> "MeanPayoffGraph":
        """
        Convert the MPG into an equivalent bipartite MPG
        :return: The bipartite MPG
        """
        if not self.bipartite:
            G = MeanPayoffGraph()
            for u in self.nodes:
                for v in self.succ[u]:
                    L = self[u][v]["weight"]
                    G.add_edge((u, self.player0), (v, self.player1), weight=L)
                    G.add_edge((u, self.player1), (v, self.player0), weight=L)
            G.bipartite = True
        else:
            G = self
        return G

    # This method is used to convert a graph into a min-max offset system
    def as_min_max_system(self) -> ma.MinMaxSystem:
        """
        Convert the MPG into an equivalent min-max offset system
        :return: The min-max offset system
        """
        S = ma.MinMaxSystem()
        BG = self.as_bipartite()
        for u in BG.nodes:
            vertex, p = u
            V = [v for v in BG.succ[u]]
            L = [BG[u][v]["weight"] for v in V]
            S.add_constraint("min" if p == 1 else "max", VertexVariable(vertex=vertex, turn=p),
                             [VertexVariable(vertex=successor, turn=q) for successor, q in V], L)
        return S

    def dual(self) -> "MeanPayoffGraph":
        """
        Compute the dual MPG.

        The dual MPG is the MPG obtained by negating the weights of the edges.
        :return: The dual MPG
        """
        G = MeanPayoffGraph()
        for u in self.nodes:
            for v in self.succ[u]:
                G.add_edge(u, v, -self[u][v]["weight"])
        return G

    def winners(self, L=None, R=None, method=MinMaxSystem.DEFAULT_METHOD) -> Tuple[Dict[Any, bool], Dict[Any, bool]]:
        solution = self.as_min_max_system().solve(L=L, R=R, method=method)
        Z1 = {}
        Z2 = {}
        for u in solution:
            s = u.id
            if s[1] == 0:
                Z1[s[0]] = solution[u] > -np.inf
            else:
                Z2[s[0]] = solution[u] == -np.inf
        return Z1, Z2


# This function is used to check if a graph is winning everywhere
def winning_everywhere(G: MeanPayoffGraph) -> bool:
    return G.as_min_max_system().satisfiable()


def winning_somewhere(G: MeanPayoffGraph):
    return G.as_min_max_system().solve()


# This function is used to get a winning strategy from a graph
# If the graph is not winning everywhere, it returns None
def winning_everywhere_strategy_csp(G: MeanPayoffGraph) -> Union[Dict[Any, Any], None]:
    S = G.as_min_max_system()
    admissible_values = S.solve()
    if not all(len(admissible_values[u]) > 0 for u in S.variables):
        return None
    assignment = {u: max(admissible_values[u]) for u in S.variables}
    strategy = {}
    for op, u, Y, C in S.constraints:
        if op == "max":
            R = assignment[Y[0]] + C[0]
            strategy[u.id[0]] = Y[0].id[0]
            for y, c in zip(Y, C):
                if assignment[y] + c > R:
                    R = assignment[y] + c
                    strategy[u.id[0]] = y.id[0]
    return strategy


def mpg_from_digraph(graph: nx.DiGraph) -> MeanPayoffGraph:
    """
    This function is used to convert a digraph into a MPG
    :param graph: The digraph
    :return: The mean payoff game
    :rtype: MeanPayoffGraph
    :raise ValueError: If the graph is not weighted
    """
    MPG = MeanPayoffGraph()
    for u in graph.nodes:
        for v in graph.succ[u]:
            MPG.add_edge(u, v, graph[u][v]["weight"])
    return MPG


# This function is used to get optimal strategies from a graph
# It returns a pair of strategies, one for each player
def optimal_strategy_pair(game: MeanPayoffGraph, method=MinMaxSystem.DEFAULT_METHOD) -> \
        Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """
    This function is used to get optimal strategies from a mean payoff graph
    :param game: The mean payoff graph
    :param method: The method used to solve the min-max offset system
    :return: A pair of strategies, one for each player. The first strategy is the strategy of the first player, the second
    strategy is the strategy of the second player
    """
    # We first solve the min-max offset system associated to the MPG
    S = game.as_min_max_system()
    assignment = S.solve(method=method)
    # We then solve the min-max offset system associated to the dual MPG
    # TODO: Prove that this transformation is correct
    SD = game.dual().as_min_max_system()
    counter_assignment = SD.solve(method=method)
    # We then compute the optimal strategies
    strategy = {}
    counter_strategy = {}
    # We iterate over the two systems
    for current_system, current_assigment, current_strategy in zip([S, SD], [assignment, counter_assignment],
                                                                   [strategy, counter_strategy]):
        # We iterate over the constraints
        for op, u, Y, C in current_system.constraints:
            if op == "max":
                R = current_assigment[Y[0]] + C[0]
                current_strategy[u.id[0]] = Y[0].id[0]
                # We iterate over the successors of the current vertex
                # We find the one with the highest value
                # This is the optimal strategy with respect to the current system
                # TODO: Prove that this gives optimal strategy
                for y, c in zip(Y, C):
                    if current_assigment[y] + c >= R:
                        R = current_assigment[y] + c
                        current_strategy[u.id[0]] = y.id[0]
    return strategy, counter_strategy


# This function is used to get optimal strategies from a graph
# It returns a pair of strategies, one for each player
def strategy_pair_tropical(game: MeanPayoffGraph, iters=100, L=-np.inf, R=+np.inf) -> \
        Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """
    This function is used to get approximate optimal strategies from a mean payoff graph using the tropical method
    :param iters: The number of iterations
    :param game: The mean payoff graph
    :param L: The lower bound
    :param R: The upper bound
    :return: A pair of strategies, one for each player. The first strategy is the strategy of the first player, the second
    strategy is the strategy of the second player
    """
    f: Dict[Any, float] = {u: 0 for u in game.nodes}
    g: Dict[Any, float] = {u: 0 for u in game.nodes}
    for i in range(2 * iters):
        h = f if i % 2 == 0 else g
        tmp = h.copy()
        op = max if i % 2 == 0 else min
        for u in game.nodes:
            for v in game.succ[u]:
                tmp[u] = op(tmp[u], game[u][v]["weight"] + h[v])
                if tmp[u] < L:
                    tmp[u] = -np.inf
                if tmp[u] > R:
                    g[u] = +np.inf
        if i % 2 == 0:
            f = tmp
        else:
            g = tmp
    strategy = {}
    counter_strategy = {}
    for u in game.nodes:
        A = -np.inf
        B = np.inf
        for v in game.succ[u]:
            if game[u][v]["weight"] + f[v] >= A:
                strategy[u] = v
                A = game[u][v]["weight"] + f[v]
            if game[u][v]["weight"] + g[v] <= B:
                counter_strategy[u] = v
                B = game[u][v]["weight"] + g[v]
    return strategy, counter_strategy


# This function is used to read a graph from a file
# The graph format is the following:
# Each line is either:
# - a single integer, which is the id of a node
# - three integers, which are the ids of two nodes and the weight of the edge
def mpg_from_file(file_name: str, ignore_header=0) -> MeanPayoffGraph:
    """
    Read a mean pay-off game from a file
    :param file_name: The name of the file
    :param ignore_header: The number of lines to ignore at the beginning of the file
    :return: The mean pay-off game
    """
    G = MeanPayoffGraph()
    with open(file_name) as f:
        while ignore_header > 0:
            f.readline()
            ignore_header -= 1
        for line in f:
            if line.startswith("#"):
                continue
            L = line.split()
            if len(L) == 1:
                G.add_node(int(L[0]))
            elif len(L) == 3:
                u, v, w = L
                G.add_edge(int(u), int(v), int(w))
            else:
                continue
    return G


# This function is used to compute the counter-strategy of a graph
def counter_strategy(game: MeanPayoffGraph, psi: Dict[Any, Any], source=0, method="floyd-warshall",
                     player=MeanPayoffGraph.player1) -> Dict[Any, Any]:
    """
    Compute the counter-strategy of a graph
    :param game: The mean pay-off game
    :param psi: The strategy of the player
    :param source: The source node
    :param method: The method to use to compute the shortest paths
    :param player: The player for which we compute the counter-strategy
    :return: The counter-strategy
    """
    # If the counter-player is 0, we need to compute the dual graph
    if player == MeanPayoffGraph.player0:
        game = game.dual()
    # Compute the cost of the strategy for each node
    # TODO: This can be done more efficiently
    strategyCost = {u: 0 for u in game.nodes}
    for u in game.nodes:
        for v in game.succ[u]:
            l = game[u][v]["weight"]
            if v == psi[u]:
                strategyCost[u] = l
    # The one-player equivalent game
    # The equivalent game is build differently depending on the player
    G1 = nx.DiGraph()
    for u in game.nodes:
        # If the player is 1, we need to add the edges from the dual graph
        if player == MeanPayoffGraph.player1:
            for v in game.succ[psi[u]]:
                L = game[psi[u]][v]["weight"]
                G1.add_edge(u, v, weight=L + strategyCost[u], label=L + strategyCost[u])
        # If the player is 0, we need to add the edges from the original graph
        else:
            for v in game.succ[u]:
                L = game[u][v]["weight"]
                G1.add_edge(u, psi[v], weight=L + strategyCost[v], label=L + strategyCost[v])
    match method:
        case "floyd-warshall":
            # 1. Detect negative cycles
            successor = graph_algorithms.floyd_warshall_negative_paths(G1)
            if player == MeanPayoffGraph.player1:
                strategy = {psi[u]: successor[u] for u in game.nodes}
            else:
                strategy = {}
                for u in game.nodes:
                    for v in game.succ[u]:
                        if successor[u] == psi[v]:
                            strategy[u] = v
                            break
            for u in game.nodes:
                if not u in strategy:
                    strategy[u] = next(iter(game.succ[u]))
            return strategy

        case "bellman-ford":
            # 1. Detect negative cycles
            if nx.negative_edge_cycle(G1):
                # 2. Finds a negative cycle
                try:
                    C = nx.find_negative_cycle(G1, source=source)
                except nx.NetworkXError:
                    # If there is no negative cycle starting from the source, we return any strategy
                    return {u: next(iter(game.succ[u])) for u in game.nodes}
                S = set(C)
                Q = queue.Queue()
                Q.put(source)
                visited = {u: False for u in game.nodes}
                visited[source] = True
                parent = {u: -1 for u in game.nodes}
                dest = -1
                # 3. Finds a vertex in the negative cycle, starting from the source, and computes the parent array
                while not Q.empty():
                    s = Q.get()
                    if s in S:
                        dest = s
                        break
                    for v in G1.succ[s]:
                        if not visited[v]:
                            Q.put(v)
                            parent[v] = s
                            visited[v] = True
                if dest == -1:
                    raise RuntimeError("Error in the algorithm")
                counterStrategy = {u: -1 for u in game.nodes}

                if player == MeanPayoffGraph.player1:
                    # 4. Computes the counter-strategy along the path from the source to the vertex in the negative cycle
                    while parent[dest] != -1:
                        counterStrategy[psi[parent[dest]]] = dest
                        dest = parent[dest]
                    # 5. Computes the counter-strategy along the negative cycle
                    m = len(C) - 1
                    for k in range(m):
                        counterStrategy[psi[C[k]]] = C[(k + 1) % m]
                    # 6. Computes the counter-strategy for the remaining vertices
                    for u in game.nodes:
                        if counterStrategy[u] == -1:
                            counterStrategy[u] = next(iter(game.succ[u]))
                else:
                    # 4. Computes the counter-strategy along the path from the source to the vertex in the negative cycle
                    while parent[dest] != -1:
                        counterStrategy[parent[dest]] = dest
                        dest = parent[dest]

                    # 5. Computes the counter-strategy along the negative cycle
                    m = len(C) - 1
                    for k in range(m):
                        counterStrategy[C[k]] = C[(k + 1) % m]

                    for u in game.nodes:
                        if counterStrategy[u] == -1:
                            continue
                        for v in game.succ[u]:
                            if counterStrategy[u] == psi[v]:
                                counterStrategy[u] = v
                                break
                    # 6. Computes the counter-strategy for the remaining vertices
                    for u in game.nodes:
                        if counterStrategy[u] == -1:
                            counterStrategy[u] = next(iter(game.succ[u]))
                return counterStrategy
        case _:
            raise NotImplementedError(f"Method {method} is not implemented yet")


# This function is used to compute the mean payoff of a graph
# G is the graph
# starting_position is the starting position
# strategy1 is the strategy of the first player
# strategy2 is the strategy of the second player
def mean_payoff(game: MeanPayoffGraph, starting_position, strategy1,
                strategy2, starting_turn=MeanPayoffGraph.player0) -> float:
    """
    Compute the mean payoff of a graph
    :param game: The mean payoff graph
    :param starting_position: The starting position
    :param strategy1: The strategy of the first player
    :param strategy2: The strategy of the second player
    :param starting_turn: The starting player
    :return: The mean payoff
    """
    # The graph is converted into a bipartite graph
    BG = game.as_bipartite()
    # The visited array is used to detect cycles
    visited = {u: False for u in BG.nodes}
    # The total payoff array is used to compute the mean payoff
    total_payoff = {u: 0 for u in BG.nodes}
    # The payoff of the last visited node
    last_payoff = 0
    # The index of each node
    index = {u: 0 for u in BG.nodes}
    # The index of the last visited node
    last_index = 0
    # The starting node of the bipartite graph
    U = (starting_position, starting_turn)
    while not visited[U]:
        # The current position and the current player
        current_position, current_player = U
        # The next position and the next player
        V = (strategy1[current_position] if not current_player else strategy2[current_position], not current_player)
        next_position, next_player = V
        visited[U] = True
        # If the next node has already been visited, then a cycle has been detected
        if visited[V]:
            # The number of nodes in the cycle
            n = last_index - index[V] + 1
            # The mean payoff is computed
            return (last_payoff + game[current_position][next_position]["weight"] - total_payoff[V]) / n
        else:
            total_payoff[V] = last_payoff + game[current_position][next_position]["weight"]
            last_payoff = total_payoff[V]
        last_index += 1
        index[V] = last_index
        U = V


# This function is used to compute the mean payoffs of a graph at each position
# G is the graph
# strategy1 is the strategy of the first player
# strategy2 is the strategy of the second player
def mean_payoffs(game: MeanPayoffGraph, strategy1, strategy2) -> Dict[Tuple[Any, Any], float]:
    """
    Compute the mean payoff of a graph at each position
    :param game: The mean payoff graph
    :param strategy1: The strategy of the first player
    :param strategy2: The strategy of the second player
    :return: The mean payoff as a function of the position and the player. The key is a tuple (position, player)
    """
    # The graph is converted into a bipartite graph
    BG = game.as_bipartite()
    # The visited array is used to detect cycles
    visited = {u: False for u in BG.nodes}
    has_value = {u: False for u in BG.nodes}
    # The total payoff array is used to compute the mean payoff
    total_payoff = {u: 0 for u in BG.nodes}
    # The payoff of the last visited node
    # The index of each node
    index = {u: 0 for u in BG.nodes}
    # The index of the last visited node
    # The mean payoffs
    MPOs = {}
    for starting_position, starting_turn in BG.nodes:
        last_index = 0
        last_payoff = 0
        # The starting node of the bipartite graph
        U = (starting_position, starting_turn)
        if visited[U]:
            continue
        L = []
        MPO_source = 0
        while not visited[U]:
            L.append(U)
            # The current position and the current player
            current_position, current_player = U
            # The next position and the next player
            V = (strategy1[current_position] if not current_player else strategy2[current_position], not current_player)
            next_position, next_player = V
            visited[U] = True
            # If the next node has already been visited, then a cycle has been detected
            if visited[V] and not has_value[V]:
                # The number of nodes in the cycle
                n = last_index - index[V] + 1
                # The mean payoff is computed
                MPOs[V] = (last_payoff + game[current_position][next_position]["weight"] - total_payoff[V]) / n
                has_value[V] = True
            elif not visited[V]:
                total_payoff[V] = last_payoff + game[current_position][next_position]["weight"]
                last_payoff = total_payoff[V]
            last_index += 1
            index[V] = last_index
            U = V
        for V in L:
            if not has_value[V]:
                MPOs[V] = MPOs[U]
                has_value[V] = True
    return MPOs


# This function is used to get the winner of a game
# G is the graph
# starting_point is the starting position
# strategy1 is the strategy of the first player
# strategy2 is the strategy of the second player
def winner(game: MeanPayoffGraph, starting_point, strategy1, strategy2, starting_turn=MeanPayoffGraph.player0) -> bool:
    """
    Get the winner of a game
    :param game: The mean payoff graph
    :param starting_point: The starting position
    :param strategy1: The strategy of the first player
    :param strategy2: The strategy of the second player
    :param starting_turn: The starting player
    :return: True if the first player wins, False otherwise
    """
    return mean_payoff(game, starting_point, strategy1, strategy2, starting_turn=starting_turn) >= 0


# This function is used to get winners of a game as a function of the starting position
# G is the graph
# strategy1 is the strategy of the first player
# strategy2 is the strategy of the second player
def winners(game: MeanPayoffGraph, strategy1, strategy2) -> Dict[Tuple[Any, Any], bool]:
    """
    Get the winners of a game as a function of the starting position
    :param game: The mean payoff graph
    :param strategy1: The strategy of the first player
    :param strategy2: The strategy of the second player
    :return: The winners of a game as a function of the starting position and the starting player.
    The key is a tuple (position, player).
    """
    winners = {}
    for s, p in game.as_bipartite().nodes:
        winners[(s, p)] = winner(game, s, strategy1, strategy2, starting_turn=p)
    return winners


if __name__ == "__main__":
    import visualisation.game as vgame

    G = MeanPayoffGraph()
    G.add_edge(0, 1, 5)
    G.add_edge(0, 2, -5)
    G.add_edge(2, 1, 3)
    G.add_edge(2, 0, -6)
    G.add_edge(1, 0, -6)
    G.closure()
    S1, S2 = optimal_strategy_pair(G)
    print(mean_payoffs(G, S1, S2))
