import queue
from typing import Dict, Any, Union, Tuple

import networkx as nx
import numpy as np

from csp import max_atom as ma
from csp.max_atom import MinMaxSystem
from graph import algorithms as graph_algorithms


class MeanPayoffGraph(nx.DiGraph):
    player0: int = 0
    player1: int = 1

    def __init__(self):
        super().__init__()
        self.bipartite = False

    # This method is used to close an MPG
    # It adds a self-loop to each node that has no outgoing edge with zero weight
    def closure(self):
        for u in self.nodes:
            if not self.out_edges(u):
                self.add_edge(u, u, weight=0)

    # This method is used to add an edge to the MPG
    def add_edge(self, u, v, weight, **attr):
        super().add_edge(u, v, weight=weight, label=weight, **attr)

    # This method is used to convert a MPG into a bipartite MPG
    def as_bipartite(self) -> "MeanPayoffGraph":
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
        S = ma.MinMaxSystem()
        BG = self.as_bipartite()
        for u in BG.nodes:
            _, p = u
            V = [v for v in BG.succ[u]]
            L = [BG[u][v]["weight"] for v in V]
            namer = lambda v: f"@{v[0]}P{v[1]}"
            S.add_constraint("min" if p == 1 else "max", ma.Variable(id=u, name=namer(u)),
                             [ma.Variable(id=v, name=namer(v)) for v in V], L)
        return S

    def dual(self) -> "MeanPayoffGraph":
        G = MeanPayoffGraph()
        for u in self.nodes:
            for v in self.succ[u]:
                G.add_edge(u, v, -self[u][v]["weight"])
        return G

    def winners(self,L=None,R=None,method=MinMaxSystem.DEFAULT_METHOD) -> Tuple[Dict[Any,bool], Dict[Any,bool]]:
        solution=self.as_min_max_system().solve(L=L,R=R,method=method)
        Z1={}
        Z2={}
        for u in solution:
            s=u.id
            if s[1]==0:
                Z1[s[0]]=solution[u] > -np.inf
            else:
                Z2[s[0]]=solution[u] == -np.inf
        return Z1,Z2
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


def mpg_from_digraph(G: nx.DiGraph) -> MeanPayoffGraph:
    MPG = MeanPayoffGraph()
    for u in G.nodes:
        for v in G.succ[u]:
            MPG.add_edge(u, v, G[u][v]["weight"])
    return MPG


# This function is used to get optimal strategies from a graph
# It returns a pair of strategies, one for each player
def optimal_strategy_pair(G: MeanPayoffGraph, method=MinMaxSystem.DEFAULT_METHOD) -> Tuple[
    Dict[Any, Any], Dict[Any, Any]]:
    S = G.as_min_max_system()
    assignment = S.solve(method=method)
    strategy = {}
    counter_strategy = {}
    for op, u, Y, C in S.constraints:
        if op == "max":
            R = assignment[Y[0]] + C[0]
            strategy[u.id[0]] = Y[0].id[0]
            for y, c in zip(Y, C):
                if assignment[y] + c > R:
                    R = assignment[y] + c
                    strategy[u.id[0]] = y.id[0]
        else:
            R = assignment[Y[0]] + C[0]
            counter_strategy[u.id[0]] = Y[0].id[0]
            for y, c in zip(Y, C):
                if assignment[y] + c < R:
                    R = assignment[y] + c
                    counter_strategy[u.id[0]] = y.id[0]
    return strategy, counter_strategy


# This function is used to read a graph from a file
# The graph format is the following:
# Each line is either:
# - a single integer, which is the id of a node
# - three integers, which are the ids of two nodes and the weight of the edge
def mpg_from_file(file_name: str, ignore_header=0) -> MeanPayoffGraph:
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
def counter_strategy(G: MeanPayoffGraph, psi: Dict[Any, Any], source=0, method="floyd-warshall",
                     player=MeanPayoffGraph.player1) -> Dict[Any, Any]:
    # If the counter-player is 0, we need to compute the dual graph
    if player == MeanPayoffGraph.player0:
        G = G.dual()
    strategyCost = {u: 0 for u in G.nodes}
    for u in G.nodes:
        for v in G.succ[u]:
            l = G[u][v]["weight"]
            if v == psi[u]:
                strategyCost[u] = l
    # The one-player equivalent game
    G1 = nx.DiGraph()
    for u in G.nodes:
        if player == MeanPayoffGraph.player1:
            for v in G.succ[psi[u]]:
                L = G[psi[u]][v]["weight"]
                G1.add_edge(u, v, weight=L + strategyCost[u], label=L + strategyCost[u])
        else:
            for v in G.succ[u]:
                L = G[u][v]["weight"]
                G1.add_edge(u, psi[v], weight=L + strategyCost[v], label=L + strategyCost[v])
    match method:
        case "floyd-warshall":
            successor = graph_algorithms.floyd_warshall_negative_paths(G1)
            if player == MeanPayoffGraph.player1:
                strategy = {psi[u]: successor[u] for u in G.nodes}
            else:
                strategy = {}
                for u in G.nodes:
                    for v in G.succ[u]:
                        if successor[u] == psi[v]:
                            strategy[u] = v
                            break
            for u in G.nodes:
                if not u in strategy:
                    strategy[u] = next(iter(G.succ[u]))
            return strategy

        case "bellman-ford":
            # 1. Detect negative cycles
            if nx.negative_edge_cycle(G1):
                # 2. Finds a negative cycle
                C = nx.find_negative_cycle(G1, source=source)
                S = set(C)
                Q = queue.Queue()
                Q.put(source)
                visited = {u: False for u in G.nodes}
                visited[source] = True
                parent = {u: -1 for u in G.nodes}
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
                counterStrategy = {u: -1 for u in G.nodes}

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
                    for u in G.nodes:
                        if counterStrategy[u] == -1:
                            counterStrategy[u] = next(iter(G.succ[u]))
                else:
                    # 4. Computes the counter-strategy along the path from the source to the vertex in the negative cycle
                    while parent[dest] != -1:
                        counterStrategy[parent[dest]] = dest
                        dest = parent[dest]

                    # 5. Computes the counter-strategy along the negative cycle
                    m = len(C) - 1
                    for k in range(m):
                        counterStrategy[C[k]] = C[(k + 1) % m]

                    for u in G.nodes:
                        if counterStrategy[u] == -1:
                            continue
                        for v in G.succ[u]:
                            if counterStrategy[u] == psi[v]:
                                counterStrategy[u] = v
                                break
                    # 6. Computes the counter-strategy for the remaining vertices
                    for u in G.nodes:
                        if counterStrategy[u] == -1:
                            counterStrategy[u] = next(iter(G.succ[u]))
                return counterStrategy
        case _:
            raise NotImplementedError(f"Method {method} is not implemented yet")

# This function is used to compute the mean payoff of a graph
# G is the graph
# starting_position is the starting position
# strategy1 is the strategy of the first player
# strategy2 is the strategy of the second player
def mean_payoff(G:MeanPayoffGraph,starting_position,strategy1,strategy2,starting_turn=MeanPayoffGraph.player0) -> float:
    BG = G.as_bipartite()
    visited = {u: False for u in BG.nodes}
    total_payoff = {u: 0 for u in BG.nodes}
    last_payoff = 0
    index={u:0 for u in BG.nodes}
    last_index=0
    U = (starting_position, starting_turn)
    while not visited[U]:
        current_position, current_player = U
        V=(strategy1[current_position] if not current_player else strategy2[current_position],not current_player)
        next_position, next_player = V
        visited[U] = True
        if visited[V]:
            n=last_index-index[V]
            return (last_payoff + G[current_position][next_position]["weight"]-total_payoff[V])/n
        else:
            total_payoff[V] = last_payoff + G[current_position][next_position]["weight"]
            last_payoff = total_payoff[V]
        index[V]=last_index
        last_index+=1
        U = V

# This function is used to compute the mean payoffs of a graph at each position
# G is the graph
# strategy1 is the strategy of the first player
# strategy2 is the strategy of the second player
def mean_payoffs(G:MeanPayoffGraph,strategy1,strategy2) -> Dict[Tuple[Any,Any],float]:
    payoffs={}
    for s,p in G.as_bipartite().nodes:
        payoffs[(s,p)]=mean_payoff(G,s,strategy1,strategy2,starting_turn=p)
    return payoffs

# This function is used to get the winner of a game
# G is the graph
# starting_point is the starting position
# strategy1 is the strategy of the first player
# strategy2 is the strategy of the second player
def winner(G:MeanPayoffGraph,starting_point,strategy1,strategy2,starting_turn=MeanPayoffGraph.player0) -> bool:
    return mean_payoff(G,starting_point,strategy1,strategy2,starting_turn=starting_turn)>=0

# This function is used to get winners of a game as a function of the starting position
# G is the graph
# strategy1 is the strategy of the first player
# strategy2 is the strategy of the second player
def winners(G:MeanPayoffGraph,strategy1,strategy2) -> Dict[Tuple[Any,Any],bool]:
    winners={}
    for s,p in G.as_bipartite().nodes:
        winners[(s,p)]=winner(G,s,strategy1,strategy2,starting_turn=p)
    return winners

if __name__ == "__main__":
    G = mpg_from_file("data/test01.in", ignore_header=1)
    G.closure()
    Z = "1 2 3 6 5 6 1 1".split()
    psi = dict(zip(range(len(Z)), map(int, Z)))
    # print(counter_strategy(G,psi,method="bellman-ford"))
    # print(G.edges(data=True))
    print(G.winners())
    W = optimal_strategy_pair(G, method="ACO")
    print(mean_payoff(G, 0, *W))
    #print(W)
    # print(optimal_strategy_pair(G))
