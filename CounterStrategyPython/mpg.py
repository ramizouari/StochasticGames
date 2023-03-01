import queue
from types import NoneType
from typing import Dict, Any, Union

import networkx as nx
import numpy as np

import max_atom as ma
from graph import FloydWarshallAlgorithm, ExperimentalMethod


class MeanPayoffGraph(nx.DiGraph):

    player0:int=0
    player1:int=1

    def __init__(self):
        super().__init__()
        self.bipartite=False

# This method is used to close an MPG
# It adds a self-loop to each node that has no outgoing edge with zero weight
    def closure(self):
        for u in self.nodes:
            if not self.out_edges(u):
                self.add_edge(u, u, weight=0)

# This method is used to add an edge to the MPG
    def add_edge(self, u, v, weight,**kwattr):
        super().add_edge(u, v, weight=weight,label=weight,**kwattr)

# This method is used to convert a MPG into a bipartite MPG
    def as_bipartite(self) -> "MeanPayoffGraph":
        if not self.bipartite:
            G = MeanPayoffGraph()
            for u in self.nodes:
                for v in self.succ[u]:
                    L= self[u][v]["weight"]
                    G.add_edge((u,self.player0), (v,self.player1), weight=L)
                    G.add_edge((u,self.player1),(v,self.player0),weight=L)
            G.bipartite=True
        else:
            G=self
        return G

# This method is used to convert a graph into a min-max offset system
    def as_min_max_system(self) -> ma.MinMaxSystem:
        S=ma.MinMaxSystem()
        BG=self.as_bipartite()
        for u in BG.nodes:
            _,p=u
            V=[v for v in BG.succ[u]]
            L=[BG[u][v]["weight"] for v in V]
            namer=lambda v: f"@{v[0]}P{v[1]}"
            S.add_constraint("min" if p==1 else "max",ma.Variable(id=u,name=namer(u)),[ma.Variable(id=v,name=namer(v)) for v in V],L)
        return S


# This function is used to check if a graph is winning everywhere
def winning_everywhere(G:MeanPayoffGraph) -> bool:
    return G.as_min_max_system().satisfiable()

def winning_somewhere(G:MeanPayoffGraph):
    return G.as_min_max_system().solve(include_inf=True)

# This function is used to get a winning strategy from a graph
# If the graph is not winning everywhere, it returns None
def winning_everywhere_strategy_csp(G:MeanPayoffGraph) -> Union[Dict[Any,Any],None]:
    S=G.as_min_max_system()
    admissible_values=S.solve()
    if not all(len(admissible_values[u])>0 for u in S.variables):
        return None
    assignment={u: max(admissible_values[u]) for u in S.variables}
    strategy={}
    for op,u,Y,C in S.constraints:
        if op=="max":
            R=assignment[Y[0]]+C[0]
            strategy[u.id[0]]=Y[0].id[0]
            for y,c in zip(Y,C):
                if assignment[y]+c > R:
                    R=assignment[y]+c
                    strategy[u.id[0]]=y.id[0]
    return strategy
def winning_somewhere_strategy_csp(G:MeanPayoffGraph) -> Union[Dict[Any,Any],None]:
    S=G.as_min_max_system()
    admissible_values=S.solve(include_inf=True)
    assignment={u: max(admissible_values[u]) for u in S.variables}
    strategy={}
    for op,u,Y,C in S.constraints:
        if op=="max":
            R=assignment[Y[0]]+C[0]
            strategy[u.id[0]]=Y[0].id[0]
            for y,c in zip(Y,C):
                if assignment[y]+c > R:
                    R=assignment[y]+c
                    strategy[u.id[0]]=y.id[0]
    return strategy
# This function is used to read a graph from a file
# The graph format is the following:
# Each line is either:
# - a single integer, which is the id of a node
# - three integers, which are the ids of two nodes and the weight of the edge
def mpg_from_file(file_name:str,ignore_header=0)->MeanPayoffGraph:
    G=MeanPayoffGraph()
    with open(file_name) as f:
        while ignore_header>0:
            f.readline()
            ignore_header-=1
        for line in f:
            if line.startswith("#"):
                continue
            L=line.split()
            if len(L)==1:
                G.add_node(int(L[0]))
            elif len(L)==3:
                u,v,w=L
                G.add_edge(int(u),int(v),int(w))
            else:
                continue
    return G


# This function is used to compute the counter-strategy of a graph
def counter_strategy(G: MeanPayoffGraph, psi: Dict[int,int],source=0, method="floyd-warshall") -> Dict[Any,Any]:
    strategyCost={u:0 for u in G.nodes}
    for u in G.nodes:
        for v in G.succ[u]:
            l = G[u][v]["weight"]
            if v == psi[u]:
                strategyCost[u] = l
    # The one-player equivalent game
    G1 = nx.DiGraph()
    for u in G.nodes:
        for v in G.succ[psi[u]]:
            L = G[psi[u]][v]["weight"]
            G1.add_edge(u, v, weight=L + strategyCost[u],label=L+strategyCost[u])
    match method:
        case "floyd-warshall":
            # 1. Apply Floyd-Warshall algorithm
            D = FloydWarshallAlgorithm(G1, retCycles=False)
            # 2. Compute the counter-strategy
            source=0
            counterStrategy=[-1 for _ in range(G.V)]
            for u in range(G.V):
                for v in G.adjacencyList[u]:
                    if D[u,v]==-np.inf:
                        counterStrategy[u]=v
            return counterStrategy

        case "experimental":
            _, C = ExperimentalMethod(G1, True)
            n = len(C)
            C.append(C[0])
            R = 0
            for k in range(n):
                for v, L in G1.adjacencyList[C[k]]:
                    if v == C[k + 1]:
                        R += L
            R /= n
            return C, R
        case "bellman-ford":
            # 1. Detect negative cycles
            if nx.negative_edge_cycle(G1):
                # 2. Finds a negative cycle
                C=nx.find_negative_cycle(G1,source=source)
                S=set(C)
                Q=queue.Queue()
                Q.put(source)
                visited={u:False for u in G.nodes}
                visited[source]=True
                parent={u:-1 for u in G.nodes}
                dest=-1
                # 3. Finds a vertex in the negative cycle, starting from the source, and computes the parent array
                while not Q.empty():
                    s=Q.get()
                    if s in S:
                        dest=s
                        break
                    for v in G1.succ[s]:
                        if not visited[v]:
                            Q.put(v)
                            parent[v]=s
                            visited[v]=True
                if dest==-1:
                    raise RuntimeError("Error in the algorithm")
                counterStrategy={u:-1 for u in G.nodes}
                # 4. Computes the counter-strategy along the path from the source to the vertex in the negative cycle
                while parent[dest]!=-1:
                    counterStrategy[psi[parent[dest]]]=dest
                    dest=parent[dest]
                # 5. Computes the counter-strategy along the negative cycle
                m=len(C)-1
                for k in range(m):
                    counterStrategy[psi[C[k]]]=C[(k+1)%m]
                # 6. Computes the counter-strategy for the remaining vertices
                for u in G.nodes:
                    if counterStrategy[u]==-1:
                        counterStrategy[u]=next(iter(G.succ[u]))
                return counterStrategy
        case _:
            raise NotImplementedError(f"Method {method} is not implemented yet")

if __name__=="__main__":
    G=mpg_from_file("data/test01.in",ignore_header=1)
    G.closure()
    print(G.edges(data=True))
    W=winning_somewhere(G)
    print({u.id[0]:max(W[u]) for u in W})
