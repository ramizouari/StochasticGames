import queue
from typing import List, Tuple, Callable

import networkx as nx
import numpy as np

import union_find
from extended_integer import TropicalInteger


class Graph:
    def __init__(self, V):
        self.V = V
        self.E = 0
        self.adjacencyList: List[List[int]] = [[] for _ in range(V)]
        self.reverseList: List[List[int]] = [[] for i in range(V)]

    def addEdge(self, edge):
        (u, v) = edge
        self.adjacencyList[u].append(v)
        self.reverseList[v].append(u)
        self.E += 1

    def _topological_sort_rec(self, r: int, L: List[int], visited: List[bool]):
        if not visited[r]:
            visited[r] = True
            for s in self.adjacencyList[r]:
                self._topological_sort_rec(s, L, visited)
            L.append(r)
            pass

    def _assign_component(self, u, v, C: union_find.UnionFind, isAssigned: List[bool]):
        if not isAssigned[u]:
            isAssigned[u] = True
            C.connect(u, v)
            for s in self.reverseList[u]:
                self._assign_component(s, v, C, isAssigned)
        pass

    def topologicalSort(self, source=0):
        L = []
        visited = [False for u in range(self.V)]
        for u in range(self.V):
            self._topological_sort_rec(u, L, visited)
        L.reverse()
        return L

    def stronglyConnectedComponents(self):
        C = union_find.UnionFind(self.V)
        isAssigned = [False for u in range(self.V)]
        topologicalOrder = []
        for l in self.topologicalSort():
            self._assign_component(l, l, C, isAssigned)
            if l == C.representative(l):
                topologicalOrder.append(l)

        idAssigned = [False for k in range(self.V)]
        componentId = [u for u in range(self.V)]
        components = []
        for u in range(self.V):
            r = C.representative(u)
            if not idAssigned[r]:
                idAssigned[r] = True
                componentId[r] = len(components)
                components.append([])
            componentId[u] = componentId[r]
            components[componentId[r]].append(u)
        return components, componentId, C, topologicalOrder
    def vertices(self):
        return range(self.V)


class LabeledGraph:
    def __init__(self, V):
        self.V = V
        self.E = 0
        self.adjacencyList: List[List[Tuple[int, TropicalInteger]]] = [[] for _ in range(V)]
        self.reverseList: List[List[Tuple[int, TropicalInteger]]] = [[] for i in range(V)]

    def addEdge(self, edge):
        (u, v, l) = edge
        self.adjacencyList[u].append((v, l))
        self.reverseList[v].append((u, l))
        self.E += 1

    pass

    def adjacencyMatrix(self):
        M = np.zeros((self.V, self.V))
        for u in range(self.V):
            for v, L in self.adjacencyList[u]:
                M[u, v] = L

    def __repr__(self):
        return f"""Labeled Graph with {self.V} vertices and {self.E} edges:
{ {u: list(map(lambda s: {"dest": s[0], "weight": s[1]}, self.adjacencyList[u])) for u in range(self.V)} }
"""
    def vertices(self):
        return range(self.V)


WeightedGraph = LabeledGraph


def MeanPayoffGraph(LabeledGraph):
    def __init__(self, V):
        super().__init__(V)

    def closure(self):
        for u in range(self.V):
            if len(self.adjacencyList[u]) == 0:
                self.addEdge(u, u, 0)


def FloydWarshallAlgorithm(G: LabeledGraph, retCycles=True):
    D = np.full((G.V, G.V), np.inf)
    for i in range(G.V):
        D[i, i] = 0
    for u in range(G.V):
        for v, L in G.adjacencyList[u]:
            D[u, v] = L
    cycles = []
    ancestor = [i for i in range(G.V)]
    C = -1
    for w in range(G.V):
        for u in range(G.V):
            for v in range(G.V):
                D[u,v]=np.minimum(D[u,v],D[u,w]+D[w,v])
                if u==v and D[u,w] + D[w,v] < 0:
                    D[u,v]=-np.inf
                    D[u,w]=-np.inf
                    D[w,v]=-np.inf
    for u in range(G.V):
        for v in range(G.V):
            for w, L in G.adjacencyList[v]:
                if D[u, v] + L < D[u, w]:
                    C = v
                    D[u,w]=-np.inf
    return D


def BellmanFordAlgorithm(G: LabeledGraph, source=0, retCycles=True):
    # 1. Initialization
    d = np.full(G.V,np.inf)
    d[source] = 0
    # 2. Relaxation
    for k in range(G.V - 1):
        visited = [False for v in range(G.V)]
        visited[source] = True
        Q = queue.Queue()
        Q.put(source)
        while not Q.empty():
            s = Q.get()
            for v, L in G.adjacencyList[s]:
                d[v] = np.minimum(d[s] + L, d[v])
                if not visited[v]:
                    visited[v] = True
                    Q.put(v)
    # 3. Cycle detection
    Q = queue.Queue()
    Q.put(source)
    visited = [False for v in range(G.V)]
    visited[source] = True
    Q = queue.Queue()
    Q.put(source)
    parent = [v for v in range(G.V)]
    while not Q.empty():
        s = Q.get()
        for v, L in G.adjacencyList[s]:
            if d[v] > d[s] + L:
                parent[v] = s
            if not visited[v]:
                visited[v] = True
                Q.put(v)
            pass
    if not retCycles:
        return d

    # 4. Cycle construction
    cycles = []
    for v in range(G.V):
        if visited[v]:
            cycles.append(v)
            w = parent[v]
            while w != v:
                cycles.append(w)
                w = parent[w]
            cycles.reverse()
            break
    return d, cycles

# This is a suboptimal method, running in O(n²m), But it is used for now for negative cycles construction
def ExperimentalMethod(G: LabeledGraph, retCycles=True):
    D = np.full((G.V, G.V), np.inf)
    for i in range(G.V):
        D[i, i] = 0
    cycles = []
    ancestor = [i for i in range(G.V)]
    C = -1
    for k in range(G.V):
        for u in range(G.V):
            for v in range(G.V):
                for w, L in G.adjacencyList[v]:
                    if D[u, w] > D[u, v] + L:
                        D[u, w] = D[u, v] + L
                        ancestor[w] = v
    for u in range(G.V):
        for v in range(G.V):
            for w, L in G.adjacencyList[v]:
                if D[u, v] + L < D[u, w]:
                    C = v
    k = ancestor[C]
    cycles.append(C)
    while k != C:
        cycles.append(k)
        k = ancestor[k]
    cycles.reverse()
    if retCycles:
        return D, cycles
    return D


def counterStrategy(G: LabeledGraph, psi: List[int], method="floyd-warshall"):
    strategyCost: List[TropicalInteger] = [TropicalInteger(0) for _ in range(G.V)]
    for u in range(G.V):
        for (v, l) in G.adjacencyList[u]:
            if v == psi[u]:
                strategyCost[u] = l
    # The one-player equivalent game
    G1 = LabeledGraph(G.V)
    G2 = LabeledGraph(G.V)
    for u in range(G.V):
        for v, L in G.adjacencyList[psi[u]]:
            G1.addEdge((u, v, L + strategyCost[u]))
        for v, L in G.adjacencyList[u]:
            G2.addEdge((u, psi[v], L+strategyCost[v]))
    match method:
        case "floyd-warshall":
            # 1. Apply Floyd-Warshall algorithm
            D = FloydWarshallAlgorithm(G1, retCycles=False)
            # 2. Compute the counter-strategy
            source=0
            counterStrategy=[-1 for _ in range(G.V)]
            for u in range(G.V):
                for v,L in G1.adjacencyList[u]:
                    if D[u,v]==-np.inf:
                        counterStrategy[psi[u]]=v
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
        case "bellman-ford" | "bellmanford" | "bellman" | "bf" | "bford" | "b-ford" | "b-f" | "b-ford" | "bellman_ford" | "bellman_f" | "bellmanford" | "bellman_f":
            # 1. Apply Bellman-Ford algorithm on the one-player equivalent game
            d = BellmanFordAlgorithm(G1, 0, retCycles=False)
            # 2. Calculate the counter strategy
            source=0
            counterStrategy = [-1 for v in range(G.V)]
            outward=[[] for v in range(G.V)]
            Q = queue.Queue()
            Q.put(source)
            visited = [False for v in range(G.V)]
            valuation=[+np.inf for v in range(G.V)]
            while not Q.empty():
                s = Q.get()
                if visited[s]:
                    continue
                visited[s]=True
                for v, L in G1.adjacencyList[s]:
                    if d[v] <= valuation[s]:
                        counterStrategy[psi[s]]=v
                        outward[s].append(v)
                        valuation[s]=d[v]
                    Q.put(v)
            return outward
        case _:
            raise NotImplementedError(f"Method {method} is not implemented yet")


def negative_cycle(g: nx.DiGraph,weight='weight'):
    def getW(e):
        if weight is None:
            return 1
        return e[2][weight]
    n = len(g.nodes())
    edges = list(g.edges().data())
    d = np.ones(n) * 10000
    p = np.ones(n) * -1
    x = -1
    for i in range(n):
        for e in edges:
            if d[int(e[0])] + getW(e) < d[int(e[1])]:
                d[int(e[1])] = d[int(e[0])] + getW(e)
                p[int(e[1])] = int(e[0])
                x = int(e[1])
    if x == -1:
        print("No negative cycle")
        return None
    for i in range(n):
        x = p[int(x)]
    cycle = []
    v = x
    while True:
        cycle.append(str(int(v)))
        if v == x and len(cycle) > 1:
            break
        v = p[int(v)]
    return list(reversed(cycle))

# This method reads a graph from a file
def read_from_text_file(file_name,graph_type="auto",directed=True):
    with open(file_name,"r") as file:
        autoDetect=graph_type=="auto"
        line =file.readline().rstrip()
        V,E=map(int,line.split())
        line =file.readline().rstrip()
        splitted=line.split()
        match graph_type:
            case "auto":
                if len(splitted)>2:
                    G=WeightedGraph(V)
                elif len(splitted)==2:
                    G=Graph(V)
                else:
                    raise RuntimeError("File format error")
            case "weighted"|"labeled":
                G=WeightedGraph(V)
            case "unweighted"|"unlabeled":
                G=Graph(V)
        while len(splitted) in [2,3]:
            G.addEdge(map(int,splitted))
            line=file.readline().rstrip()
            splitted=line.split()
        return G

if __name__ == "__main__":

    I = input().split()
    if len(I) > 2:
        V, E = map(int, I[:2])
        G = LabeledGraph(V)
        for k in range(E):
            G.addEdge(map(int, I[3 * k + 2:3 * k + 5]))
    else:
        V, E = map(int, I)
        G = LabeledGraph(V)
        for k in range(E):
            G.addEdge(map(int, input().split()))
    strategy=map(int,input().split())
    print(counterStrategy(G, psi=list(strategy), method="floyd-warshall"))
