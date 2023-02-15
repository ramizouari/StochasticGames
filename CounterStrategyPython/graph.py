import queue
from typing import List, Tuple, Callable

import numpy as np

import union_find
from extended_integer import TropicalInteger

class Graph:
    def __init__(self,V):
        self.V=V
        self.E=0
        self.adjacencyList: List[List[int]] = [[] for _ in range(V)]
        self.reverseList: List[List[int]] = [[] for i in range(V)]
    def addEdge(self,edge):
        (u,v)=edge
        self.adjacencyList[u].append(v)
        self.adjacencyList[v].append(u)
        self.E+=1

    def _topological_sort_rec(self,r:int,L:List[int],visited:List[bool]):
        if not visited[r]:
            visited[r] = True
            for s in self.adjacencyList[r]:
                self._topological_sort_rec(s, L, visited)
            L.append(r)
            pass
    def _assign_component(self,u,v,C:union_find.UnionFind,isAssigned:List[bool]):
        if not isAssigned[u]:
            isAssigned[u]=True
            C.connect(u,v)
            for s in self.reverseList[u]:
                self._assign_component(s,v,C,isAssigned)
        pass
    def topologicalSort(self,source=0):
        L=[]
        visited=[False for u in range(self.V)]
        for u in range(self.V):
            self._topological_sort_rec(u,L,visited)
        L.reverse()
        return L
    def stronglyConnectedComponents(self):
        C=union_find.UnionFind(self.V)
        isAssigned=[False for u in range(self.V)]
        topologicalOrder=[]
        for l in self.topologicalSort():
            self._assign_component(l,l,C,isAssigned)
            if l==C.representative(l):
                topologicalOrder.append(l)

        idAssigned = [False for k in range(self.V)]
        componentId = [u for u in range(self.V)]
        components=[]
        for u in range(self.V):
            r=C.representative(u)
            if not idAssigned[r]:
                idAssigned[r]=True
                componentId[r]=len(components)
                components.append([])
            componentId[u]=componentId[r]
            components[componentId[r]].append(u)
        pass
        return components,componentId,C,topologicalOrder

"""
std::vector<bool> componentAssigned(n);
        UnionFind C(n);
        std::vector<int> topologicalOrder;
        for(auto l: topologicalSort())
        {
            assignComponents(l, l, C, componentAssigned);
            if(l==C.representative(l))
                topologicalOrder.push_back(l);
        }
        std::vector<bool> idAssigned(n);
        std::vector<int> componentId(n);
        std::vector<std::vector<int>> components;
        for(int i=0;i<n;i++)
        {
            auto r=C.representative(i);
            if(!idAssigned[r])
            {
                idAssigned[r]=true;
                componentId[r]=components.size();
                components.emplace_back();
            }
            componentId[i]=componentId[r];
            components[componentId[r]].push_back(i);
        }
        return ConnectedComponentMetaData(std::move(components),std::move(componentId),std::move(C),std::move(topologicalOrder));
"""
class LabeledGraph:
    def __init__(self, V):
        self.V = V
        self.E=0
        self.adjacencyList: List[List[Tuple[int, TropicalInteger]]] = [[] for _ in range(V)]
        self.reverseList: List[List[Tuple[int, TropicalInteger]]] = [[] for i in range(V)]

    def addEdge(self, edge):
        (u, v, l) = edge
        self.adjacencyList[u].append((v, l))
        self.reverseList[v].append((u, l))
        self.E+=1

    pass

    def adjacencyMatrix(self):
        M = np.zeros((self.V, self.V))
        for u in range(self.V):
            for v, L in self.adjacencyList[u]:
                M[u, v] = L

    def __repr__(self):
        return f"""Labeled Graph with {self.V} vertices and {self.E} edges:
{
{ u:list(map(lambda s: {"dest":s[0],"weight":s[1]},self.adjacencyList[u])) for u in range(self.V) }
}
"""


def MeanPayoffGraph(LabeledGraph):
    def __init__(self, V):
        super().__init__(V)

    def closure(self):
        for u in range(self.V):
            if len(self.adjacencyList[u]) == 0:
                self.addEdge(u, u, 0)

def FloydWarshallAlgorithm(G:LabeledGraph,retCycles=True):
    D = np.full((G.V, G.V), np.inf)
    for i in range(G.V):
        D[i, i] = 0
    for u in range(G.V):
        for v,L in G.adjacencyList[u]:
            D[u,v]=L
    cycles = []
    ancestor = [i for i in range(G.V)]
    C = -1
    for w in range(G.V):
        for u in range(G.V):
            for v in range(G.V):
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

def BellmanFordAlgorithm(G:LabeledGraph,source=0,retCycles=True):
    d=np.full(np.inf,G.V)
    d[source]=0
    for v in range(G.V-1):
        visited=[False for v in range(G.V)]
        visited[source]=True
        Q = queue.Queue()
        Q.put(source)
        while not Q.empty():
            s=Q.get()
            for v,L in G.adjacencyList[s]:
                if not visited[s]:
                    d[v]=np.minimum(d[s]+L,d[v])
                    visited[v]=False
    Q=queue.Queue()
    Q.put(source)
    visited = [False for v in range(G.V)]
    parent = [u for u in range(G.V)]
    while not Q.empty():
        s = Q.get()
        for v, L in G.adjacencyList[s]:
            if not visited[s]:
                visited[s]=True
                if d[s] + L < d[v]:
                    parent[v]=s


# This is a suboptimal method, running in O(n²m), But it is used for now for negative cycles construction
def ExperimentalMethod(G: LabeledGraph, retCycles=True):
    D = np.full((G.V, G.V), np.inf)
    for i in range(G.V):
        D[i, i] = 0
    cycles = []
    ancestor=[i for i in range(G.V)]
    C=-1
    for k in range(G.V):
        for u in range(G.V):
            for v in range(G.V):
                for w, L in G.adjacencyList[v]:
                    if D[u, w]>  D[u, v] + L:
                        D[u,w]=D[u,v]+L
                        ancestor[w]=v
    for u in range(G.V):
        for v in range(G.V):
            for w, L in G.adjacencyList[v]:
                if D[u, v] + L < D[u, w]:
                    C=v
    k=ancestor[C]
    cycles.append(C)
    while k!=C:
        cycles.append(k)
        k=ancestor[k]
    cycles.reverse()
    if retCycles:
        return D, cycles
    return D


def counterStrategyBellmanFord(G:LabeledGraph, psi:List[int],method="floyd-warshall"):
    strategyCost: List[TropicalInteger] = [TropicalInteger(0) for _ in range(G.V)]
    for u in range(G.V):
        for (v, l) in G.adjacencyList[u]:
            if v == psi[u]:
                strategyCost[u] = l
    #The one-player equivalent game
    G1=LabeledGraph(G.V)
    for u in range(G.V):
        for v,L in G.adjacencyList[psi[u]]:
            G1.addEdge((u,v,L+strategyCost[u]))
    match method:
        case "floyd-warshall" | "bellman-ford":
            raise NotImplementedError(f"Method {method} is not implemented yet")
        case "experimental":
            _,C =ExperimentalMethod(G1,True)
            n=len(C)
            C.append(C[0])
            R=0
            for k in range(n):
                for v,L in G1.adjacencyList[C[k]]:
                    if v==C[k+1]:
                        R+=L
            R/=n
            return C,R

        case _:
            raise NotImplementedError(f"Method {method} is not implemented yet")


if __name__ == "__main__":
    I=input().split()
    if len (I) > 2:
        V,E=map(int,I[:2])
        G = LabeledGraph(V)
        for k in range(E):
            G.addEdge(map(int,I[3*k+2:3*k+5]))
    else:
        V, E = map(int, I)
        G = LabeledGraph(V, E)
        for k in range(E):
            G.addEdge(map(int, input().split()))

    print(ExperimentalMethod(G, True))
