# This is a sample Python script.
from typing import Tuple, List

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

# Class to represent an Extended Integer
# An extended integer is an integer element or ±∞, with the addition operation extended to ±∞
# and the multiplication operation extended to ±∞
class ExtendedInteger:
    threshold: int = 1000000

    def __init__(self, value):
        if np.isnan(value):
            self.value = 0
        if value > self.threshold:
            self.value = np.inf
        elif value < -self.threshold:
            self.value = -np.inf
        else:
            self.value = value

    def __add__(self, other):
        return ExtendedInteger(self.value + other.value)

    def __sub__(self, other):
        return ExtendedInteger(self.value - other.value)

    def __mul__(self, other):
        return ExtendedInteger(self.value * other.value)

    def __radd__(self, other):
        return ExtendedInteger(self.value + other)

    def __rsub__(self, other):
        return ExtendedInteger(self.value - other)

    def __rmul__(self, other):
        return ExtendedInteger(self.value * other)

    def __repr__(self):
        return f"ExtendedInteger({self.value})"

    def __eq__(self, other):
        if isinstance(other, ExtendedInteger):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, float):
            return self.value == other
        else:
            return False

    def __pow__(self, power, modulo=None):
        return ExtendedInteger(self.value ** power)

# Class to represent a Tropical Integer
# A tropical integer is an element of the min tropical semi-ring, with the addition operation being the min operation
# and the multiplication operation being the addition operation. The zero element is +inf, and the one element is 0.
# The -inf element is not part of the semi-ring. But it may be used in calculations
class TropicalInteger:
    threshold: int = 10000000

    def __init__(self, value):
        if np.isnan(value):
            self.value=np.inf
        elif value > self.threshold:
            self.value = np.inf
        elif value < -self.threshold:
            self.value = -np.inf
        else:
            self.value = value

    def __add__(self, other):
        if isinstance(other, TropicalInteger):
            return TropicalInteger(min(self.value, other.value))
        else:
            return TropicalInteger(min(self.value, other))

    def __mul__(self, other):
        if isinstance(other, TropicalInteger):
            return TropicalInteger(self.value + other.value)
        else:
            return TropicalInteger(self.value + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"TropicalInteger({self.value})"

    def __eq__(self, other):
        if isinstance(other, TropicalInteger):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, float):
            return self.value == other
        else:
            return False

    def __pow__(self, power, modulo=None):
        return TropicalInteger(power*self.value)

# The class to represent a labeled graph, with edges labeled by integers
class LabeledGraph:
    def __init__(self,V,E):
        self.V = V
        self.E = E
        self.adjacencyList:List[List[Tuple[int,TropicalInteger]]] = [[] for _ in range(V)]
        self.reverseList:List[List[Tuple[int,TropicalInteger]]]= [[] for i in range(V)]
    def addEdge(self,edge):
        (u,v,l) = edge
        self.adjacencyList[u].append((v,l))
        self.reverseList[v].append((u,l))
    pass


def counterStrategy1(graph: LabeledGraph, S:List[int],iters:int=None) -> List[int]:
    strategyCost:List[TropicalInteger] = [TropicalInteger(0) for _ in range(graph.V)]
    for u in range(graph.V):
        for (v,l) in graph.adjacencyList[u]:
            if v==S[u]:
                strategyCost[u]=l

    L:List[List[TropicalInteger]]= [[TropicalInteger(np.inf) for _ in range(graph.V)] for _ in range(graph.V)]
    for u in range(graph.V):
        for (v,l) in graph.adjacencyList[S[u]]:
            L[u][v]= TropicalInteger(l+strategyCost[u])
    ## TODO: Implement the algorithm
    f=np.array([TropicalInteger(0) for _ in range(graph.V)])
    return f

def counterStrategy23(graph: LabeledGraph, S:List[int],iters:int=None) -> List[int]:
    strategyCost:List[TropicalInteger] = [TropicalInteger(0) for _ in range(graph.V)]
    for u in range(graph.V):
        for (v,l) in graph.adjacencyList[u]:
            if v==S[u]:
                strategyCost[u]=l

    L:List[List[TropicalInteger]]= [[TropicalInteger(np.inf) for _ in range(graph.V)] for _ in range(graph.V)]
    for u in range(graph.V):
        for (v,l) in graph.adjacencyList[S[u]]:
            L[u][v]= TropicalInteger(l+strategyCost[u])
    M=np.array(L)
    f=np.array([TropicalInteger(0) for _ in range(graph.V)])
    if iters is None:
        while (M@M@f != M@f).any():
            M = M@M
    else:
        M=np.linalg.matrix_power(M,iters)
    return M@f

# Read the graph parameters
V,E=map(int,input().split())
graph = LabeledGraph(V,E)
# Read the edges
for _ in range(E):
    u,v,l=map(int,input().split())
    graph.addEdge((u,v,l))
# Read the strategy
S = list(map(int,input().split()))
print(counterStrategy23(graph,S))