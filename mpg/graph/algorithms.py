import queue
from typing import List, Tuple, Union

import numpy as np

from ds import union_find
from algbera.extended_integer import TropicalInteger
import networkx as nx

def floyd_warshall_negative_paths(G:nx.DiGraph):
    # map nodes to integers
    mapper=dict(zip(G.nodes(),range(G.number_of_nodes())))
    # initialize distance matrix
    D=np.full((G.number_of_nodes(),G.number_of_nodes()),np.inf)
    # fill distance matrix
    for i in range(G.number_of_nodes()):
        D[i,i]=0
    for u in G.nodes():
        for v in G[u]:
            D[mapper[u],mapper[v]]=G[u][v]['weight']
    # find paths with negative cycles
    successor={}
    for w in range(G.number_of_nodes()):
        for u in range(G.number_of_nodes()):
            for v in range(G.number_of_nodes()):
                if D[u,w]==np.inf or D[w,v]==np.inf:
                    continue
                D[u,v]=np.minimum(D[u,v],D[u,w]+D[w,v])
                if u==v and D[u,w]+D[w,v]<0:
                    D[u,v]=-np.inf
                    D[u,w]=-np.inf
                    D[w,v]=-np.inf
                if D[u,v]==-np.inf and v in G.succ[u] and not u in successor:
                    successor[u]=v
    # find short paths
    for u in G.nodes:
        if u in successor:
            continue
        R=np.inf
        for v in G.succ[u]:
            if D[mapper[u],mapper[v]]<R:
                R=D[mapper[u],mapper[v]]
                successor[u]=v
    return successor



"""
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
"""