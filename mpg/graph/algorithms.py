import queue
from typing import List, Tuple, Union, Dict, Any

import numpy as np

import networkx as nx

def floyd_warshall_negative_paths(graph:nx.DiGraph)-> Dict[Any,Any]:
    """
    Floyd-Warshall algorithm for finding negative cycles in a graph
    :param graph: The graph to be processed
    :return: dictionary of nodes with their successors. The successors will eventually lead to a negative cycle, if any.
    """
    # map nodes to integers
    mapper=dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    # initialize distance matrix
    D=np.full((graph.number_of_nodes(), graph.number_of_nodes()), np.inf)
    # fill distance matrix
    for i in range(graph.number_of_nodes()):
        D[i,i]=0
    for u in graph.nodes():
        for v in graph[u]:
            D[mapper[u],mapper[v]]=graph[u][v]['weight']
    # find paths with negative cycles
    successor={}
    for w in range(graph.number_of_nodes()):
        for u in range(graph.number_of_nodes()):
            for v in range(graph.number_of_nodes()):
                if D[u,w]==np.inf or D[w,v]==np.inf:
                    continue
                D[u,v]=np.minimum(D[u,v],D[u,w]+D[w,v])
                if u==v and D[u,w]+D[w,v]<0:
                    D[u,v]=-np.inf
                    D[u,w]=-np.inf
                    D[w,v]=-np.inf
                if D[u,v]==-np.inf and v in graph.succ[u] and not u in successor:
                    successor[u]=v
    # find short paths
    for u in graph.nodes:
        if u in successor:
            continue
        R=np.inf
        for v in graph.succ[u]:
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