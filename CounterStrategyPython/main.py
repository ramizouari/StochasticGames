# This is a sample Python script.
from typing import List, Tuple
import graph as g
import random_graph as rg
import numpy as np
import extended_integer

print(rg.generate_labeled_graph(10,0.5,d=lambda :np.random.binomial(10,0.1),seed=0))
# Read the graph parameters
V,E=map(int,input().split())
graph = g.LabeledGraph(V)
# Read the edges
for _ in range(E):
    u,v,l=map(int,input().split())
    graph.addEdge((u,v,l))
# Read the strategy
S = list(map(int,input().split()))
print(g.counterStrategyBellmanFord(graph, S,method="experimental"))