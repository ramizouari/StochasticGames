# This is a sample Python script.
from typing import List, Tuple
import graph as g
import random_graph as rg
import numpy as np
import extended_integer

rg.set_generation_method(np.random.MT19937(35))
d=lambda :rg.graph_generator.binomial(10,0.1)

print(rg.generate_labeled_graph(10,0.5,d=d,seed=0))
# Read the graph parameters
V,E=map(int,input().split())
graph = g.LabeledGraph(V)
# Read the edges
for _ in range(E):
    u,v,l=map(int,input().split())
    graph.addEdge((u,v,l))
# Read the strategy
S = list(map(int,input().split()))
print(g.counterStrategy(graph, S, method="floyd_warshall"))