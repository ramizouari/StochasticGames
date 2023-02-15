import itertools
import random
from typing import Union
import numpy as np

import graph

graph_generator=np.random.Generator(np.random.MT19937())

def set_generation_method(method):
    global graph_generator
    graph_generator = np.random.Generator(method)
    return graph_generator

# Generate a labeled graph given the number of vertices
def generate_labeled_graph(V:int,E:Union[int,float],d ,seed=None,loops=False,generator:np.random.Generator=None):
    if seed is None:
        seed=0

    if generator is None:
        generator=graph_generator

    G=graph.LabeledGraph(V)
    match E:
        # When E is an integer, it is the number of edges
        case int(E):
            L=[]
            for (u,v) in itertools.product(range(V),range(V)):
                if u==v and loops:
                    continue
                L.append((u,v))
            generator.shuffle(L)
            for k in range(E):
                G.addEdge((*L[k],d()))
            pass
        # When E is a float, it is the probability of an edge
        case float(E):
            p=E
            for (u,v) in itertools.product(range(V),range(V)):
                if u==v and loops:
                    continue
                if graph_generator.binomial(1,p) > 0:
                    G.addEdge((u,v,d()))
            pass
        case _:
            raise TypeError(f"E should be an integer or a floating point")
    return G