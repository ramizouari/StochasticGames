import itertools
import random
from typing import Union
import numpy as np

import graph

# Generate a labeled graph given the number of vertices
def generate_labeled_graph(V:int,E:Union[int,float],d ,seed=None,loops=False):
    if seed is None:
        seed=0

    G=graph.LabeledGraph(V)
    match E:
        case int(E):
            L=[]
            for (u,v) in itertools.product(range(V),range(V)):
                if u==v and loops:
                    continue
                L.append((u,v))
            random.shuffle(L)
            for k in range(E):
                G.addEdge((*L[k],d()))
            pass
        case float(E):
            p=E
            for (u,v) in itertools.product(range(V),range(V)):
                if u==v and loops:
                    continue
                if np.random.binomial(1,p) > 0:
                    G.addEdge((u,v,d()))
            pass
        case _:
            raise TypeError(f"E should be an integer or a floating point")
    return G