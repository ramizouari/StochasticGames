import timeit
from datetime import time,datetime
import networkx as nx
import pickle

import games.mpg as mpg
import pandas as pd
import numpy as np
from graph import random_graph as rg

def generate_gnp_uniform_mpg(N,C,directory:str=None,save_as="pickle",a=-1,b=1,iterations=10,seed=932) -> pd.DataFrame:
    """
    Benchmark the mpg.gnp_random_mpg function
    :param N: An array of values, each value is the number of nodes in the graph
    :param C: An array of values, each value is the expected number of edges for each node
    :param directory: The directory to save the graphs
    :param a: The low value of the discrete uniform distribution
    :param b: The high value of the discrete uniform distribution
    :param iterations: The number of times to repeat the benchmark
    :param seed: The seed for the random number generator
    :return: A pandas dataframe containing the benchmark results
    """
    rg.set_generation_method(np.random.MT19937(seed))
    benchmark=pd.DataFrame(columns=["n","c","p","start","end","time"])
    for i in range(iterations):
        for n in N:
            for c in C:
                p=c/n
                start=timeit.default_timer()
                execution_start_time=datetime.now()
                G=rg.gnp_random_mpg(n,p,distribution="integers",low=a,high=b)
                end=timeit.default_timer()
                execution_end_time=datetime.now()
                if directory is not None:
                    match save_as:
                        case "pickle":
                            pickle.dump(G,open(f"{directory}/gnp_uniform_mpg_{n}_{c}_{i}.gpickle","wb"))
                        case "graphml":
                            nx.write_graphml(G,f"{directory}/gnp_uniform_mpg_{n}_{c}_{i}.graphml")
                        case _:
                            raise ValueError("Unknown save_as value")
                benchmark=pd.concat([benchmark,pd.DataFrame({"n":[n],"c":[c],"p":[p],
                                                             "execution_start":[execution_start_time],
                                                             "execution_end":[execution_end_time],"time":[end-start]})])
    return benchmark

if __name__=="__main__":
    N=np.arange(10,100)**2
    C=np.arange(2,11)
    seed=932
    benchmark=generate_gnp_uniform_mpg(N,C,"data/generated",iterations=10,seed=seed)
    benchmark.to_csv("benchmark_gnp_random_mpg.csv")