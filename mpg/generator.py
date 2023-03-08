import abc
import timeit
from datetime import time,datetime
from typing import List, Callable

import networkx as nx
import pickle

import games.mpg as mpg
import pandas as pd
import numpy as np
from graph import random_graph as rg


class Callback(abc.ABC):
    def __call__(self,*args,**kwargs):
        pass

class StepCallback(Callback):
    def __init__(self,func:Callable,step:int=1):
        self.step=step
        self.func=func
        self.count=0
    def __call__(self,*args,**kwargs):
        self.count+=1
        if self.count%self.step==0:
            self.func(*args,**kwargs)

class PrintCallback(StepCallback):
    def __init__(self,step:int):
        super().__init__(print,step)

    def __call__(self,*args,**kwargs):
        super().__call__(*args,**kwargs)

class ProgressCallback(StepCallback):
    def __init__(self,step:int=1):
        super().__init__(self.print_progress,step=step)

    def print_progress(self,iteration:int,n:int,p:float,graph:mpg.MeanPayoffGraph,execution_start_time:datetime,execution_end_time:datetime,execution_time:float,
                       **kwargs):
        print(f"Iteration {iteration} of {n} nodes with edge probability {p:.3f} took {execution_time:.6f} seconds")
        print("Additional arguments:")
        for key,value in kwargs.items():
            print(f"\t{key}={value}")
        print("-"*80)
class SaveCallback(StepCallback):
    def __init__(self,directory:str,save_as:str):
        super().__init__(self.save)
        self.directory=directory
        self.save_as=save_as

    def save(self,iteration:int,n:int,c:int,p:float,graph:mpg.MeanPayoffGraph,
             execution_start_time:datetime,execution_end_time:datetime,execution_time:float):
        match self.save_as:
            case "pickle":
                pickle.dump(graph,open(f"{self.directory}/gnp_uniform_mpg_{n}_{c}_{iteration}.gpickle","wb"))
            case "graphml":
                nx.write_graphml(graph,f"{self.directory}/gnp_uniform_mpg_{n}_{c}_{iteration}.graphml")
            case _:
                raise ValueError("Unknown save_as value")

def generate_gnp_uniform_mpg(N,C,a=-1,b=1,iterations=10,seed=932,callbacks:List[Callback]=None) -> pd.DataFrame:
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
    benchmark=pd.DataFrame(columns=["n","c","p","execution_start","execution_end","time"])
    if callbacks is None:
        callbacks=[]
    for i in range(iterations):
        for n in N:
            for c in C:
                p=c/n
                start=timeit.default_timer()
                execution_start_time=datetime.now()
                graph=rg.gnp_random_mpg(n,p,distribution="integers",low=a,high=b)
                end=timeit.default_timer()
                execution_end_time=datetime.now()
                for callback in callbacks:
                    callback(n=n,c=c,p=p,graph=graph,execution_start_time=execution_start_time,
                             execution_end_time=execution_end_time,
                             execution_time=end-start,iteration=i)
                benchmark=pd.concat([benchmark,pd.DataFrame({"n":[n],"c":[c],"p":[p],
                                                             "execution_start":[execution_start_time],
                                                             "execution_end":[execution_end_time],"time":[end-start]})])
    return benchmark

if __name__=="__main__":
    N=np.arange(10,12)**2
    C=np.arange(2,11)
    seed=932
    callbacks=[ProgressCallback(),SaveCallback("data/generated","pickle")]
    benchmark=generate_gnp_uniform_mpg(N,C,iterations=10,seed=seed,callbacks=callbacks)
    benchmark.to_csv("benchmark_gnp_random_mpg.csv",index=False)