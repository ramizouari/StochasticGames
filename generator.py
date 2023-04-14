import abc
import timeit
from concurrent.futures import  ProcessPoolExecutor
from datetime import time,datetime
from typing import List, Callable, TypedDict, Dict

from mpg.games.mpg import MeanPayoffGraph
import pandas as pd
import numpy as np
from mpg.graph import random_graph as rg
from mpg import mpgio


class Callback(abc.ABC):
    """
    Abstract class for callbacks
    A callback is a function that is called at a certain point in the execution of a function
    """
    def __call__(self,*args,**kwargs):
        pass

class StepCallback(Callback):
    """
    A callback that is called every step iterations
    """
    def __init__(self,func:Callable,step:int=1):
        self.step=step
        self.func=func
        self.count=0
    def __call__(self,*args,**kwargs):
        self.count+=1
        if self.count%self.step==0:
            self.func(*args,**kwargs)

class ProgressCallback(StepCallback):
    """
    A callback that prints the progress of the execution of a function
    """
    def __init__(self,step:int=1):
        super().__init__(self.print_progress,step=step)

    def print_progress(self,iteration:int,n:int,p:float,execution_time:float,**kwargs):
        print(f"Iteration {iteration} of {n} nodes with edge probability {p:.3f} took {execution_time:.6f} seconds")
        print("Additional arguments:")
        for key,value in kwargs.items():
            print(f"\t{key}={value}")
        print("-"*80)
class SaveGraphCallback(StepCallback):
    """
    A callback that saves the graph to a file
    """
    def __init__(self,directory:str,save_as:str,compression:bool=False):
        super().__init__(self.save)
        self.directory=directory
        self.save_as=save_as
        self.compression=compression

    def save(self,iteration:int,n:int,c:int,p:float,graph:MeanPayoffGraph,row:Dict,**kwargs):
        filename=f"gnp_uniform_mpg_{n}_{c}_{iteration}"
        mpgio.save_mpg(f"{self.directory}/{filename}",graph,save_as=self.save_as,compression=self.compression)
        row["filename"]=filename


def _generate_gnp_uniform_mpg_instance(benchmark,iteration,n,p=None,c=None,a=-1,b=1,callbacks:List[Callback]=None) -> MeanPayoffGraph:
    p = min(p, 1)
    start = timeit.default_timer()
    execution_start_time = datetime.now()
    graph = rg.gnp_random_mpg(n, p, distribution="integers", low=a, high=b, endpoint=True)
    end = timeit.default_timer()
    execution_end_time = datetime.now()
    row = {"n": n, "c": c, "p": p,
           "execution_start": execution_start_time,
           "execution_end": execution_end_time, "time": end - start,
           "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
           "distribution": f"integers({a},{b})"}
    for callback in callbacks:
        callback(n=n, c=c, p=p, graph=graph, execution_start_time=execution_start_time,
                 execution_end_time=execution_end_time,
                 execution_time=end - start, iteration=iteration, row=row)
    benchmark.append(row)
    return graph



def generate_gnp_uniform_mpg(N,P=None,C=None,a=-1,b=1,iterations=10,seed=932,loops=True,callbacks:List[Callback]=None,threads:int=None) -> pd.DataFrame:
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
    # Check that either P or C is specified
    if P is None and C is None:
        raise ValueError("Either P or C must be specified")
    # Check that only one of P or C is specified
    if P is not None and C is not None:
        raise ValueError("Exactly one of P or C must be specified")
    # Check that P is in the range [0,1]
    if P is not None:
        if np.any(P<0) or np.any(P>1):
            raise ValueError("P must be in the range [0,1]")
    # Check that C > 0
    if C is not None:
        if np.any(C<=0):
            raise ValueError("C must be positive")
    # Check that a < b
    if a >= b:
        raise ValueError("a must be less than b")
    # Check that N is positive
    if np.any(N<=0):
        raise ValueError("N must be positive")
    # Check that iterations is positive
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    # Set the seed
    rg.set_generation_method(np.random.MT19937(seed))
    # Create the benchmark dataframe
    benchmark:List[Dict]=[]
    if callbacks is None:
        callbacks=[]
    use_threads = threads is not None and threads > 1
    use_probabilities = P is not None
    with ProcessPoolExecutor(max_workers=threads) as executor:
        for n in N:
            if use_probabilities:
                C = np.round(P * n).astype(int)
            else:
                P = C / n
            for c,p in zip(C,P):
                for i in range(iterations):
                    p=min(p,1)
                    if use_threads:
                            executor.submit(_generate_gnp_uniform_mpg_instance,benchmark,iteration=i,n=n,p=p,c=c,a=a,b=b,callbacks=callbacks)
                    else:
                        _generate_gnp_uniform_mpg_instance(benchmark,iteration=i,n=n,p=p,c=c,a=a,b=b,callbacks=callbacks)
    return pd.DataFrame(benchmark)

if __name__=="__main__":
    import argparse
    import os
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="dataset",help="The dataset to benchmark")
    parser.add_argument("--directory",type=str,required=True,help="The directory to save the graphs")
    parser.add_argument("--a",type=int,default=-1,help="The low value of the discrete uniform distribution")
    parser.add_argument("--b",type=int,default=1,help="The high value of the discrete uniform distribution")
    parser.add_argument("--iterations",type=int,default=10,help="The number of times to repeat the benchmark")
    parser.add_argument("--seed",type=int,default=932,help="The seed for the random number generator")
    parser.add_argument("--loops",action="store_true",help="Whether to include loops in the graph")
    parser.add_argument("--save_as",type=str,default="weighted_edgelist",help="The format to save the graph in")
    parser.add_argument("--compression",action="store_true",default=True,help="Whether to compress the graph when saving")
    parser.add_argument("--output",type=str,default="benchmark_gnp_random_mpg.csv",help="The output file")
    parser.add_argument("--N",type=int,nargs="+",required=True,help="The number of nodes in the graph")
    parser.add_argument("--C",type=int,nargs="+",help="The expected number of edges for each node")
    parser.add_argument("--P",type=float,nargs="+",help="The probability of an edge existing")
    parser.add_argument("--threads",type=int,help="The number of threads to use")
    args=parser.parse_args()
    N=np.array(args.N)
    P=None
    C=None
    if args.C is not None:
        C=np.array(args.C)
    if args.P is not None:
        P=np.array(args.P)
    callbacks=[ProgressCallback(), SaveGraphCallback(os.path.join(args.directory,args.dataset), args.save_as, compression=args.compression)]
    os.makedirs(os.path.join(args.directory,args.dataset),exist_ok=True)
    benchmark=generate_gnp_uniform_mpg(N=N,P=P,C=C,iterations=args.iterations,seed=args.seed,callbacks=callbacks,a=args.a,b=args.b,threads=args.threads)
    benchmark.to_csv(os.path.join(args.directory,args.output),index=False)
