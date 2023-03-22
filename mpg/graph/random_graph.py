from typing import Union, TypeVar, Type

import networkx as nx
import numpy as np

from ..games import mpg

graph_generator = np.random.Generator(np.random.MT19937())


def set_generation_method(method):
    global graph_generator
    graph_generator = np.random.Generator(method)
    return graph_generator


# Generate a random graph with n nodes and edge probability p
# The graph is guaranteed to be sinkless

GraphType = TypeVar("GraphType", bound=Union[nx.DiGraph, nx.Graph, mpg.MeanPayoffGraph])


def gnp_random_graph_sinkless(n: int, p: float, create_using: Type[GraphType] = nx.DiGraph, directed=None, seed=None,
                              method="fast", loops=True) -> GraphType:
    """
    Generate a random graph with n nodes and edge probability p
    The graph is guaranteed to be sinkless
    :param loops:
    :param n: The number of nodes
    :param p: The probability of an edge
    :param directed: If the graph is directed
    :param seed: The seed for the random number generator
    :param method: The method used to generate the graph. "normal" is the naive method, "fast" is a faster method
    :return: The generated graph
    """
    generator: np.random.Generator
    if seed is not None:
        generator = np.random.Generator(np.random.MT19937(seed))
    else:
        generator = graph_generator
    if directed is not None:
        create_using = nx.DiGraph if directed else nx.Graph
    G = create_using()
    match create_using:
        case nx.DiGraph | mpg.MeanPayoffGraph:
            directed = True
        case nx.Graph:
            directed = False
        case _:
            raise ValueError("Unknown graph type")

    G.add_nodes_from(range(n))
    match method:
        case "normal":
            for u in range(n):
                while len(G.succ[u]) == 0:
                    ran = range(n) if directed else range(u, n)
                    for v in ran:
                        if not loops and u == v:
                            continue
                        if generator.random() < p:
                            G.add_edge(u, v, weight=generator.random())
            return G
        case "fast":
            for u in range(n):
                while len(G.succ[u]) == 0:
                    # The degree of the node u is m
                    m = generator.binomial(n, p)
                    # Choose m nodes from the set {0,1,...,n-1} uniformly at random
                    A = generator.choice(n, m, replace=False)
                    for v in A:
                        # TODO: Check if this is correct. It should give the correct distribution
                        if not loops and u == v:
                            continue
                        G.add_edge(u, v, weight=generator.random())
            return G
        case _:
            raise ValueError("Unknown method")


def gnm_random_graph_sinkless(n: int, m: int, create_using: Type[GraphType] = nx.DiGraph, directed=None, seed=None,
                              method="fast", loops=True) -> GraphType:
    """
    Generate a random graph with n nodes and edge probability p
    The graph is guaranteed to be sinkless
    :param loops:
    :param n: The number of nodes
    :param m: The number of edges
    :param directed: If the graph is directed
    :param seed: The seed for the random number generator
    :param method: The method used to generate the graph. "normal" is the naive method, "fast" is a faster method
    :return: The generated graph
    """
    if m < n:
        raise ValueError("The number of edges must be at least n")

    generator: np.random.Generator
    if seed is not None:
        generator = np.random.Generator(np.random.MT19937(seed))
    else:
        generator = graph_generator
    if directed is not None:
        create_using = nx.DiGraph if directed else nx.Graph
    G = create_using()
    match create_using:
        case nx.DiGraph | mpg.MeanPayoffGraph:
            directed = True
            if not loops and m > n * (n - 1):
                raise ValueError("The number of edges for a directed graph must be at most n*(n-1)")
            elif loops and m > n * n:
                raise ValueError("The number of edges for a directed graph with loops must be at most n*n")
        case nx.Graph:
            directed = False
            if not loops and m > n * (n - 1) / 2:
                raise ValueError("The number of edges for an undirected graph must be at most n*(n-1)/2")
            elif loops and m > n * (n + 1) / 2:
                raise ValueError("The number of edges for an undirected graph with loops must be at most n*(n+1)/2")
        case _:
            raise ValueError("Unknown graph type")

    G.add_nodes_from(range(n))
    # Generate a set of edges
    S = set()
    for u in range(n):
        v = generator.integers(0, n)
        if not loops and u == v:
            while v == u:
                v = generator.integers(0, n)
        S.add((u, v))
        G.add_edge(u, v, weight=generator.random())
    match method:
        case "normal":

            E = [(u, v) for u in range(n) for v in range(n) if not loops and u == v and (u, v) not in S]
            E = generator.choice(E,  m , replace=False)

            for u, v in E:
                G.add_edge(u, v, weight=generator.random())
            return G
        case "fast":
            while len(S) < m:
                u = generator.integers(0, n)
                v = generator.integers(0, n)
                if not loops and u == v:
                    while v == u:
                        v = generator.integers(0, n)
                G.add_edge(u, v, weight=generator.random())
                S.add((u, v))
            return G
        case _:
            raise ValueError("Unknown method")


def set_weights(G: nx.DiGraph, distribution="normal", seed=None, **kwargs):
    generator: np.random.Generator
    if seed is not None:
        generator = np.random.Generator(np.random.MT19937(seed))
    else:
        generator = graph_generator
    if not hasattr(generator, distribution):
        raise ValueError("Unknown distribution")
    distribution = getattr(generator, distribution)
    for u in G.nodes:
        for v in G.succ[u]:
            G[u][v]["weight"] = distribution(**kwargs)
            if type(G) == mpg.MeanPayoffGraph:
                G[u][v]["label"] = G[u][v]["weight"]
    return G


def gnp_random_mpg(n, p, seed=None, method="fast", loops=True, distribution="normal", **kwargs) -> mpg.MeanPayoffGraph:
    """
    Generate a mean-payoff game with n nodes and edge probability p. The weights are generated according to the distribution
    The underlying graph is guaranteed to be sinkless
    :param n: The number of nodes
    :param p: The probability of an edge
    :param seed: The seed for the random number generator
    :param method: The method used to generate the graph. "normal" is the naive method, "fast" is a faster method
    :param loops: If loops are allowed
    :param distribution: The distribution used to generate the weights
    :param kwargs: The parameters of the distribution
    :return: The generated mean-payoff game
    """
    G = gnp_random_graph_sinkless(n, p, method=method, loops=loops, seed=seed, create_using=mpg.MeanPayoffGraph)
    set_weights(G, distribution=distribution, seed=seed, **kwargs)
    return G


def gnm_random_mpg(n: int, m: int, seed=None, method="fast", loops=True, distribution="normal",
                   **kwargs) -> mpg.MeanPayoffGraph:
    """
    Generate a mean-payoff game with n nodes and edge probability p. The weights are generated according to the distribution
    The underlying graph is guaranteed to be sinkless
    :param n: The number of nodes
    :param m: The number of edges
    :param seed: The seed for the random number generator
    :param method: The method used to generate the graph. "normal" is the naive method, "fast" is a faster method
    :param loops: If loops are allowed
    :param distribution: The distribution used to generate the weights
    :param kwargs: The parameters of the distribution
    :return: The generated mean-payoff game
    """
    G = gnm_random_graph_sinkless(n, m, method=method, loops=loops, seed=seed, create_using=mpg.MeanPayoffGraph)
    set_weights(G, distribution=distribution, seed=seed, **kwargs)
    return G


if __name__ == "__main__":
    print("HII")
    n = 1000
    G = gnm_random_mpg(n, 1000000, method="fast", distribution="normal", loc=0, scale=1)
    print(len(G.edges))
