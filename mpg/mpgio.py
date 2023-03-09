import pickle
from typing import Type, Union

import networkx as nx
import numpy as np

from games import mpg

def get_compression(compression:Union[None,bool,str]) -> Union[str,None]:
    """
    Get the compression type
    :param compression: The compression type. If True, use "gz". If False, use None. If str, use the str value
    :return: The compression type
    """
    match compression:
        case True:
            return "gz"
        case False | None:
            return None
        case str():
            if compression not in ["gz","bz2"]:
                raise ValueError(f"Unknown compression value: {compression}")
            return compression

def save_mpg(path:str, graph:mpg.MeanPayoffGraph, save_as:str= "weighted_edgelist", compression:Union[bool,str]=False,delimeter=" ") -> None:
    """
    Save the graph to a file
    :param path: The path to save the MPG
    :param graph: The Mean Pay-off game to save
    :param save_as: The format to save the graph as
    :param compression: The compression type. See get_compression for more details
    :return:
    """
    match save_as:
        case "pickle":
            pickle.dump(graph, open(f"{path}.gpickle", "wb"))
        case "graphml":
            nx.write_graphml(graph,f"{path}.graphml")
        case "weighted_edgelist":
            compression=get_compression(compression)
            if compression is not None:
                nx.write_weighted_edgelist(graph,f"{path}.edgelist.{compression}")
            else:
                nx.write_weighted_edgelist(graph,f"{path}.edgelist")
        case "adjlist":
            compression=get_compression(compression)
            if compression is not None:
                nx.write_adjlist(graph,f"{path}.adjlist.{compression}")
            else:
                nx.write_adjlist(graph,f"{path}.adjlist")
        case "gml":
            nx.write_gml(graph,f"{path}.gml")
        case "gexf":
            nx.write_gexf(graph,f"{path}.gexf")
        case "graph6":
            nx.write_graph6(graph,f"{path}.graph6")
        case "sparse6":
            nx.write_sparse6(graph,f"{path}.sparse6")
        case "pajek":
            nx.write_pajek(graph,f"{path}.pajek")
        case _:
            raise ValueError(f"Unknown save_as value: {save_as}")


def convert_edges_to(graph:Type[nx.DiGraph],edgetype) -> Type[nx.DiGraph]:
    """
    Convert the edges to the given type and return it.
    :param graph: The graph to convert
    :param edgetype: The type to convert to
    :return: The converted graph
    """
    for e in graph.edges:
        for u in graph.edges[e]:
            graph.edges[e][u] = edgetype(graph.edges[e][u])
    return graph

def read_mpg(path:str, delimiter:str=" ",nodetype:type=int,edgetype:type=int) -> mpg.MeanPayoffGraph:
    """
    Read a graph from a csv file
    :param path: The path to the file
    :param delimiter: The delimiter to use
    :return: The mean pay-off game.
    """
    S=path.split(".")
    if len(S) == 0:
        raise ValueError(f"Unknown file extension: {path}")
    match S[-1]:
        case "gpickle":
            return pickle.load(open(path, "rb"))
        case "graphml":
            return mpg.mpg_from_digraph(nx.read_graphml(path))
        case "gz" | "bz2":
            if len(S) == 1:
                raise ValueError(f"Unknown file extension: {path}")
            match S[-2]:
                case "edgelist":
                    graph:mpg.MeanPayoffGraph=nx.read_weighted_edgelist(path,delimiter=delimiter,create_using=mpg.MeanPayoffGraph,nodetype=nodetype)
                    return convert_edges_to(graph,edgetype)
        case "edgelist":
            graph:mpg.MeanPayoffGraph=nx.read_weighted_edgelist(path,delimiter=delimiter,create_using=mpg.MeanPayoffGraph,nodetype=nodetype)
            return convert_edges_to(graph,edgetype)
        case "adjlist":
            return nx.read_adjlist(path,delimiter=delimiter,create_using=mpg.MeanPayoffGraph)
        case "adjlist.gz":
            return nx.read_adjlist(path,delimiter=delimiter,create_using=mpg.MeanPayoffGraph)
        case "gml":
            return mpg.mpg_from_digraph(nx.read_gml(path))
        case _ as ext:
            raise ValueError(f"Unknown file extension: {ext}")

if __name__=="__main__":
    graph=nx.erdos_renyi_graph(100, 0.1,directed=True)
    for u,v in graph.edges():
        graph.edges[u,v]["weight"]=np.random.randint(0,100)
    graph = mpg.mpg_from_digraph(graph)
    save_mpg("test",graph,save_as="weighted_edgelist",compression=True)
    graph2=read_mpg("test.edgelist.gz")
    print(graph2.edges(data=True))
