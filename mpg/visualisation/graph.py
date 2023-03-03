import networkx as nx
from yfiles_jupyter_graphs import GraphWidget
from typing import TypedDict, Any, List, Dict


class NodePropertiesMetadata(TypedDict):
    label: Any


class NodeMetadata(TypedDict):
    id: Any
    properties: NodePropertiesMetadata


class EdgePropertiesMetadata(TypedDict):
    label: Any


class EdgeMetadata(TypedDict):
    id: int
    start: Any
    end: Any
    properties: EdgePropertiesMetadata


class GraphVisualisation(GraphWidget):

    def _node_metadata(self,u:Any)-> NodeMetadata:
        return {"id": u, "properties": {"label": u}}

    def  _edge_metadata(self,id,u,v,weight) -> EdgeMetadata:
        return {"id": id,"start":u,"end":v,"properties":{"label":weight}}
    def __init__(self, graph: nx.DiGraph):
        super().__init__(graph=graph)
        self.graph = graph
        self.nodes: List[NodeMetadata] = [self._node_metadata(u) for u in self.graph.nodes]
        self.edges: List[EdgeMetadata] = [self._edge_metadata(k,U[0],U[1],weight=self.graph.edges[U[0],U[1]]["weight"]) for k,U in enumerate(self.graph.edges)]

    def add_edge(self, u, v, weight, **attrs):
        metadata = attrs.copy()
        metadata.update(weight=weight)
        for w in [u,v]:
            if w not in self.graph.nodes:
                self.add_node(w)
        self.edges.append(self._edge_metadata(len(self.graph.edges),u,v,weight))
        self.graph.add_edge(u, v, weight=weight, **attrs)

    def add_node(self, u, **attrs):
        self.graph.add_node(u, **attrs)
        self.nodes.append(self._node_metadata(u))
