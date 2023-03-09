import networkx as nx
from matplotlib import pyplot as plt
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


def _node_metadata(u: Any) -> NodeMetadata:
    return {"id": u, "properties": {"label": u}}


def _edge_metadata(id, u, v, weight) -> EdgeMetadata:
    return {"id": id, "start": u, "end": v, "properties": {"label": weight}}


class AbstractGraphGraphics:
    """
    An abstract class for graph visualisation. It is used to define the interface for graph visualisation
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.nodes: List[NodeMetadata] = [_node_metadata(u) for k, u in enumerate(self.graph.nodes)]
        self.edges: List[EdgeMetadata] = [_edge_metadata(k, U[0], U[1], weight=self.graph.edges[U[0], U[1]]["weight"])
                                          for k, U in enumerate(self.graph.edges)]

    def add_edge(self, u, v, weight, **attrs):
        metadata = attrs.copy()
        metadata.update(weight=weight)
        for w in [u, v]:
            if w not in self.graph.nodes:
                self.add_node(w)
        self.edges.append(_edge_metadata(len(self.graph.edges), u, v, weight))
        self.graph.add_edge(u, v, weight=weight, **attrs)

    def add_node(self, u, **attrs):
        self.graph.add_node(u, **attrs)
        self.nodes.append(_node_metadata(u))


class GraphVisualisation(GraphWidget, AbstractGraphGraphics):
    """
    A class for visualising graphs using yfiles
    """

    def __init__(self, graph: nx.DiGraph):
        GraphWidget.__init__(self, graph=graph)
        AbstractGraphGraphics.__init__(self, graph=graph)


class GraphPlot(AbstractGraphGraphics):
    """
    A class for plotting graphs using networkx
    """

    def __init__(self, graph: nx.DiGraph, layout=None):
        super().__init__(graph=graph)
        self.kwargs = {"node_size": 1000, "node_color": "white", "edgecolors": "black"}
        if layout is None:
            layout = nx.spring_layout
        self.layout = layout

    def plot(self, ax=None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        layout = self.layout(self.graph)
        for e in self.edges:
            u,v=e["start"],e["end"]
            layout[u,v] = (layout[u][0], layout[v][1])
        nx.draw(self.graph, ax=ax, pos=layout, **self.kwargs,connectionstyle='arc3, rad = 0.1')
        nx.draw_networkx_edge_labels(self.graph, ax=ax, pos=layout,
                                     edge_labels=nx.get_edge_attributes(self.graph, "weight"),label_pos=0.65)
        nx.draw_networkx_labels(self.graph, ax=ax, pos=layout)
        return ax

    def draw(self):
        self.plot()
        plt.show()
