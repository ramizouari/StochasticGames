import abc
from typing import Dict, Any

from visualisation.graph import GraphVisualisation
import visualisation.graph as vg
from games import MeanPayoffGraph

CURRENT_POSITION_COLOUR: str = "olive"
PLAYER_1_COLOUR: str = "green"
PLAYER_2_COLOUR: str = "blue"
SHARED_COLOUR: str = "yellow"
EDGE_COLOUR: str = "black"
NODE_COLOUR: str = "aqua"

class EdgeColourer(abc.ABC):
    def __call__(self, index: Any, node: vg.EdgeMetadata):
        pass


class StrategyVisualiser(EdgeColourer):

    def __init__(self, strategy1, strategy2):
        self.strategy1 = strategy1
        self.strategy2 = strategy2

    def __call__(self, index: Any, edge: vg.EdgeMetadata):
        u = edge["start"]
        v = edge["end"]
        match [self.strategy1 is not None and v == self.strategy1[u],
               self.strategy2 is not None and v == self.strategy2[u]]:
            case [True, True]:
                colour = SHARED_COLOUR
            case [False, True]:
                colour = PLAYER_2_COLOUR
                pass
            case [True, False]:
                colour = PLAYER_1_COLOUR
                pass
            case _:
                colour = EDGE_COLOUR
                pass
        return colour


class NodeColourer(abc.ABC):
    def __call__(self, index: int, node: vg.NodeMetadata):
        pass

class PositionVisualiser(NodeColourer):

    def __init__(self,position:Any=None):
        self.position=position
    def __call__(self, index: Any, node: vg.NodeMetadata):
        if index == self.position:
            return CURRENT_POSITION_COLOUR
        else:
            return NODE_COLOUR
class MPGVisualisation(GraphVisualisation):

    def __init__(self, game: MeanPayoffGraph, strategy1=None, strategy2=None,starting_position:Any=None):
        super().__init__(graph=game)
        self.set_node_color_mapping(PositionVisualiser(starting_position))

    def set_strategy_pair(self, strategy1=None, strategy2=None):
        self.set_edge_color_mapping(StrategyVisualiser(strategy1,strategy2))

    def set_current_position(self,position:Any):
        self.set_node_color_mapping(PositionVisualiser(position))
