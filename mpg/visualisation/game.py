import abc
import sys
from copy import deepcopy
from typing import Dict, Any, List

from visualisation import colors
from visualisation.graph import GraphVisualisation
import visualisation.graph as vg
from games import MeanPayoffGraph
import inspect
from games import mpg
from visualisation.colors import Colour

CURRENT_POSITION_COLOUR: Colour = Colour("olive")
PLAYER_1_COLOUR: Colour = Colour("grass green")
PLAYER_2_COLOUR: Colour = Colour("burnt orange")
SHARED_COLOUR: Colour = Colour("gold")
EDGE_COLOUR: Colour = Colour("black")
NODE_COLOUR: Colour = Colour("turquoise")


class ColourDescription:
    def __init__(self, colour: colors.Colour, description: str):
        self.colour = colour
        self.description = description

    def __repr__(self):
        return f"{self.description} ({self.colour})"

    def _repr_html_(self):
        return f"{self.colour._repr_html_()}: {self.description}"


def colours_info():
    print("Available colours:")
    for name, value in inspect.getmembers(sys.modules[__name__]):
        if name.endswith("_COLOUR"):
            print(f"{name} = {value}")


class Legend:
    def __init__(self, colours: List[ColourDescription]):
        self.colours = deepcopy(colours)

    def __repr__(self):
        return f"Legend({self.colours})"

    def _repr_html_(self):
        outputs = []
        for C in self.colours:
            outputs.append(f"<li>{C._repr_html_()}</li>")
        return f"<ul>{''.join(outputs)}</ul>"

    def __add__(self, other):
        return Legend(self.colours + other.colours)

    def __radd__(self, other):
        return Legend(self.colours + other.colours)

    def __iadd__(self, other):
        self.colours += other.colours
        return self

    def __len__(self):
        return len(self.colours)

    def __getitem__(self, item):
        return self.colours[item]

    def __setitem__(self, key, value):
        self.colours[key] = value

    def __delitem__(self, key):
        del self.colours[key]

    def __iter__(self):
        return iter(self.colours)

    def __reversed__(self):
        return reversed(self.colours)

    def __contains__(self, item):
        return item in self.colours

    def __eq__(self, other):
        return self.colours == other.colours

    def __ne__(self, other):
        return self.colours != other.colours


class EdgeLegend(Legend):
    def __init__(self, colours: List[ColourDescription]):
        super().__init__(colours)
        for C in self.colours:
            C.colour._symbol = "&rarr;"

    def __add__(self, other) -> Legend:
        return Legend(self.colours + other.colours)

    def __radd__(self, other) -> Legend:
        return Legend(self.colours + other.colours)


class NodeLegend(Legend):
    def __init__(self, colours: List[ColourDescription]):
        super().__init__(colours)
        for C in self.colours:
            C.colour._symbol = "&#9679;"
    def __add__(self, other) -> Legend:
        return Legend(self.colours + other.colours)

    def __radd__(self, other) -> Legend:
        return Legend(self.colours + other.colours)


class EdgeColourer(abc.ABC):
    def __call__(self, index: Any, node: vg.EdgeMetadata):
        pass

    @abc.abstractmethod
    def legend(self) -> EdgeLegend:
        pass


class StrategyVisualiser(EdgeColourer):

    def __init__(self, strategy1=None, strategy2=None, game: MeanPayoffGraph = None):
        self.strategy1 = strategy1
        self.strategy2 = strategy2
        if game is not None:
            self.strategy1, self.strategy2 = mpg.optimal_strategy_pair(game)

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
        return colour.hex

    def legend(self) -> EdgeLegend:
        return EdgeLegend([
            ColourDescription(EDGE_COLOUR, "No strategy"),
            ColourDescription(PLAYER_1_COLOUR, "Player 1's strategy"),
            ColourDescription(PLAYER_2_COLOUR, "Player 2's strategy"),
            ColourDescription(SHARED_COLOUR, "Shared strategy"),
        ])


class NodeColourer(abc.ABC):
    def __call__(self, index: int, node: vg.NodeMetadata):
        pass

    @abc.abstractmethod
    def legend(self) -> NodeLegend:
        pass


class PositionVisualiser(NodeColourer):

    def __init__(self, position: Any = None):
        self.position = position

    def __call__(self, index: Any, node: vg.NodeMetadata):
        colour = NODE_COLOUR
        if node["id"] == self.position:
            colour = CURRENT_POSITION_COLOUR
        return colour.hex

    def legend(self) -> NodeLegend:
        return NodeLegend([
            ColourDescription(NODE_COLOUR, "Other node"),
            ColourDescription(CURRENT_POSITION_COLOUR, "Current position"),
        ])


class WinnerVisualiser(NodeColourer):
    def __init__(self, game: MeanPayoffGraph):
        # W is a dictionary of nodes and whether they are winners
        W = mpg.winners(game,*mpg.optimal_strategy_pair(G=game))
        # W1 is a dictionary of nodes and whether they are winners for player 1
        self.W1={u[0]:W[u] for u in W if u[1]==MeanPayoffGraph.player0}
        # W2 is a dictionary of nodes and whether they are winners for player 2
        self.W2={u[0] :not W[u] for u in W if u[1]==MeanPayoffGraph.player1}


    def __call__(self, index: Any, node: vg.NodeMetadata):
        colour: Colour = NODE_COLOUR
        u=node["id"]
        if self.W1[u] and self.W2[u]:
            colour = SHARED_COLOUR
        elif self.W1[u]:
            colour = PLAYER_1_COLOUR
        elif self.W2[u]:
            colour = PLAYER_2_COLOUR
        else:
            colour = NODE_COLOUR
        return colour.hex

    def legend(self) -> NodeLegend:
        return NodeLegend([
            ColourDescription(NODE_COLOUR, "If either player starts, he is not a winner"),
            ColourDescription(PLAYER_1_COLOUR, "If player 1 starts, he is a winner"),
            ColourDescription(PLAYER_2_COLOUR, "If player 2 starts, he is a winner"),
            ColourDescription(SHARED_COLOUR, "If either player starts, he is a winner"),
        ])


class MPGVisualisation(GraphVisualisation):

    def __init__(self, game: MeanPayoffGraph):
        super().__init__(graph=game)

    def set_node_color_mapping(self, node_color_mapping):
        if inspect.isclass(node_color_mapping):
            node_color_mapping = node_color_mapping(game=self.graph)
        super().set_node_color_mapping(node_color_mapping)

    def set_edge_color_mapping(self, edge_color_mapping):
        if inspect.isclass(edge_color_mapping):
            edge_color_mapping = edge_color_mapping(game=self.graph)
        super().set_edge_color_mapping(edge_color_mapping)

    def legend(self) -> Legend:
        return self.node_color_mapping.legend() + self.edge_color_mapping.legend()
