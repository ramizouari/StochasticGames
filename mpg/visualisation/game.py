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
    """
    A description of a colour used in a graph visualisation.
    """
    def __init__(self, colour: colors.Colour, description: str):
        self.colour = colour
        self.description = description

    def __repr__(self):
        return f"{self.description} ({self.colour})"

    def _repr_html_(self):
        return f"{self.colour._repr_html_()}: {self.description}"

class Legend:
    """
    A legend for a graph visualisation.
    """
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


class NodeLegend(Legend):
    def __init__(self, colours: List[ColourDescription]):
        super().__init__(colours)
        for C in self.colours:
            C.colour._symbol = "&#9679;"


class EdgeColourer(abc.ABC):
    def __call__(self, index: Any, node: vg.EdgeMetadata):
        pass

    @abc.abstractmethod
    def legend(self) -> EdgeLegend:
        pass


class StrategyVisualiser(EdgeColourer):
    """
    Visualise a strategy of both players in a game.
    Each player's strategy is represented by a different colour.
    If both players have the same strategy, it is represented by a third colour.
    Edges avoided by both players are represented by a fourth colour.
    """

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
            ColourDescription(EDGE_COLOUR, "Not in either player's strategy"),
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
    """
    Visualise the current position in a game.
    """
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
    """
    Visualise the winner of a game if a player starts at a node.
    If both players have a winning strategy starting at that node, the node is coloured in a third colour.
    If neither player has a winning strategy starting at that node, the node is coloured in a fourth colour.
    """
    def __init__(self, game: MeanPayoffGraph,strategy1=None, strategy2=None):
        """
        Initialise the visualiser.
        :param game: The game to visualise.
        """
        if strategy1 is None or strategy2 is None:
            strategy1, strategy2 = mpg.optimal_strategy_pair(game)
        # W is a dictionary of nodes and whether they are winners
        W = mpg.winners(game, strategy1, strategy2)
        # W1 is a dictionary of nodes and whether they are winners for player 1
        self.W1={u[0]:W[u] for u in W if u[1]==MeanPayoffGraph.player0}
        # W2 is a dictionary of nodes and whether they are winners for player 2
        self.W2={u[0] :not W[u] for u in W if u[1]==MeanPayoffGraph.player1}

    def __call__(self, index: Any, node: vg.NodeMetadata):
        """
        Return the colour of a node.
        :param index: The index of the node.
        :param node: The node.
        :return: The colour of the node.
        """
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
        """
        Return the legend for the visualisation.
        :return: The legend for the visualisation.
        """
        return NodeLegend([
            ColourDescription(NODE_COLOUR, "The starting position is losing for both players"),
            ColourDescription(PLAYER_1_COLOUR, "If player 1 starts, he is a winner"),
            ColourDescription(PLAYER_2_COLOUR, "If player 2 starts, he is a winner"),
            ColourDescription(SHARED_COLOUR, "The starting position is winning for both players"),
        ])


class MPGVisualisation(GraphVisualisation):
    """
    Visualise a mean payoff graph.
    """
    def __init__(self, game: MeanPayoffGraph):
        super().__init__(graph=game)

    def set_node_color_mapping(self, node_color_mapping) -> None:
        """
        Set the node colour mapping.
        :param node_color_mapping: The node colour mapping. If this is a class, it is instantiated with the game as an argument.
        :return: None
        """
        if inspect.isclass(node_color_mapping):
            node_color_mapping = node_color_mapping(game=self.graph)
        super().set_node_color_mapping(node_color_mapping)

    def set_edge_color_mapping(self, edge_color_mapping) -> None:
        """
        Set the edge colour mapping.
        :param edge_color_mapping: The edge colour mapping. If this is a class, it is instantiated with the game as an argument.
        :return: None
        """
        if inspect.isclass(edge_color_mapping):
            edge_color_mapping = edge_color_mapping(game=self.graph)
        super().set_edge_color_mapping(edge_color_mapping)

    def legend(self) -> Legend:
        return self.node_color_mapping.legend() + self.edge_color_mapping.legend()
