import abc

import numpy as np

from . import mpg


class AbstractStrategy:
    def __init__(self, game: mpg.MeanPayoffGraph):
        self.game = game

    @abc.abstractmethod
    def call(self, vertex: int) -> int:
        pass

    def __call__(self, vertex: int) -> int:
        return self.call(vertex)

    def get_game(self) -> mpg.MeanPayoffGraph:
        return self.game


class RandomStrategy(AbstractStrategy):

    def __init__(self, game: mpg.MeanPayoffGraph, seed: int = None):
        super().__init__(game)
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
        self.rng = np.random.default_rng(seed)

    def call(self, vertex: int) -> int:
        next_vertex = self.rng.choice(list(self.game.successors(vertex)))
        return next_vertex


Strategy = AbstractStrategy


class FractionalStrategy(AbstractStrategy):
    def __init__(self, game: mpg.MeanPayoffGraph, probabilities, seed=None):
        super().__init__(game)
        self.probabilities = probabilities
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
        self.rng = np.random.default_rng(seed)

    def call(self, vertex: int) -> int:
        next_vertex = self.rng.choice(self.game.successors(vertex), p=self.probabilities[vertex])
        return next_vertex


class EpsilonGreedyStrategy(AbstractStrategy):
    def __init__(self, game: mpg.MeanPayoffGraph, turn=mpg.MeanPayoffGraph.player0, epsilon=0.1, seed=None):
        super().__init__(game)
        self.epsilon = epsilon
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
        self.rng = np.random.default_rng(seed)
        strategy = {}
        self.turn = turn
        for u in self.game:
            k = 0
            if turn == mpg.MeanPayoffGraph.player0:
                W = -np.inf
            else:
                W = np.inf
            for v in self.game.successors(u):
                R = self.game.edges[u, v]["weight"]
                if turn == mpg.MeanPayoffGraph.player0:
                    if R > W:
                        W = R
                        k = v
                else:
                    if R < W:
                        W = R
                        k = v
            strategy[u] = k

        self.strategy = strategy

    def call(self, vertex: int) -> int:
        if self.rng.random() < self.epsilon:
            next_vertex = self.rng.choice(self.game.successors(vertex))
        else:
            next_vertex = self.strategy[vertex]
        return next_vertex


class GreedyStrategy(AbstractStrategy):
    def __init__(self, game: mpg.MeanPayoffGraph, turn=mpg.MeanPayoffGraph.player0):
        super().__init__(game)
        self.turn = turn
        strategy = {}
        for u in self.game:
            k = 0
            if turn == mpg.MeanPayoffGraph.player0:
                W = -np.inf
            else:
                W = np.inf
            for v in self.game.successors(u):
                R = self.game.edges[u, v]["weight"]
                if turn == mpg.MeanPayoffGraph.player0:
                    if R > W:
                        W = R
                        k = v
                else:
                    if R < W:
                        W = R
                        k = v
            strategy[u] = k
        self.strategy = strategy

    def call(self, vertex: int) -> int:
        next_vertex = self.strategy[vertex]
        return next_vertex


class DeterministicStrategy(AbstractStrategy):
    def __init__(self, game: mpg.MeanPayoffGraph, strategy):
        super().__init__(game)
        self.strategy = strategy

    def call(self, vertex: int) -> int:
        next_vertex = self.strategy[vertex]
        return next_vertex
