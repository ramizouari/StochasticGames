from __future__ import print_function
import sys
sys.path.append('..')
from azg.Game import Game
from .MPGLogic import MPGState
import numpy as np
import mpg.graph.random_graph as rg

import itertools


"""
Game class implementation for the game of TicTacToe.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""
import abc
class MPGGame(Game,abc.ABC):
    def __init__(self, n:int):
        self.n = n
        self.graph = None
        self.game_tensor=None
    @abc.abstractmethod
    def getInitBoard(self):
        pass

    def getBoardSize(self):
        # (a,b) tuple
        return self.n

    def getActionSize(self):
        # return number of actions
        # Add 1 for the pass action?
        return self.n+1

    def getNextState(self, board, player, action):
        current_state=board.state
        board.state=action
        K=board.turn/(board.turn+1)
        board.mean_payoffs=K*board.mean_payoffs+board.env[1,current_state,action]
        return board,-player

    def getValidMoves(self, board, player):
        return self.game_tensor[0,board.state]

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1

        if board.is_win(player):
            return 1
        if board.is_win(-player):
            return -1
        if board.has_legal_moves():
            return 0
        # draw has a very little value
        return 1e-4

    def getCanonicalForm(self, board, player):
        if player==-1:
            board=~board
        return {"environment": board.environment, "state": board.state}

    def getSymmetries(self, board, pi):
        return [(board,pi)]

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return str(self.game_tensor)

class UniformGnpMPGGame(MPGGame):
    def __init__(self,n,p,a,b,max_turns,seeder=None):
        super().__init__(n)
        self.p=p
        self.a=a
        self.b=b
        if seeder is None:
            seeder=np.random.randint(0,2**32)
        self.rng=np.random.RandomState(seeder)
        self.max_turns=max_turns
    def getInitBoard(self):
        self.graph=rg.gnp_random_mpg(self.n,self.p,distribution="uniform",low=self.a,high=self.b,seed=self.rng.randint(0,2**32))
        self.game_tensor=self.graph.tensor_representation
        state=self.rng.randint(0,self.n)
        return MPGState(self.game_tensor,state,0,self.max_turns)

class MPGGameBuffer(MPGGame):
    def __init__(self,games):
        self.games=games
        self.buffer=itertools.cycle(games)
        self.game=None

    def InitBoard(self):
        self.game=next(self.buffer)
        return self.game.getInitBoard()

    def getBoardSize(self):
        return self.game.getBoardSize()

    def getActionSize(self):
        return self.game.getActionSize()

    def getNextState(self,board,player,action):
        return self.game.getNextState(board,player,action)

    def getValidMoves(self,board,player):
        return self.game.getValidMoves(board,player)

    def getGameEnded(self,board,player):
        return self.game.getGameEnded(board,player)

    def getCanonicalForm(self,board,player):
        return self.game.getCanonicalForm(board,player)

    def getSymmetries(self,board,pi):
        return self.game.getSymmetries(board,pi)

    def stringRepresentation(self,board):
        return self.game.stringRepresentation(board)
