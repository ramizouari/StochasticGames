'''
Board class for the game of TicTacToe.
Default board size is 3x3.
Board data:
  1=white(O), -1=black(X), 0=empty
  first dim is column , 2nd is row:
     pieces[0][0] is the top left square,
     pieces[2][0] is the bottom left square,
Squares are stored and manipulated as (x,y) tuples.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the board for the game of Othello by Eric P. Nichols.

'''
import numpy as np


# from bkcharts.attributes import color
class MPGState:

    def __init__(self, environment,state,turn:int,max_turns:int):
        "Set up initial board configuration."

        self.environment=environment
        self.state=state
        self.mean_payoffs=0
        self.turn=turn
        self.max_turns=max_turns

    def get_legal_moves(self, color=None):
        return self.environment[0,self.state,:]

    def has_legal_moves(self):
        return True

    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction;
        @param color (1=white,-1=black)
        """
        if self.turn < self.max_turns:
            return False
        return color*self.mean_payoffs > 0

    def execute_move(self, move, color):
        """Perform the given move on the board;
        color gives the color pf the piece to play (1=white,-1=black)
        """
        # Add the piece to the empty square.
        K=self.turn/(self.turn+1)
        self.turn+=1
        self.mean_payoffs=K*self.mean_payoffs+self.environment[1,self.state,move]/self.turn
        self.state=move


    def __neg__(self):
        dual_state=MPGState(np.copy(self.environment),self.state,self.turn,self.max_turns)
        dual_state.environment[1]=-dual_state.environment
        return dual_state

    def __invert__(self):
        return self.__neg__()