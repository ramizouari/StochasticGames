import sys
from azg.utils import *

import argparse
from tensorflow import keras



"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""
class MPGNNet():
    def __init__(self, game, args):
        # game params
        self.action_size = game.getActionSize()
        self.args = args

        env_shape=(2,game.getBoardSize(),game.getBoardSize())

        # Neural Net
        self.input_environment = keras.layers.Input(shape=env_shape)    # s: batch_size x board_x x board_y
        self.input_state=keras.layers.Input(shape=())
        flattened=keras.layers.Flatten()(self.input_environment,name="flatten")
        stack=keras.layers.Concatenate(axis=1)([flattened,self.input_state])
        y=keras.layers.BatchNormalization()(stack)
        y=keras.layers.Dense(128,activation="relu")(y)
        z=keras.layers.BatchNormalization()(y)
        self.pi=keras.layers.Dense(self.action_size,activation="softmax",name="pi")(z)   # batch_size x self.action_size
        self.v=keras.layers.Dense(1,activation="tanh",name="v")(z)                    # batch_size x 1
        self.model=keras.models.Model(inputs=[self.input_environment,self.input_state],outputs=[self.pi,self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=keras.optimizers.Adam(args.lr))
