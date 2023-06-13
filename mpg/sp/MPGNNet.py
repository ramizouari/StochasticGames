from tensorflow import keras



"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""
class MPGNNet:
    def __init__(self, game, args):
        # game params
        self.action_size = game.getActionSize()
        self.args = args

        env_shape=(2,game.getBoardSize(),game.getBoardSize())

        # Neural Net
        self.input_environment = keras.layers.Input(shape=env_shape,name="environment")    # s: batch_size x board_x x board_y
        self.input_state=keras.layers.Input(shape=(),name="state")
        state_reshape=keras.layers.Reshape((1,))(self.input_state)
        flattened=keras.layers.Flatten(name="flatten")(self.input_environment)
        stack=keras.layers.Concatenate()([flattened,state_reshape])
        y=keras.layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(1),gamma_regularizer=keras.regularizers.l2(1))(stack)
        y=keras.layers.Dense(128,activation="relu",kernel_regularizer=keras.regularizers.l2(1))(y)
        z=keras.layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(1),gamma_regularizer=keras.regularizers.l2(1))(y)
        self.pi=keras.layers.Dense(self.action_size,activation="softmax",name="pi",kernel_regularizer=keras.regularizers.l2(1))(z)   # batch_size x self.action_size
        self.v=keras.layers.Dense(1,activation="tanh",name="v",kernel_regularizer=keras.regularizers.l2(1))(z)                    # batch_size x 1
        self.model=keras.models.Model(inputs=[self.input_environment,self.input_state],outputs=[self.pi,self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=keras.optimizers.Adam(args.lr))
