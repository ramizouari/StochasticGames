import tensorflow as tf
import mpg.gnn.gcn as gcn
keras=tf.keras

class ResGNN(keras.Model):

    ADJACENCY_MATRIX_AXIS=0
    WEIGHT_MATRIX_AXIS=1

    def __init__(self, conv_layers=None,flat_model=None, transposed=False,name="MPGGNN"):
        super().__init__(name=name)
        if transposed:
            environment_shape=(2,None,None)
        else:
            environment_shape=(None,None,2)

        if conv_layers is None:
            conv_layers=[16,24,8]
        self.input_environment = keras.layers.Input(shape=environment_shape,
                                                    name="environment")  # s: batch_size x board_x x board_y
        if transposed:
            self.input_environment=tf.transpose(self.input_environment,perm=[0,2,3,1],name="transpose_environment")
        self.input_state = keras.layers.Input(shape=(), name="state")
        state_reshape = keras.layers.Reshape((1,))(self.input_state)
        graph_size=tf.shape(self.input_environment)[1]
        state_encoding = tf.one_hot(state_reshape,depth=graph_size,name="state_encoding")
        A=self.input_environment[...,self.ADJACENCY_MATRIX_AXIS]
        W=self.input_environment[...,self.WEIGHT_MATRIX_AXIS]

        legals_mask = keras.layers.Dot(axes=(-2, -1), name="legals_mask")([A, state_encoding])
        X=state_encoding

        for i,filters in enumerate(conv_layers):
            X=gcn.GCN(filters=filters,activation="relu",name=f"conv_{i}")([A,W,X])

        if flat_model is None:
            flat_model=[128]

        flattened = keras.layers.Flatten(name="flatten")(X)
        for i,units in enumerate(flat_model):
            flattened=keras.layers.BatchNormalization()(flattened)
            flattened=keras.layers.Dense(units,activation="relu",name=f"flat_{i}")(flattened)

        self.pi = keras.layers.Dense(self.action_size, activation="softmax", name="policy_targets_unmasked")(flattened)  # batch_size x self.action_size
        self.pi = keras.layers.Multiply(name="policy_targets")([self.pi, legals_mask])
        self.v = keras.layers.Dense(1, activation="tanh", name="value_targets")(flattened)  # batch_size x 1
        super().__init__(inputs=[self.input_environment, self.input_state],
                                   outputs=[self.v, self.pi],name=name)