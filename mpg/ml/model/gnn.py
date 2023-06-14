import tensorflow as tf
import mpg.gnn.gcn as gcn
from . import utils
keras=tf.keras
from ..layers import augmentation,normalisation

class GNN(keras.Model):

    ADJACENCY_MATRIX_AXIS=0
    WEIGHT_MATRIX_AXIS=1

    def __init__(self, conv_layers=None, flat_layers=None, flat_model=None, transposed=False, name="MPGGNN"):
        if transposed:
            environment_shape=(2,None,None)
        else:
            environment_shape=(None,None,2)

        if conv_layers is None:
            conv_layers=[16,24,8]
        self.input_environment = keras.layers.Input(shape=environment_shape,
                                                    name="environment")  # s: batch_size x board_x x board_y
        input_environment=self.input_environment

        if transposed:
            input_environment=tf.transpose(self.input_environment,perm=[0,2,3,1],name="transpose_environment")
        self.input_state = keras.layers.Input(shape=(), name="state")
        input_state = tf.cast(self.input_state,tf.int32)
        state_reshape = keras.layers.Reshape((1,))(input_state)
        graph_size=tf.shape(self.input_environment)[1]
        #input_environment = normalisation.MPGNormalisationLayer(weights_matrix=utils.WEIGHT_MATRIX_AXIS,
        #                                                             mask_tensor=utils.ADJACENCY_MATRIX_AXIS,
        #                                                             name="normalisation")(input_environment)
        A=input_environment[...,utils.ADJACENCY_MATRIX_AXIS]
        W=input_environment[...,utils.WEIGHT_MATRIX_AXIS]

        state_encoding = tf.one_hot(state_reshape,depth=graph_size,name="state_encoding")
        legals_mask = keras.layers.Dot(axes=(-2, -1), name="legals_mask")([A, state_encoding])
        X=tf.cast(state_encoding,tf.float32)
        X=tf.transpose(X,perm=[0,2,1],name="transpose_X")
        for i,filters in enumerate(conv_layers):
            X=gcn.WGCN(filters=filters,activation="relu",name=f"conv_{i}")([A,W,X])
        Y=tf.transpose(X,perm=[0,2,1],name="transpose_X")
        Y=tf.linalg.matvec(Y,state_encoding,name="extract_state")
        Y=tf.reduce_mean(Y,axis=-2,name="average_state")

        if flat_layers is None:
            flat_layers=[128]

        flattened = keras.layers.Flatten(name="flatten")(Y)
        for i,units in enumerate(flat_layers):
            flattened=keras.layers.BatchNormalization()(flattened)
            flattened=keras.layers.Dense(units,activation="relu",name=f"flat_{i}")(flattened)

        if flat_model is not None:
            flattened=flat_model(flattened)


        Z=tf.reduce_mean(self.input_environment,axis=[-1,-2],name="average_X")
        print(Z.shape)
        self.pi = keras.layers.Softmax(name="policy_targets")(Z)  # batch_size x self.action_size
#        self.pi = keras.layers.Multiply(name="policy_targets")([self.pi, legals_mask])
        self.v = keras.layers.Dense(1, activation="tanh", name="value_targets")(flattened)  # batch_size x 1
        super().__init__(inputs=[self.input_environment, self.input_state],
                                   outputs=[self.v, self.pi],name=name)