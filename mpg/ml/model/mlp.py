import tensorflow as tf
import mpg.gnn.gcn as gcn
from ..layers import augmentation,normalisation,padding
keras=tf.keras
from . import utils
class MLP(keras.Model):
    def __init__(self,graph_size,num_actions,state_shape, flat_layers=None,graph_max_size=None, transposed=False,name=None):
        if transposed:
            environment_shape=(2,graph_size,graph_size)
        else:
            environment_shape=(graph_size,graph_size,2)
        if graph_max_size is not None:
            graph_size=graph_max_size
            if transposed:
                environment_shape=(2,None,None)
            else:
                environment_shape=(None,None,2)
            self.input_environment = keras.layers.Input(shape=environment_shape,
                                                        name="environment")
            self.input_environment=padding.GraphPaddingLayer(graph_size,name="padding")(self.input_environment)
        else:
            self.input_environment = keras.layers.Input(shape=environment_shape,
                                                        name="environment")  # s: batch_size x board_x x board_y
        if transposed:
            self.input_environment=tf.transpose(self.input_environment,perm=[0,2,3,1],name="transpose_environment")

        #input_environment = normalisation.MPGNormalisationLayer(weights_matrix=utils.WEIGHT_MATRIX_AXIS,
        #                                                             mask_tensor=utils.ADJACENCY_MATRIX_AXIS,
        #                                                             name="normalisation")(self.input_environment)
        #self.input_environment=augmentation.EdgeWeightsNoiseLayer(weights_matrix=utils.WEIGHT_MATRIX_AXIS,
        #                mask_tensor=utils.ADJACENCY_MATRIX_AXIS,name="edge_noise")(self.input_environment)
        self.input_state = keras.layers.Input(shape=state_shape, name="state")
        state_reshape = keras.layers.Reshape((1,))(self.input_state)
        state_encoding = keras.layers.CategoryEncoding(num_tokens=num_actions, output_mode="one_hot",
                                                       name="state_encoding")(state_reshape)
        flattened = keras.layers.Flatten(name="flatten")(self.input_environment)
        stack = keras.layers.Concatenate()([flattened, state_encoding])

        legals_mask = keras.layers.Dot(axes=(-2, -1), name="legals_mask")(
            [self.input_environment[..., utils.ADJACENCY_MATRIX_AXIS], state_encoding])

        y = keras.layers.BatchNormalization()(stack)
        y = keras.layers.Dense(128)(y)
        z = keras.layers.BatchNormalization()(y)
        self.pi = keras.layers.Dense(num_actions, activation="softmax", name="policy_targets_unmasked")(
            z)  # batch_size x self.action_size
        self.pi = keras.layers.Multiply(name="policy_targets")([self.pi, legals_mask])
        self.v = keras.layers.Dense(1, activation="tanh", name="value_targets")(z)  # batch_size x 1
        super().__init__(inputs=[self.input_environment, self.input_state],
                                   outputs=[self.v, self.pi],name=name)