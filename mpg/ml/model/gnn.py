import tensorflow as tf
import mpg.gnn.gcn as gcn
from . import utils

keras = tf.keras
from ..layers import augmentation, normalisation, activation,padding


class GNN(keras.Model):
    """
    Graph Neural Network model for predicting policy and value targets
    This model is based on the Graph Convolutional Network (GCN) model from Kipf and Welling (2017) adjusted for
    weighted graphs.

    Compared to the MLP model, the GNN model has the following advantages:
    1. It is permutation equivariant, meaning that the order of the nodes gives the same output, up to a permutation of the nodes.
    2. It also supports graph of arbitrary sizes. The MLP model in contrast only supports graphs of fixed size.
    3. It is also padding invariant, meaning that a padding does not change the output of the model.
    """

    ADJACENCY_MATRIX_AXIS = 0
    WEIGHT_MATRIX_AXIS = 1

    def __init__(self, conv_layers=None, fc_layers=None, flat_model=None, transposed=False, masked_softmax=False,
                 ragged_batch=False, name=None, *, weight_normalisation=False, weight_normalisation_regularization=0.1,
                 random_connection_probability: float = 0, weight_noise_layer=None,
                 vacuous_connections=False):
        if transposed:
            environment_shape = (2, None, None)
        else:
            environment_shape = (None, None, 2)

        if conv_layers is None:
            conv_layers = [16, 24, 64]
        if ragged_batch:
            batched_environment_shape = (None, *environment_shape)
            self.input_environment = keras.layers.Input(
                type_spec=tf.RaggedTensorSpec(shape=batched_environment_shape, ragged_rank=2),
                name="environment")  # s: batch_size x board_x x board_y
            ragged_input_environment = utils.RagTensor(name="rag")(self.input_environment)
            input_environment = utils.UnragTensor(name="unrag")(ragged_input_environment)
        else:
            self.input_environment = keras.layers.Input(shape=environment_shape,
                                                        name="environment")  # s: batch_size x board_x x board_y
            input_environment = self.input_environment
            ragged_input_environment = input_environment

        if transposed:
            input_environment = tf.transpose(self.input_environment, perm=[0, 2, 3, 1], name="transpose_environment")

        self.input_state = keras.layers.Input(shape=(), name="state")
        input_state = tf.cast(self.input_state, tf.int32)
        input_state = keras.layers.Reshape((1,))(input_state)
        # Input environment is of shape (batch_size,graph_size,graph_size,2)
        # Input state is of shape (batch_size,1)
        graph_size = tf.shape(input_environment)[1]

        A = input_environment[..., utils.ADJACENCY_MATRIX_AXIS]
        if vacuous_connections:
            A = padding.VacuousConnections(name="vacuous_connections")(A)
        W = input_environment[..., utils.WEIGHT_MATRIX_AXIS]

        if random_connection_probability > 0:
            A, W = augmentation.RandomConnectionLayer(p=random_connection_probability, name="random_connection")([A, W])

        if weight_normalisation:
            if weight_normalisation == "uniform":
                normalisation_layer = normalisation.MPGUniformNormalisationLayer(mask_zeros=True,
                                                                                 regularisation=weight_normalisation_regularization,
                                                                                 name="weight_normalisation")
                A, W = normalisation_layer([A, W])
            elif weight_normalisation == "std":
                normalisation_layer = normalisation.MPGNormalisationLayer(mask_zeros=True,
                                                                          regularisation=weight_normalisation_regularization,
                                                                          name="weight_normalisation")
                A, W = normalisation_layer([A, W])
            else:
                raise ValueError(f"Unknown weight normalisation {weight_normalisation}")

        if weight_noise_layer is not None:
            if isinstance(weight_noise_layer, augmentation.EdgeWeightsNoiseLayer):
                A, W = weight_noise_layer([A, W])
            else:
                noise_layer=augmentation.EdgeWeightsNoiseLayer(noise_layer=weight_noise_layer,mask_zeros=True,name="weight_noise")
                A, W = noise_layer([A, W])

        # state_encoding is of shape (batch_size,1,graph_size)
        state_encoding = tf.one_hot(input_state, depth=graph_size, name="state_encoding")
        # legals mask is of shape (batch_size,graph_size,1)
        legals_mask = keras.layers.Dot(axes=(-2, -1), name="legals_mask")([A, state_encoding])
        # We transform it to a boolean mask with shape (batch_size,graph_size)
        legals_mask = legals_mask[..., 0] > 0.5
        # X should be of shape (batch_size,graph_size,1) and type float32
        # X can be considered as function mapping the starting state to 1 and the other states to 0
        X = tf.cast(state_encoding, tf.float32)
        X = tf.transpose(X, perm=[0, 2, 1], name="pre_gnn_transpose_X")
        # Applying the convolutional layers
        # TODO: Add batch normalisation
        for i, args in enumerate(conv_layers):
            X = utils.build_layer_from_args(gcn.WGCN, args, name=f"conv_{i}")([A, W, X])
            X = keras.layers.BatchNormalization(name=f"conv_batch_norm_{i}")(X)
        # At this point, X should be of shape (batch_size,graph_size,last_channels) and type float32
        # Y should be of shape (batch_size,last_channels, graph_size) and type float32
        Y = tf.transpose(X, perm=[0, 2, 1], name="post_gnn_transpose_X")
        # matrix-vector multiplication between Y and state_encoding.
        # shapes are (batch_size,last_channels,graph_size) and (batch_size,1,graph_size)
        projection = tf.math.multiply(Y, state_encoding, name="state_mask")
        # projection should be of shape (batch_size,last_channels) and type float32
        projection = tf.reduce_sum(projection, axis=-1, name="average_pooling")

        # Applying the flat layers. The output should be of shape (batch_size,1)
        # The input should be of shape (batch_size,last_channels)

        if fc_layers is None:
            fc_layers = [128]

        # projection is already of shape (batch_size,last_channels)
        flattened = projection
        for i, units in enumerate(fc_layers):
            flattened = keras.layers.BatchNormalization(name=f"flat_batch_norm_{i}")(flattened)
            flattened = keras.layers.Dense(units, activation="relu", name=f"flat_{i}")(flattened)

        if flat_model is not None:
            flattened = flat_model(flattened)

        # Z should be of shape (batch_size,graph_size) and type float32 after both operations
        Z = gcn.WGCN(filters=1, activation="relu", name="policy_logits")([A, W, X])
        Z = Z[..., 0]
        if ragged_batch:
            if masked_softmax:
                pi = activation.MaskedSoftmax(name="policy_targets_static")(
                    [Z, legals_mask])  # batch_size x num_actions
            else:
                pi = keras.layers.Softmax(name="policy_targets_static")(Z)  # batch_size x num_actions
            row_lengths = tf.cast(ragged_input_environment.row_lengths(), tf.int32)
            self.pi = utils.PotentiallyRagPolicy(name="policy_targets")([self.input_environment, pi, row_lengths])
        else:
            if masked_softmax:
                self.pi = activation.MaskedSoftmax(name="policy_targets")([Z, legals_mask])  # batch_size x num_actions
            else:
                self.pi = keras.layers.Softmax(name="policy_targets")(Z)  # batch_size x num_actions
        #        self.pi = keras.layers.Multiply(name="policy_targets")([self.pi, legals_mask])
        self.v = keras.layers.Dense(1, activation="tanh", name="value_targets")(flattened)  # batch_size x 1
        super().__init__(inputs=[self.input_environment, self.input_state],
                         outputs=[self.v, self.pi], name=name)


class ResGNN(keras.Model):
    """
    Graph Neural Network model for predicting policy and value targets
    This model is based on the Graph Convolutional Network (GCN) model from Kipf and Welling (2017) adjusted for
    weighted graphs.

    Compared to the MLP model, the GNN model has the following advantages:
    1. It is permutation equivariant, meaning that the order of the nodes gives the same output, up to a permutation of the nodes.
    2. It also supports graph of arbitrary sizes. The MLP model in contrast only supports graphs of fixed size.
    3. It is also padding invariant, meaning that a padding does not change the output of the model.
    """

    ADJACENCY_MATRIX_AXIS = 0
    WEIGHT_MATRIX_AXIS = 1

    def __init__(self, conv_layers=None, fc_layers=None, flat_model=None, transposed=False, masked_softmax=False,
                 ragged_batch=False, name=None, *, weight_normalisation=False, weight_normalisation_regularization=0.1,
                 random_connection_probability: float = 0, weight_noise_layer=None,
                 residual_connections=None, difference_residual=False,
                 graph_normalisation=False,vacuous_connections=False):
        if transposed:
            environment_shape = (2, None, None)
        else:
            environment_shape = (None, None, 2)

        if residual_connections is None:
            residual_connections= {}

        if conv_layers is None:
            conv_layers = [16, 24, 64]
        if ragged_batch:
            batched_environment_shape = (None, *environment_shape)
            self.input_environment = keras.layers.Input(
                type_spec=tf.RaggedTensorSpec(shape=batched_environment_shape, ragged_rank=2),
                name="environment")  # s: batch_size x board_x x board_y
            ragged_input_environment = utils.RagTensor(name="rag")(self.input_environment)
            input_environment = utils.UnragTensor(name="unrag")(ragged_input_environment)
        else:
            self.input_environment = keras.layers.Input(shape=environment_shape,
                                                        name="environment")  # s: batch_size x board_x x board_y
            input_environment = self.input_environment
            ragged_input_environment = input_environment

        if transposed:
            input_environment = tf.transpose(self.input_environment, perm=[0, 2, 3, 1], name="transpose_environment")
        self.input_state = keras.layers.Input(shape=(), name="state")
        input_state = tf.cast(self.input_state, tf.int32)
        input_state = keras.layers.Reshape((1,))(input_state)
        # Input environment is of shape (batch_size,graph_size,graph_size,2)
        # Input state is of shape (batch_size,1)
        graph_size = tf.shape(input_environment)[1]

        A = input_environment[..., utils.ADJACENCY_MATRIX_AXIS]
        if vacuous_connections:
            A = padding.VacuousConnections(name="vacuous_connections")(A)
        W = input_environment[..., utils.WEIGHT_MATRIX_AXIS]

        if random_connection_probability > 0:
            A, W = augmentation.RandomConnectionLayer(p=random_connection_probability, name="random_connection")([A, W])

        if weight_normalisation and weight_normalisation != "none":
            if weight_normalisation == "uniform":
                normalisation_layer = normalisation.MPGUniformNormalisationLayer(mask_zeros=True,
                                                                                 regularisation=weight_normalisation_regularization,
                                                                                 name="weight_normalisation")
                A, W = normalisation_layer([A, W])
            elif weight_normalisation == "std":
                normalisation_layer = normalisation.MPGNormalisationLayer(mask_zeros=True,
                                                                          regularisation=weight_normalisation_regularization,
                                                                          name="weight_normalisation")
                A, W = normalisation_layer([A, W])
            else:
                raise ValueError(f"Unknown weight normalisation {weight_normalisation}")

        if weight_noise_layer is not None:
            if isinstance(weight_noise_layer, augmentation.EdgeWeightsNoiseLayer):
                A, W = weight_noise_layer([A, W])
            else:
                noise_layer=augmentation.EdgeWeightsNoiseLayer(noise_layer=weight_noise_layer,mask_zeros=True,name="weight_noise")
                A, W = noise_layer([A, W])

        # state_encoding is of shape (batch_size,1,graph_size)
        state_encoding = tf.one_hot(input_state, depth=graph_size, name="state_encoding")
        # legals mask is of shape (batch_size,graph_size,1)
        legals_mask = keras.layers.Dot(axes=(-2, -1), name="legals_mask")([A, state_encoding])
        # We transform it to a boolean mask with shape (batch_size,graph_size)
        legals_mask = legals_mask[..., 0] > 0.5
        # X should be of shape (batch_size,graph_size,1) and type float32
        # X can be considered as function mapping the starting state to 1 and the other states to 0
        X = tf.cast(state_encoding, tf.float32)
        X = tf.transpose(X, perm=[0, 2, 1], name="pre_gnn_transpose_X")
        X_list=[X]
        # Applying the convolutional layers
        # TODO: Add batch normalisation
        for i, args in enumerate(conv_layers):
            X = utils.build_layer_from_args(gcn.WGCN, args, name=f"conv_{i}")([A, W, X])
            if graph_normalisation:
                X = keras.layers.BatchNormalization(name=f"conv_batch_norm_{i}")(X)
            if i in residual_connections:
                if isinstance(residual_connections[i], int):
                    if difference_residual:
                        X = keras.layers.Subtract(name=f"conv_residual_{i}")([X, X_list[residual_connections[i]]])
                    else:
                        X = keras.layers.Add(name=f"conv_residual_{i}")([X, X_list[residual_connections[i]]])
                elif isinstance(residual_connections[i],list):
                    X_res=[X_list[res] for res in residual_connections[i]]
                    if difference_residual:
                        X_res=[-x for x in X_res]
                    X = keras.layers.Add(name=f"conv_residual_{i}")([X, *X_res])
            X_list.append(X)


        # At this point, X should be of shape (batch_size,graph_size,last_channels) and type float32
        # Y should be of shape (batch_size,last_channels, graph_size) and type float32
        Y = tf.transpose(X, perm=[0, 2, 1], name="post_gnn_transpose_X")
        # matrix-vector multiplication between Y and state_encoding.
        # shapes are (batch_size,last_channels,graph_size) and (batch_size,1,graph_size)
        projection = tf.math.multiply(Y, state_encoding, name="state_mask")
        # projection should be of shape (batch_size,last_channels) and type float32
        projection = tf.reduce_sum(projection, axis=-1, name="average_pooling")

        # Applying the flat layers. The output should be of shape (batch_size,1)
        # The input should be of shape (batch_size,last_channels)

        if fc_layers is None:
            fc_layers = [128]

        # projection is already of shape (batch_size,last_channels)
        flattened = projection
        for i, units in enumerate(fc_layers):
            flattened = keras.layers.BatchNormalization(name=f"flat_batch_norm_{i}")(flattened)
            flattened = keras.layers.Dense(units, activation="relu", name=f"flat_{i}")(flattened)

        if flat_model is not None:
            flattened = flat_model(flattened)

        # Z should be of shape (batch_size,graph_size) and type float32 after both operations
        Z = gcn.WGCN(filters=1, activation="relu", name="policy_logits")([A, W, X])
        Z = Z[..., 0]
        if ragged_batch:
            if masked_softmax:
                pi = activation.MaskedSoftmax(name="policy_targets_static")(
                    [Z, legals_mask])  # batch_size x num_actions
            else:
                pi = keras.layers.Softmax(name="policy_targets_static")(Z)  # batch_size x num_actions
            row_lengths = tf.cast(ragged_input_environment.row_lengths(), tf.int32)
            self.pi = utils.PotentiallyRagPolicy(name="policy_targets")([self.input_environment, pi, row_lengths])
        else:
            if masked_softmax:
                self.pi = activation.MaskedSoftmax(name="policy_targets")([Z, legals_mask])  # batch_size x num_actions
            else:
                self.pi = keras.layers.Softmax(name="policy_targets")(Z)  # batch_size x num_actions
        #        self.pi = keras.layers.Multiply(name="policy_targets")([self.pi, legals_mask])
        self.v = keras.layers.Dense(1, activation="tanh", name="value_targets")(flattened)  # batch_size x 1
        super().__init__(inputs=[self.input_environment, self.input_state],
                         outputs=[self.v, self.pi], name=name)
