import tensorflow as tf
from tensorflow import keras


class MPGNormalisationLayer(keras.layers.Layer):
    """
    MPG normalisation layer

    This layer normalises the edge weights of a graph. The normalisation is done by dividing by the standard deviation.
    """

    def __init__(self, edges_interval: tuple = None, weights_matrix: int = None, mask_zeros=False,
                 mask_tensor: int = None,
                 regularisation=1, **kwargs):
        """
        :param edges_interval: The interval of the edge weights to normalise. The interval is given as a tuple of the form
        (start, end). The start and end are included in the interval.
        :param weights_matrix: The index of the edge weights to normalise. The index is the index of the tensor in the input
        tensor.
        :param mask_zeros: If true, the standard deviation is calculated only on the non-zero edge weights.
        :param mask_tensor: The index of the tensor that contains the mask. The index is the index of the tensor in the
        input tensor.
        :param regularisation: The regularisation term. The standard deviation is added to the regularisation factor.
        :param kwargs: The keyword arguments of the keras layer.
        """
        super(MPGNormalisationLayer, self).__init__(**kwargs)
        self.edges_interval = edges_interval
        self.edges_matrix = weights_matrix
        if weights_matrix is not None and edges_interval is not None:
            raise ValueError("edges_interval and edges_matrix can not be set at the same time")
        self.mask_zeros = mask_zeros
        self.mask_tensor = mask_tensor
        self.regularisation = regularisation

    def get_std(self, input):
        if self.mask_tensor is not None:
            mask = input[..., self.mask_tensor:self.mask_tensor + 1, :, :]
            mask = tf.cast(mask, tf.bool)
            return tf.math.reduce_std(tf.ragged.boolean_mask(input, mask), axis=(-2, -1), keepdims=True).to_tensor()
        elif self.mask_zeros:
            mask = input != 0
            return tf.math.reduce_std(tf.ragged.boolean_mask(input, mask), axis=(-2, -1), keepdims=True).to_tensor()
        else:
            return tf.math.reduce_std(input, axis=(-2, -1), keepdims=True)

    def build(self, input_shape):
        super(MPGNormalisationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.edges_matrix is not None:
            submatrix = inputs[..., self.edges_matrix:self.edges_matrix + 1, :, :]
            std = self.get_std(submatrix)
            return tf.concat(
                [inputs[..., :self.edges_matrix, :, :], submatrix / (std+self.regularisation), inputs[..., self.edges_matrix + 1:, :, :]],
                axis=-3)
        elif self.edges_interval is not None:
            a, b = self.edges_interval
            slice = inputs[..., a:b]
            std = tf.math.reduce_std(slice, axis=-1, keepdims=True)
            return tf.concat([inputs[..., :a], slice / (std+self.regularisation), inputs[..., b:]], axis=-1)
        else:
            std = self.get_std(inputs)
            return inputs / (std+self.regularisation)

    def compute_output_shape(self, input_shape):
        return input_shape


class MPGUniformNormalisationLayer(keras.layers.Layer):
    """
    MPG normalisation layer

    This layer normalises the edge weights of a graph. The normalisation is done by dividing by the standard deviation.
    """

    def __init__(self, edges_interval: tuple = None, edges_matrix: int = None, mask_zeros=False,
                 mask_tensor: int = None,
                 regularisation=0.01, **kwargs):
        """
        :param edges_interval: The interval of the edge weights to normalise. The interval is given as a tuple of the form
        (start, end). The start and end are included in the interval.
        :param edges_matrix: The index of the edge weights to normalise. The index is the index of the tensor in the input
        tensor.
        :param mask_zeros: If true, the standard deviation is calculated only on the non-zero edge weights.
        :param mask_tensor: The index of the tensor that contains the mask. The index is the index of the tensor in the
        input tensor.
        :param regularisation: The regularisation term. The standard deviation is added to the regularisation factor.
        :param kwargs: The keyword arguments of the keras layer.
        """
        super(MPGUniformNormalisationLayer, self).__init__(**kwargs)
        self.edges_interval = edges_interval
        self.edges_matrix = edges_matrix
        if edges_matrix is not None and edges_interval is not None:
            raise ValueError("edges_interval and edges_matrix can not be set at the same time")
        self.mask_zeros = mask_zeros
        self.mask_tensor = mask_tensor
        self.regularisation = regularisation

    def get_max(self, input):
        if self.mask_tensor is not None:
            mask = input[..., self.mask_tensor:self.mask_tensor + 1, :, :]
            mask = tf.cast(mask, tf.bool)
            return tf.math.reduce_max(tf.ragged.boolean_mask(tf.abs(input), mask), axis=(-2, -1), keepdims=True).to_tensor()
        elif self.mask_zeros:
            mask = input != 0
            return tf.math.reduce_max(tf.ragged.boolean_mask(tf.abs(input), mask), axis=(-2, -1), keepdims=True).to_tensor()
        else:
            return tf.math.reduce_max(tf.abs(input), axis=(-2, -1), keepdims=True)

    def build(self, input_shape):
        super(MPGUniformNormalisationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.edges_matrix is not None:
            submatrix = inputs[..., self.edges_matrix:self.edges_matrix + 1, :, :]
            M=self.get_range(submatrix)
            return tf.concat(
                [inputs[..., :self.edges_matrix, :, :], submatrix / (M+self.regularisation), inputs[..., self.edges_matrix + 1:, :, :]],
                axis=-3)
        elif self.edges_interval is not None:
            a, b = self.edges_interval
            slice = inputs[..., a:b]
            M = tf.math.reduce_max(tf.abs(slice), axis=-1, keepdims=True)
            return tf.concat([inputs[..., :a], slice / (M+self.regularisation), inputs[..., b:]], axis=-1)
        else:
            M = self.get_max(inputs)
            return inputs / (M+self.regularisation)

    def compute_output_shape(self, input_shape):
        return input_shape


class LaplacianMatrixLayer(keras.layers.Layer):
    """
    Laplacian matrix layer

    This layer computes the laplacian matrix of a graph.
    """

    def __init__(self, edges_matrix: int = None, negate: bool = False, **kwargs):
        """
        :param edges_matrix: the index of the edges matrix in the input tensor.
        If None, compute the laplacian matrix of each matrix in the input tensor
        :param negate: if True, the laplacian matrix is negated
        """
        super(LaplacianMatrixLayer, self).__init__(**kwargs)
        self.negate = negate
        self.edges_matrix = edges_matrix

    def build(self, input_shape):
        super(LaplacianMatrixLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.edges_matrix is not None:
            submatrix = inputs[..., self.edges_matrix:self.edges_matrix + 1, :, :]
            D = tf.linalg.diag(tf.reduce_sum(submatrix, axis=(-1,)))
            result = D - submatrix
            if self.negate:
                return -result
            result = tf.concat(
                [inputs[..., :self.edges_matrix, :, :], result, inputs[..., self.edges_matrix + 1:, :, :]], axis=-3)
        else:
            D = tf.linalg.diag(tf.reduce_sum(inputs, axis=-1))
            result = D - inputs
            if self.negate:
                result = -result
        return result

    def compute_output_shape(self, input_shape):
        return input_shape
