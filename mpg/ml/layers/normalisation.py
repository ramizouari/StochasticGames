import tensorflow as tf
from tensorflow import keras


class MPGNormalisationLayer(keras.layers.Layer):
    """
    MPG normalisation layer

    This layer normalises the edge weights of a graph. The normalisation is done by dividing by the standard deviation.
    """

    def __init__(self, mask_zeros=False, regularisation=1, name=None):
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
        super(MPGNormalisationLayer, self).__init__(name=name)
        self.mask_zeros = mask_zeros
        self.regularisation = regularisation


    def get_std(self, A,W):
        if self.mask_zeros:
            mask = tf.cast(A, tf.bool)
            return tf.math.reduce_std(tf.ragged.boolean_mask(W, mask), axis=(-2, -1), keepdims=True)
        else:
            return tf.math.reduce_std(W, axis=(-2, -1), keepdims=True)

    def build(self, input_shape):
        super(MPGNormalisationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A,W=inputs
        std = self.get_std(A,W)
        return A,W / (std + self.regularisation)

    def compute_output_shape(self, input_shape):
        return input_shape


class MPGUniformNormalisationLayer(keras.layers.Layer):
    """
    MPG normalisation layer

    This layer normalises the edge weights of a graph. The normalisation is done by dividing by the standard deviation.
    """

    def __init__(self, mask_zeros=False, regularisation=1, name=None):
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
        super(MPGUniformNormalisationLayer, self).__init__(name=name)
        self.mask_zeros = mask_zeros
        self.regularisation = regularisation

    def get_max(self, A, W):
        if self.mask_zeros:
            mask = tf.cast(A, tf.bool)
            return tf.math.reduce_max(tf.ragged.boolean_mask(W, mask), axis=(-2, -1), keepdims=True)
        else:
            return tf.math.reduce_max(W, axis=(-2, -1), keepdims=True)

    def build(self, input_shape):
        super(MPGUniformNormalisationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A, W = inputs
        std = self.get_max(A, W)
        return A, W / (std + self.regularisation)

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
