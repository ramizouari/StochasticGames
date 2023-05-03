import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp

class GraphIsomorphismLayer(keras.layers.Layer):
    """
    Graph isomorphism layer

    This layer permutes the vertices of a graph.
    """
    def __init__(self,vertices:int,seed=None, **kwargs):
        if seed is None:
            seed=np.random.randint(0,100000)
        self.seeder=tfp.util.SeedStream(seed=seed, salt='GraphIsomorphismLayer')
        super(GraphIsomorphismLayer, self).__init__(**kwargs)
        self.vertices=vertices

    def build(self, input_shape):
        super(GraphIsomorphismLayer, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if not training:
            return inputs
        permutation=tf.random.shuffle(tf.range(self.vertices),seed=self.seeder())
        return tf.gather(tf.gather(inputs,permutation,axis=-2),permutation,axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape


class UniformNoise(keras.layers.Layer):
    """
    Uniform noise layer

    This layer adds uniform noise to the input.
    """
    def __init__(self, noise_level=None, low = None, high=None,**kwargs):
        super(UniformNoise, self).__init__(**kwargs)
        if noise_level is None and (low is None or high is None):
            raise ValueError("noise_level or low and high must be set")
        if noise_level is None:
            self.low=low
            self.high=high
        else:
            self.low=-noise_level
            self.high=noise_level
    def build(self, input_shape):
        super(UniformNoise, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if not training:
            return inputs
        noise=tf.random.uniform(tf.shape(inputs),minval=self.low,maxval=self.high)
        return inputs+noise

    def compute_output_shape(self, input_shape):
        return input_shape

def inverse_mask(mask):
    if isinstance(mask,tf.Tensor):
        if mask.dtype == tf.bool:
            return tf.logical_not(mask)
        elif mask.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16, tf.int32, tf.int64]:
            return 1 ^ mask
        else:
            return 1- mask
    else:
        return 1-mask
class EdgeWeightsNoiseLayer(keras.layers.Layer):
    """
    Edge weights noise layer

    This layer adds noise to the edge weights of a graph.
    """
    def __init__(self, noise_layer,edges_interval=None,edges_matrix=None, mask_zeros=False,
                 mask_tensor: int = None,masked_assume_zero:bool=True, **kwargs):
        """
        :param noise_layer: The noise layer to use. Can be a string or a keras layer.
        :param edges_interval: The interval of the edge weights to add noise to.
        :param edges_matrix: The index of the edge weights to add noise to.
        :param mask_zeros: If true, the noise is only added to non-zero edge weights.
        :param mask_tensor: The index of the tensor to use as a mask. If set, the noise is only added to the edge weights
        where the mask tensor is not zero.
        :param masked_assume_zero: If true, assume the masked edge weights are zero.
        :param kwargs: Additional arguments for the noise layer.
        """
        super(EdgeWeightsNoiseLayer, self).__init__(**kwargs)
        self.noise_layer=noise_layer
        if noise_layer == "gaussian":
            self.noise_layer = tf.keras.layers.GaussianNoise(**kwargs)
        elif noise_layer == "uniform":
            self.noise_layer = UniformNoise(**kwargs)
        elif isinstance(noise_layer,type):
            self.noise_layer = noise_layer(**kwargs)
        if not isinstance(noise_layer,tf.keras.layers.Layer):
            raise TypeError("noise_layer must be a keras layer")
        self.edges_interval=edges_interval
        self.edges_matrix=edges_matrix
        if edges_interval is not None and edges_matrix is not None:
            raise ValueError("edges_interval and edges_matrix can not be set at the same time")
        self.mask_zeros=mask_zeros
        self.mask_tensor=mask_tensor
        self.masked_assume_zero=masked_assume_zero
        if self.mask_zeros:
            self.masked_assume_zero=True

    def build(self, input_shape):
        super(EdgeWeightsNoiseLayer, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if not training:
            return inputs

        mask=tf.constant(1.)

        if self.edges_matrix is not None:

            submatrix = inputs[..., self.edges_matrix:self.edges_matrix + 1, :, :]
            if self.mask_zeros:
                mask = tf.cast(submatrix != 0, dtype=tf.float32)
            if self.mask_tensor is not None:
                mask = tf.cast(inputs[..., self.mask_tensor:self.mask_tensor + 1, :, :], dtype=tf.float32)
            result=mask*self.noise_layer(submatrix)
            if not self.masked_assume_zero:
                result+=inverse_mask(mask)*submatrix
            return tf.concat(
                [inputs[..., :self.edges_matrix, :, :], result,
                 inputs[..., self.edges_matrix + 1:, :, :]],
                axis=-3)
        elif self.edges_interval is not None:
            a,b=self.edges_interval
            slice=inputs[...,a:b]
            noise=self.noise_layer(slice)
            return tf.concat([inputs[...,:a],noise+slice,inputs[...,b:]],axis=-1)
        else:
            if self.mask_zeros:
                mask = tf.cast(inputs != 0, dtype=tf.float32)
            if self.mask_tensor is not None:
                mask = tf.cast(inputs[..., self.mask_tensor:self.mask_tensor + 1, :, :],dtype=tf.float32)
            if self.masked_assume_zero:
                return mask*self.noise_layer(inputs)
            else:
                return mask*self.noise_layer(inputs)+inverse_mask(mask)*inputs
    def compute_output_shape(self, input_shape):
        return input_shape