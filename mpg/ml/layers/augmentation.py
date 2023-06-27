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
    def __init__(self, noise_layer,mask_zeros=True, **kwargs):
        """
        :param noise_layer: The noise layer to use. Can be a string or a keras layer.
        :param mask_zeros: If true, the noise is only added to non-zero edge weights.
        where the mask tensor is not zero.
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
        else:
            self.noise_layer = noise_layer
        if not isinstance(noise_layer,tf.keras.layers.Layer):
            raise TypeError("noise_layer must be a keras layer")

        self.mask_zeros=mask_zeros


    def build(self, input_shape):
        super(EdgeWeightsNoiseLayer, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if not training:
            return inputs

        A,W = inputs
        if self.mask_zeros:
            mask = A > 0.5
            W = tf.where(mask, self.noise_layer(W), W)
        else:
            W = self.noise_layer(W)
        return A,W
    def compute_output_shape(self, input_shape):
        return input_shape



class RandomConnectionLayer(keras.layers.Layer):
    """
    Random connection layer

    This layer randomly connects nodes in a graph.
    """

    def __init__(self, p=0.01, degree=None, seed=None, name=None):
        """
        :param p: The probability of a connection between two nodes.
        :param kwargs: The keyword arguments of the keras layer.
        """
        super(RandomConnectionLayer, self).__init__(name=name)
        if p is None and degree is None:
            raise ValueError("Either p or degree should be specified")
        if p is not None and degree is not None:
            raise ValueError("Only one of p and degree should be specified")
        if p is not None:
            if p < 0 or p > 1:
                raise ValueError("p should be in the interval [0,1]")
        if degree is not None:
            if degree < 0:
                raise ValueError("degree should be positive")
        self.p = p
        self.degree = degree
        if seed is None:
            seed=tf.random.uniform(shape=(),minval=0,maxval=100000,dtype=tf.int32)
        self.seed = seed


    def _get_connection_matrix(self, A):
        if self.degree is not None:
            raise NotImplementedError()
        else:
            return tf.maximum(tf.cast(tf.random.uniform(tf.shape(A)) < self.p, A.dtype), A)


    def build(self, input_shape):
        super(RandomConnectionLayer, self).build(input_shape)

    def call(self, inputs,training=False, **kwargs):
        if not training:
            return inputs

        A,W = inputs
        A = self._get_connection_matrix(A)
        return A,W

    def compute_output_shape(self, input_shape):
        return input_shape

