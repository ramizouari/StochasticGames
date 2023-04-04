from tensorflow import keras
import tensorflow as tf


class GraphIsomorphismLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GraphIsomorphismLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GraphIsomorphismLayer, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if not training:
            return inputs
        permutation=tf.random_index_shuffle(tf.range(tf.shape(inputs)[1]))
        pass

    def compute_output_shape(self, input_shape):
        return input_shape


class UniformNoise(keras.layers.Layer):
    def __init__(self, noise_level=None, low = None, hight=None,**kwargs):
        super(UniformNoise, self).__init__(**kwargs)
        if noise_level is None and (low is None or hight is None):
            raise ValueError("noise_level or low and hight must be set")
        if noise_level is None:
            self.low=low
            self.hight=hight
        else:
            self.low=-noise_level
            self.hight=noise_level
    def build(self, input_shape):
        super(UniformNoise, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if not training:
            return inputs
        noise=tf.random.uniform(tf.shape(inputs),minval=-self.noise_level,maxval=self.noise_level)
        return inputs+noise

    def compute_output_shape(self, input_shape):
        return input_shape

class EdgeWeightsNoiseLayer(keras.layers.Layer):
    def __init__(self, noise_layer,edges_interval, **kwargs):
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
        if self.edges_interval == "all":
            self.edges_interval=None
    def build(self, input_shape):
        super(EdgeWeightsNoiseLayer, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if not training:
            return inputs
        if not self.edges_interval:
            return self.noise_layer(inputs)+inputs
        else:
            a,b=self.edges_interval
            slice=inputs[...,a:b]
            noise=self.noise_layer(slice)
            return tf.concat([inputs[...,:a],noise+slice,inputs[...,b:]],axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape