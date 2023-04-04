import tensorflow as tf
from tensorflow import keras

class EdgeNormalisationLayer(keras.layers.Layer):
    def __init__(self, edges_interval : tuple, **kwargs):
        super(EdgeNormalisationLayer, self).__init__(**kwargs)
        self.edges_interval=edges_interval
        if self.edges_interval == "all":
            self.edges_interval=None

    def build(self, input_shape):
        super(EdgeNormalisationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not self.edges_interval:
            axes=tf.range(1,tf.rank(inputs))
            mu=tf.math.reduce_mean(inputs,axis=axes,keepdims=True)
            var=tf.math.reduce_std(inputs,axis=axes,keepdims=True)
            return (inputs-mu)/var
        else:
            a,b=self.edges_interval
            slice=inputs[...,a:b]
            mu=tf.math.reduce_mean(slice,axis=-1,keepdims=True)
            var=tf.math.reduce_std(slice,axis=-1,keepdims=True)
            return tf.concat([inputs[...,:a],(slice-mu)/var,inputs[...,b:]],axis=-1)
    def compute_output_shape(self, input_shape):
        return input_shape
