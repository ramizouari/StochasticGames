from tensorflow import keras
import tensorflow as tf

class GraphPaddingLayer(keras.layers.Layer):
    def __init__(self,vertices:int, **kwargs):
        super(GraphPaddingLayer, self).__init__(**kwargs)
        self.vertices=vertices
    def build(self, input_shape):
        super(GraphPaddingLayer, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        return self.right_bottom_pad(inputs)

    def right_bottom_pad(self,inputs):
        shape = tf.shape(inputs)
        right_padshape = tf.stack([*shape[:-2], self.vertices - shape[1], shape[2]])
        output = tf.concat([inputs, tf.zeros(right_padshape)], axis=-2)
        bottom_padshape = tf.stack([*shape[:-2], self.vertices, self.vertices - shape[2]])
        output = tf.concat([output, tf.zeros(bottom_padshape)], axis=-1)
        return output
    def compute_output_shape(self, input_shape):
        output_shape=tf.stack([*input_shape[:-2],self.vertices,self.vertices])
        return output_shape