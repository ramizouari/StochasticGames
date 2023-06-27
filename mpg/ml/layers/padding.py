from tensorflow import keras
import tensorflow as tf

class GraphPaddingLayer(keras.layers.Layer):
    """
    Graph padding layer

    This layer pads a graph with zeros to a given size. The padding is done on the right and bottom.
    This layer is used to make all graphs in a batch the same size.
    """
    def __init__(self,vertices:int, **kwargs):
        super(GraphPaddingLayer, self).__init__(**kwargs)
        self.vertices=vertices
    def build(self, input_shape):
        super(GraphPaddingLayer, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        return self.right_bottom_pad(inputs)

    def right_bottom_pad(self,inputs):
        shape = tf.shape(inputs)
        right_padshape = tf.concat([shape[:-2],[self.vertices - shape[-2], shape[-1]]],axis=0)
        output = tf.concat([inputs, tf.zeros(right_padshape)], axis=-2)
        bottom_padshape = tf.concat([shape[:-2], [self.vertices, self.vertices - shape[-1]]],axis=0)
        output = tf.concat([output, tf.zeros(bottom_padshape)], axis=-1)
        return output
    def compute_output_shape(self, input_shape):
        output_shape=tf.stack([*input_shape[:-2],self.vertices,self.vertices])
        return output_shape


class VacuousConnections(tf.keras.layers.Layer):

    def __init__(self,name=None):
        super().__init__(name=name)


    def call(self,inputs):
        # Get the nodes that have at least one outgoing edge.
        dtype=inputs.dtype
        inputs=tf.cast(inputs,dtype=tf.bool)
        Z=tf.reduce_any(inputs,axis=-1,name="extract_nodes_with_outgoing_edges")
        C=tf.linalg.diag(~Z,name="create_vacuous_connection_matrix")
        return tf.cast(tf.logical_or(inputs,C,name="add_vacuous_connections"),dtype=dtype)


