from typing import Callable

import tensorflow as tf
from . import aggregator



class GrpahConv(tf.keras.layers.Layer):
    def __init__(self,filters, loops=False, activation=None, normalization=None, bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zero",trainable=True, name="GraphConv"):
        super().__init__(name=name)
        self.kernel_initializer: Callable
        if type(kernel_initializer) == str:
            self.kernel_initializer = tf.initializers.get(kernel_initializer)
        else:
            self.kernel_initializer: Callable = kernel_initializer
        if type(bias_initializer) == str or type(bias_initializer) == dict:
            self.bias_initializer = tf.initializers.get(bias_initializer)
        else:
            self.bias_initializer = bias_initializer
        if type(activation) == str:
            activation = tf.keras.layers.Activation(activation)
        elif activation is None:
            activation = tf.keras.activations.linear
        self.activation = activation
        self.loops = loops
        self.normalization = normalization
        self.filters = filters
        self.bias = bias
        self.trainable=trainable
        self.aggregation=aggregator.SumAggregation(normalization=normalization,with_weights=False)

    def build(self, input_shape):
        adjacency_matrix_shape,weights_matrix, data_shape = input_shape
        C = data_shape[-1]
        r=2 if self.loops else 1
        self.W = tf.Variable(self.kernel_initializer(shape=(C, self.filters,r)), trainable=self.trainable, name="kernel")
        if self.bias:
            self.b = tf.Variable(self.bias_initializer(shape=(self.filters,)), trainable=self.trainable, name="bias")

    def call(self, inputs, **kwargs):
        adjacency_matrix,weights_matrix, data = inputs
        output = tf.linalg.matmul(self.W[:,:,0],tf.linalg.matmul(weights_matrix, data))
        if self.loops:
            output+=tf.linalg.matmul(self.W[:,:,1],data)
        if self.bias:
            output += self.b
        return self.activation(output)

class GCN(tf.keras.layers.Layer):
    def __init__(self, filters, loops=False, activation=None, normalization=None, kernel_initializer="he_normal"
                 , trainable=True,name="GCN"):
        super().__init__(name=name)
        if type(kernel_initializer) == str or type(kernel_initializer) == dict:
            self.kernel_initializer = tf.initializers.get(kernel_initializer)
        else:
            self.kernel_initializer = kernel_initializer
        if type(activation) == str:
            activation = tf.keras.layers.Activation(activation)
        elif activation is None:
            activation = tf.keras.activations.linear
        self.activation = activation
        self.loops = loops

        self.normalization = normalization
        self.filters = filters
        self.trainable=trainable
        self.aggregation=aggregator.SumAggregation(normalization=normalization,with_weights=False)
    def build(self, input_shape):
        adjacency_matrix_shape, data_shape = input_shape
        N = adjacency_matrix_shape[0]
        C = data_shape[-1]
        self.W = tf.Variable(self.kernel_initializer(shape=(C, self.filters)), trainable=self.trainable, name="kernel")

    def call(self, inputs, **kwargs):
        output=tf.linalg.matmul(self.aggregation(inputs), self.W)
        if self.bias:
            output+=self.b
        if self.activation is not None:
            output=self.activation(output)

        return output


class WGCN(tf.keras.layers.Layer):
    def __init__(self, filters, loops=False, activation=None, normalization=None, bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zero",trainable=True, name="WGCN"):
        super().__init__(name=name)
        self.kernel_initializer: Callable
        if type(kernel_initializer) == str:
            self.kernel_initializer = tf.initializers.get(kernel_initializer)
        else:
            self.kernel_initializer: Callable = kernel_initializer
        if type(bias_initializer) == str or type(bias_initializer) == dict:
            self.bias_initializer = tf.initializers.get(bias_initializer)
        else:
            self.bias_initializer = bias_initializer
        if type(activation) == str:
            activation = tf.keras.layers.Activation(activation)
        elif activation is None:
            activation = tf.keras.activations.linear
        self.activation = activation
        self.loops = loops
        self.normalization = normalization
        self.filters = filters
        self.bias = bias
        self.trainable=trainable
        self.aggregation=aggregator.SumAggregation(normalization=normalization,with_weights=True)

    def build(self, input_shape):
        adjacency_matrix_shape, weight_matrix_shape, data_shape = input_shape
        N = adjacency_matrix_shape[0]
        C = data_shape[-1]
        self.W = tf.Variable(self.kernel_initializer(shape=(C, self.filters)), trainable=self.trainable, name="kernel")
        if self.bias:
            self.b = tf.Variable(self.bias_initializer(shape=(self.filters)), trainable=self.trainable, name="bias")

    def call(self, inputs,**kwargs):
        output = tf.linalg.matmul(self.aggregation(inputs), self.W)
        if self.bias:
            output += self.b
        if self.activation is not None:
            output = self.activation(output)
        return output
