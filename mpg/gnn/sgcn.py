import tensorflow as tf

from . import aggregator

class TAGConv(tf.keras.layers.Layer):
    def __init__(self,filters,K,activation=None,kernel_initializer="he_normal",loops=True,name="TAGConv",normalization="unit"):
        super().__init__(name=name)
        if type(kernel_initializer)==str or type(kernel_initializer)==dict:
            self.kernel_initializer=tf.initializers.get(kernel_initializer)
        else:
            self.kernel_initializer=kernel_initializer
        if type(activation)==str:
            activation=tf.keras.layers.Activation(activation)
        elif activation is None:
            activation=tf.keras.activations.linear
        self.activation=activation
        self.K=K
        self.filters=filters
        self.aggregator=aggregator.SumAggregation(normalization=normalization,with_weights=True)
        self.loops=loops

    def build(self,input_shape):
        adjacency_matrix_shape,weights_matrix,data_shape=input_shape
        C=data_shape[-1]
        self.W=tf.Variable(self.kernel_initializer(shape=(C,self.filters,self.K+1)),trainable=True,name="kernel")

    def call(self,inputs,**kwargs):
        adjacency_matrices,weights_matrix,data=inputs
        if self.loops:
            weights_matrix=adjacency_matrices + tf.eye(num_rows=adjacency_matrices.shape[-1],batch_shape=adjacency_matrices.shape[:-2])
        output=tf.matmul(self.aggregator((adjacency_matrices,weights_matrix,data)),self.W[:,:,0])
        for k in range(1,self.K+1):
            data=self.aggregator((adjacency_matrices,weights_matrix,data))
            output+=tf.matmul(data,self.W[:,:,k])
        return self.activation(output)


class SGCN(tf.keras.layers.Layer):
    def __init__(self,filters,K,loops=False,activation=None,normalization=None,kernel_initializer="he_normal",name="SGCN"):
        super().__init__(name=name)
        if type(kernel_initializer)==str or type(kernel_initializer)==dict:
            self.kernel_initializer=tf.initializers.get(kernel_initializer)
        else:
            self.kernel_initializer=kernel_initializer
        if type(activation)==str:
            activation=tf.keras.layers.Activation(activation)
        elif activation is None:
            activation=tf.keras.activations.linear
        self.activation=activation
        self.loops=loops

        self.normalization=normalization
        self.filters=filters
        self.K=K
        self.aggregator=aggregator.SumAggregation(loops=False,normalization=normalization,with_weights=False)

    def build(self,input_shape):
        adjacency_matrix_shape,data_shape=input_shape
        N=adjacency_matrix_shape[0]
        C=data_shape[-1]
        self.W=tf.Variable(self.kernel_initializer(shape=(C,self.filters)),trainable=True,name="kernel")

    def call(self,inputs,**kwargs):
        adjacency_matrices,weights_matrices,data=inputs
        if self.loops:
            weights_matrices=weights_matrices + tf.eye(num_rows=adjacency_matrices.shape[-1],batch_shape=adjacency_matrices.shape[:-2])

        output=data
        for k in range(self.K):
            output=self.aggregator((adjacency_matrices,weights_matrices,output))
        output=tf.linalg.matmul(output,self.W)
        if self.activation is not None:
            output=self.activation(output)
        return output


class SSGCN(tf.keras.layers.Layer):
    def __init__(self,filters,K,alpha,loops=False,activation=None,normalization=None,kernel_initializer="he_normal",name="SSGCN"):
        super().__init__(name=name)
        if type(kernel_initializer)==str or type(kernel_initializer)==dict:
            self.kernel_initializer=tf.initializers.get(kernel_initializer)
        else:
            self.kernel_initializer=kernel_initializer
        if type(activation)==str:
            activation=tf.keras.layers.Activation(activation)
        elif activation is None:
            activation=tf.keras.activations.linear
        self.activation=activation
        self.loops=loops
        self.alpha=alpha

        self.normalization=normalization
        self.filters=filters
        self.K=K
        self.aggregator=aggregator.SumAggregation(normalization=normalization,with_weights=True)

    def build(self,input_shape):
        adjacency_matrix_shape,weights_matrix,data_shape=input_shape
        N=adjacency_matrix_shape[0]
        C=data_shape[-1]
        self.W=tf.Variable(self.kernel_initializer(shape=(C,self.filters)),trainable=True,name="kernel")

    def call(self,inputs,**kwargs):
        adjacency_matrices,weights_matrices,data=inputs
        if self.loops:
            weights_matrices=weights_matrices + tf.eye(num_rows=adjacency_matrices.shape[-1],batch_shape=adjacency_matrices.shape[:-2])

        smoothing=self.aggregator((adjacency_matrices,weights_matrices,data))
        for k in range(self.K-1):
            smoothing+=self.aggregator((adjacency_matrices,weights_matrices,smoothing))

        output=tf.matmul(self.alpha * data + (1-self.alpha)*smoothing / self.K,self.W)
        if self.activation is not None:
            output=self.activation(output)
        return self.activation(output)