import abc

import tensorflow as tf


class Aggregation(tf.Module,abc.ABC):
    def __init__(self,name=None):
        super().__init__(name=name)

    @abc.abstractmethod
    def call(self,inputs,**kwargs):
        pass

    def __call__(self,inputs,**kwargs):
        return self.call(inputs,**kwargs)



class SumAggregation(Aggregation):
    def __init__(self,loops=True,with_weights=False,normalization="mean",name=None):
        super().__init__(name=name)
        self.with_weights=with_weights
        self.loops=loops
        self.normalization=normalization
    def call(self,inputs,**kwargs):
        if self.with_weights:
            adjacency_matrices,weights_matrices,data=inputs
            matrices=weights_matrices
            size=tf.shape(weights_matrices)[-1]
            batches=tf.shape(weights_matrices)[:-2]
            if self.loops:
                matrices = matrices + tf.eye(num_rows=size, batch_shape=batches)
            degrees=tf.norm(weights_matrices,axis=-1,keepdims=True)
        else:
            adjacency_matrices,data=inputs
            matrices=adjacency_matrices
            size = tf.shape(adjacency_matrices)[-1]
            batches = tf.shape(adjacency_matrices)[:-2]
            if self.loops:
                matrices = matrices + tf.eye(num_rows=size, batch_shape=batches)
            degrees=tf.reduce_sum(matrices,axis=-1,keepdims=True)
        if self.normalization in ["mean","unit"]:
            return tf.linalg.matmul(matrices, data / tf.math.sqrt(degrees))/tf.math.sqrt(degrees)
        elif self.normalization == "mean":
            return tf.linalg.matmul(matrices, data) / degrees
        elif self.normalization is None:
            return tf.linalg.matmul(matrices, data)
        else:
            raise RuntimeError("Not Supported yet")
