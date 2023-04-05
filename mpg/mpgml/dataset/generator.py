import numba
import numpy as np
import tensorflow as tf
import mpg.graph.random_graph
import networkx as nx
import itertools
import tensorflow_probability as tfp
import mpg.wrapper as mpgwrapper
from . import utils




class MPGGeneratedDenseDataset(tf.data.Dataset):

    def __new__(cls, n, p, cardinality=tf.data.INFINITE_CARDINALITY,
                target: str = None, weight_matrix: bool = True, flatten=False, seed=None,
                weights_distribution: tfp.distributions.Distribution = None,
                weight_type : str = "int"):
        if target is None or target == False:
            target = "none"
        elif target ==True:
            target="all"
        if weight_type == "int":
            weight_type= tf.int32
        elif weight_type == "float":
            weight_type = tf.float32
        elif weight_type == "double":
            weight_type = tf.float64
        elif not isinstance(weight_type, tf.DType):
            raise ValueError("weight_type must be a string or a tf.DType")
        if weights_distribution is None:
            weights_distribution = tfp.distributions.Uniform(low=-1, high=1)
        if seed is None:
            seed = np.random.randint(0, 1 << 32)

        seeder = tfp.util.SeedStream(seed, "seeding_generator")

        shape = None
        if flatten:
            if weight_matrix:
                shape = (2 * n * n + 2,)
            else:
                shape = (n * n + 2,)
            signature = (tf.TensorSpec(shape=shape, dtype=tf.float32),)
        else:
            if weight_matrix:
                shape = (2, n, n)
            else:
                shape = (n, n)
            signature = (tf.TensorSpec(shape=shape, dtype=tf.float32), tf.TensorSpec(shape=(2,), dtype=tf.int32))
        if target:
            signature = (*signature, tf.TensorSpec(shape=()))

        generated: tf.data.Dataset
        if cardinality == tf.data.INFINITE_CARDINALITY:
            generated = tf.data.Dataset.counter(start=seed, step=1)
        else:
            generated = tf.data.Dataset.range(seed, seed + cardinality)
        return generated.map(
            lambda seed: utils.generate_dense_gnp_instance(n, p, seeder, cardinality, target, weight_matrix, flatten,
                                                           weights_distribution, weight_type),
            num_parallel_calls=12
        )
        #    range,
        #    args=(n,p,cardinality, target, weight_matrix,flatten),
        #    output_signature=signature
        # )

    def __init__(self, n, p, cardinality=tf.data.INFINITE_CARDINALITY,
                 target: bool = False, weight_matrix: bool = True, flatten=False, seed=None,
                 weights_distribution: tfp.distributions.Distribution = None,
                 weight_type : str = "int"):
        self.n = n
        self.p = p
        self.cardinality = cardinality
        self.target = target
        self.weight_matrix = weight_matrix
        self.flatten = flatten
        self.seed = seed
        self.weights_distribution = weights_distribution

    def _permutation(self, x, P):
        if self.flatten:
            S = tf.concat(tf.reshape(tf.tensordot(P, P, axes=None), shape=(-1,)))
            return tf.concat([tf.gather(x, S, axis=0), P[x[-2]], x[-1]])

    def permutation(self, P):
        return self.map(lambda x: self._permutation(x, P))


class MPGGeneratedDataset(tf.data.Dataset):
    def _generator(n, p, cardinality: int, target: bool, as_graph: bool,
                   adj_matrix: bool, weight_matrix: bool, as_dense: bool):
        if cardinality == tf.data.INFINITE_CARDINALITY:
            seed = 0
            while True:
                yield utils.generate_instances(n, p, seed, cardinality, target, as_graph, adj_matrix, weight_matrix,
                                          as_dense)
                seed += 1
        else:
            for sample_idx in range(cardinality):
                yield utils.generate_instances(n, p, sample_idx, cardinality, target, as_graph, adj_matrix, weight_matrix,
                                          as_dense)

    def __new__(cls, n, p, cardinality=tf.data.INFINITE_CARDINALITY, target: bool = False, as_graph: bool = False,
                adj_matrix: bool = True, weight_matrix: bool = True, as_dense: bool = True):
        shape = None
        if as_graph:
            signature = tf.TensorSpec(shape=(), dtype=mpg.MeanPayoffGraph)
        else:
            if adj_matrix and weight_matrix:
                shape = (2, n, n)
            elif adj_matrix or weight_matrix:
                shape = (n, n)
            else:
                raise ValueError("Must specify at least one of adj_matrix or weight_matrix")
        if as_dense:
            TensorSpec = tf.TensorSpec
        else:
            TensorSpec = tf.SparseTensorSpec
        signature = (TensorSpec(shape=shape), tf.TensorSpec(shape=(2,), dtype=tf.int32))
        if target:
            signature = (*signature, tf.TensorSpec(shape=()))
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(n, p, cardinality, target, as_graph, adj_matrix, weight_matrix, as_dense),
            output_signature=signature
        )

    def __init__(self, n, p):
        pass
