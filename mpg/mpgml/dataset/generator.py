import numba
import numpy as np
import tensorflow as tf
import mpg.graph.random_graph
import networkx as nx
import itertools
import tensorflow_probability as tfp
import mpg.wrapper as mpgwrapper


def _convert_sparse_matrix_to_sparse_tensor(X, shape_hint):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, shape_hint)


def _stack_sparse_tensors(shape_hint, *A):
    indices = []
    values = []
    for i, Z in enumerate(A):
        for L, v in zip(Z.indices, Z.values):
            indices.append([i, *L])
            values.append(v)
    return tf.sparse.SparseTensor(indices, values, shape_hint)


def _as_tensor(A, as_dense: bool, shape_hint=None):
    if as_dense:
        return tf.convert_to_tensor(A.todense())
    else:
        return _convert_sparse_matrix_to_sparse_tensor(A, shape_hint)


def _generate_instances(n, p, seed, cardinality: int, target: bool, as_graph: bool,
                        adj_matrix: bool, weight_matrix: bool, as_dense: bool):
    generator = np.random.Generator(np.random.MT19937(seed))
    graph = mpg.graph.random_graph.gnp_random_mpg(n=n, p=p, seed=seed, method="fast", loops=True,
                                                  distribution="integers", low=0, high=10, endpoint=True)
    output = None
    if as_graph:
        output = graph
    else:
        if adj_matrix and weight_matrix:
            A = _as_tensor(nx.adjacency_matrix(graph, weight=None), as_dense=as_dense, shape_hint=(n, n))
            W = _as_tensor(nx.adjacency_matrix(graph, weight="weight"), as_dense=as_dense, shape_hint=(n, n))
            if as_dense:
                output = tf.stack([A, W], axis=0)
            else:
                output = tf.cast(_stack_sparse_tensors((2, n, n), A, W), dtype=tf.float32)

        elif adj_matrix:
            output = tf.cast(_as_tensor(nx.adjacency_matrix(graph, weight=None), as_dense=as_dense, shape_hint=(n, n)),
                             dtype=tf.float32)
        elif weight_matrix:
            output = tf.cast(
                _as_tensor(nx.adjacency_matrix(graph, weight="weight"), as_dense=as_dense, shape_hint=(n, n)),
                dtype=tf.float32)
    starting = tf.constant([generator.integers(0, n), generator.integers(0, 1, endpoint=True)])
    if target:
        # TODO: Add target
        return (output, starting, 1)
    else:
        return (output, 1)

def cast_all(dtype,*args):
    return tuple(tf.cast(arg, dtype) for arg in args)

def _generate_dense_instances(n, p, seeder, cardinality: int, target: bool, weight_matrix: bool, flatten: bool,
                              weight_distribution: tfp.distributions.Distribution, weight_type):
    adjacency_distribution: tfp.distributions.Distribution = tfp.distributions.Bernoulli(probs=p)
    turn_distribution: tfp.distributions.Distribution = tfp.distributions.Bernoulli(probs=0.5)
    # discrete=tfp.distributions.DiscreteUniform(low=0,high=10)
    shape = (n, n) if not flatten else (n * n,)
    W = weight_distribution.sample(shape, seed=seeder())
    dtype=weight_distribution.dtype
    adjacency_list = []
    for k in range(n):
        adjacency_list.append(adjacency_distribution.sample((n,), seed=seeder()))
        while tf.math.reduce_all(adjacency_list[k] == 0):
            adjacency_list[k] = adjacency_distribution.sample((n,), seed=seeder())
    A = tf.cast(tf.concat(adjacency_list, 0),dtype=dtype)
    W = tf.multiply(A, W)
    vertex = tf.random.uniform((1,), 0, n, dtype=tf.int32, seed=seeder())
    player = turn_distribution.sample((1,), seed=seeder())
    if flatten:
        if weight_matrix:
            output = tf.concat(cast_all(dtype,A, W, vertex, player), axis=0)
        else:
            output = tf.concat(cast_all(dtype,A, vertex, player), axis=0)
        if target:
            if weight_type == tf.int32 or weight_type == tf.int64:
                target_value = tf.py_function(
                    lambda output: mpgwrapper.mpgcpp.winners_tensorflow_int_matrix_flattened_cxx(output.numpy().astype(np.int32).tolist()),
                    inp=[output], Tout=tf.int32)
            else:
                target_value = tf.py_function(
                    lambda output: mpgwrapper.mpgcpp.winners_tensorflow_float_matrix_flattened_cxx(output.numpy().astype(np.float32).tolist()),
                    inp=[output], Tout=tf.float32)
            target_value = tf.reshape(tf.ensure_shape(target_value, ()), shape=(1,))
            return (tf.cast(output,dtype=tf.float32), tf.cast(target_value, dtype=tf.float32))
        return output
    else:
        if weight_matrix:
            output = tf.cast(tf.stack([A, W], axis=0), dtype=tf.float32)
        else:
            output = tf.cast(A, dtype=tf.float32)
        if target:
            return (output, tf.constant([vertex, player]), 1)
        return (output, tf.constant([vertex, player]))


class MPGGeneratedDenseDataset(tf.data.Dataset):

    def __new__(cls, n, p, cardinality=tf.data.INFINITE_CARDINALITY,
                target: bool = False, weight_matrix: bool = True, flatten=False, seed=None,
                weights_distribution: tfp.distributions.Distribution = None,
                weight_type : str = "int"):
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
            lambda seed: _generate_dense_instances(n, p, seeder, cardinality, target, weight_matrix, flatten,
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
                yield _generate_instances(n, p, seed, cardinality, target, as_graph, adj_matrix, weight_matrix,
                                          as_dense)
                seed += 1
        else:
            for sample_idx in range(cardinality):
                yield _generate_instances(n, p, sample_idx, cardinality, target, as_graph, adj_matrix, weight_matrix,
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
