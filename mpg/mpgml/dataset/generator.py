import numba
import numpy as np
import tensorflow as tf
import mpg.graph.random_graph
import networkx as nx
import itertools
import tensorflow_probability as tfb
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


def _generate_dense_instances(n, p, seed, cardinality: int, target: bool, weight_matrix: bool, flatten: bool):
    bernoulli: tfb.distributions.Distribution = tfb.distributions.Bernoulli(probs=p)
    # discrete=tfb.distributions.DiscreteUniform(low=0,high=10)
    shape = (n, n) if not flatten else (n * n,)
    W = tf.random.uniform(shape,-10, 11, dtype=tf.int32)
    adjacency_list=[]
    for k in range(n):
        adjacency_list.append(bernoulli.sample((n,)))
        while tf.math.reduce_all(adjacency_list[k]==0):
            adjacency_list[k] = bernoulli.sample((n,))
    A= tf.concat(adjacency_list,0)
    W = tf.multiply(A, W)
    vertex = tf.random.uniform((1,),0, n, dtype=np.int32)
    player = bernoulli.sample((1,))
    if flatten:
        if weight_matrix:
            output = tf.cast(tf.concat([A, W, vertex, player], axis=0), dtype=tf.float32)
        else:
            output = tf.cast(tf.concat([A, W, vertex, player], axis=0), dtype=tf.float32)
        if target:
            target_value=tf.py_function(lambda output:mpgwrapper.mpgcpp.winners_tensorflow_float_matrix_flattened_cxx(output.numpy().tolist()),inp=[output],Tout=tf.int32)
            target_value=tf.reshape(tf.ensure_shape(target_value,()),shape=(1,))
            return (output, tf.cast(target_value,dtype=tf.float32))
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
                target: bool = False, weight_matrix: bool = True, flatten=False,seed=None):
        if seed is None:
            seed = np.random.randint(0, 1<<32)
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
            generated = tf.data.Dataset.range(seed,seed+cardinality)
        return generated.map(
            lambda seed: _generate_dense_instances(n, p, seed, cardinality, target, weight_matrix, flatten),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        #    range,
        #    args=(n,p,cardinality, target, weight_matrix,flatten),
        #    output_signature=signature
        # )

    def __init__(self, n, p):
        pass


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
