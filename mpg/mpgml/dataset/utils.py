import networkx as nx
import numpy as np

from mpg.games import mpg
import tensorflow as tf
import mpg.wrapper as mpgwrapper
import tensorflow_probability as tfp


def cast_all(dtype, *args):
    return tuple(tf.cast(arg, dtype) for arg in args)


def convert_sparse_matrix_to_sparse_tensor(X, shape_hint):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, shape_hint)


def stack_sparse_tensors(shape_hint, *A):
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
        return convert_sparse_matrix_to_sparse_tensor(A, shape_hint)

def string_tensor_to_string(string_tensor):
    return tf.strings.reduce_join(string_tensor, separator="").numpy().decode("utf-8")

def generate_instances(n, p, seed, cardinality: int, target: bool, as_graph: bool,
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
                output = tf.cast(stack_sparse_tensors((2, n, n), A, W), dtype=tf.float32)

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


def gnp_adjacency_matrix(n, p) -> np.ndarray:
    """
    Generate a random adjacency matrix using the G(n,p) model.
    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability of an edge.
    Returns: np.ndarray of shape (n, n)
    """
    A = np.zeros([n, n], dtype=np.uint8)
    for k in range(n):
        A[k, :] = np.random.binomial(1, p, n)
        while A[k, :].sum() == 0:
            A[k, :] = np.random.binomial(1, p, n)
    return A

def _generate_target_impl_int(X,heuristic, target,flatten:bool):
    return np.array(mpgwrapper.mpgcpp.targets_tensorflow_int_cxx (
                X.numpy().astype(np.int32).tolist(),string_tensor_to_string(heuristic),string_tensor_to_string(target),bool(flatten)
            ))

def _generate_target_impl_float(X, heuristic, target, flatten: bool):
        return np.array(mpgwrapper.mpgcpp.targets_tensorflow_float_cxx (
            X.numpy().astype(np.float32).tolist(), string_tensor_to_string(heuristic), string_tensor_to_string(target), bool(flatten)
        ))
def generate_target(X, target, weight_type,flatten:bool):
    args=[X,"dense", target,flatten]
    _generate_target_impl=_generate_target_impl_int if weight_type==tf.int32 or weight_type==tf.int64 else _generate_target_impl_float
    return tf.py_function(_generate_target_impl,
                          inp=args,
                          Tout=tf.int32 if weight_type==tf.int32 or weight_type==tf.int64 else tf.float32)

def _target_ensure_shape(output,target_name,target_value,n):
    if target_name == "strategy":
        return  tf.ensure_shape(target_value, (2, n))
    elif target_name == "winners":
        return tf.ensure_shape(target_value, (2, n))
    elif target_name == "all":
        return tf.ensure_shape(target_value, (2, 2, n))
    else:
        raise ValueError("Invalid target name")
def generate_dense_gnp_instance(n, p, seeder, cardinality: int, target: str, weight_matrix: bool, flatten: bool,
                                weight_distribution: tfp.distributions.Distribution, weight_type):
    shape = (n, n) if not flatten else (n * n,)
    W = weight_distribution.sample(shape, seed=seeder())
    dtype = weight_distribution.dtype
    A = tf.numpy_function(gnp_adjacency_matrix, inp=[n, p], Tout=tf.uint8, stateful=True)
    if flatten:
        A = tf.reshape(A, shape=(n * n,))
    A = tf.cast(A, dtype=dtype)
    W = tf.multiply(A, W)

    if flatten:
        if weight_matrix:
            output = tf.concat(cast_all(dtype, A, W), axis=0)
        else:
            output = tf.concat(cast_all(dtype, A), axis=0)
        if target != "none":
            target_value = generate_target(output, target, weight_type, flatten)
            target_value = _target_ensure_shape(output,target,target_value,n)
            return (tf.cast(output, dtype=tf.float32), tf.cast(target_value, dtype=tf.float32))
        return output
    else:
        if weight_matrix:
            output = tf.cast(tf.stack([A, W], axis=0), dtype=tf.float32)
        else:
            output = tf.cast(A, dtype=tf.float32)
        if target != "none":
            target_value = generate_target(output, target, weight_type, flatten)
            target_value = _target_ensure_shape(output, target, target_value, n)
            return (output, tf.cast(target_value, dtype=tf.float32))
        return output
