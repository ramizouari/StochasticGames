from functools import reduce
from typing import Union, Tuple

import networkx as nx
import numpy as np

from mpg.games import mpg
import tensorflow as tf
import mpg.wrapper as mpgwrapper
import tensorflow_probability as tfp


def cast_all(dtype, *args):
    """
    Casts all arguments to the given dtype
    :param dtype: The dtype to cast to
    :param args: The arguments to cast
    :return: A tuple of the cast arguments
    """
    return tuple(tf.cast(arg, dtype) for arg in args)


def convert_sparse_matrix_to_sparse_tensor(X, shape_hint):
    """
    Converts a sparse matrix to a sparse tensor
    :param X: The sparse matrix
    :param shape_hint: The shape of the sparse tensor
    :return: The sparse tensor
    """
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, shape_hint)


def stack_sparse_tensors(shape_hint, *A):
    """
    Stacks sparse tensors
    :param shape_hint: The shape of the stacked sparse tensor
    :param A: The sparse tensors to stack
    :return: The stacked sparse tensor
    """
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
    """
    Converts a string tensor to a string
    :param string_tensor: The string tensor
    :return: The string
    :detail: This is a workaround for getting the raw string from a string tensor, as str(string_tensor) returns the
    string representation of the tensor, which is not what we want.
    """
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

def gnp_adjacency_list(n, p) -> Tuple[np.ndarray, int]:
    """
    Generate a random sparse adjacency matrix using the G(n,p) model.
    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability of an edge.
    Returns: np.ndarray of shape (n, n)
    """
    I=[]
    for k in range(n):
        deg=np.random.binomial(n, p)
        while deg==0:
            deg=np.random.binomial(n, p)
        J=np.random.choice(n, deg, replace=False)
        I.extend([[k,r] for r in J])

    return np.array(I, dtype=np.int64), len(I)


def _generate_target_impl_int(X, heuristic: str, target: str, flatten: bool):
    """
Generates a target for the given heuristic and target, using int32

The target is generated using the C++ implementation of the arc consistency algorithm, using int32
:param X: The input
:param heuristic: The used heuristic
:param target: The target to generate. One of "winners", "strategy", "all"
:param flatten: Whether the input is flattened
"""
    return np.array(mpgwrapper.mpgcpp.targets_tensorflow_int_cxx(
        X.numpy().astype(np.int32).tolist(), string_tensor_to_string(heuristic), string_tensor_to_string(target),
        bool(flatten)
    ))


def _generate_target_impl_float(X, heuristic, target, flatten: bool):
    """
Generates a target for the given heuristic and target, using float32.

The target is generated using the C++ implementation of the arc consistency algorithm, using float32
:param X: The input
:param heuristic: The used heuristic
:param target: The target to generate. One of "winners", "strategy", "all"
:param flatten: Whether the input is flattened
    """
    return np.array(mpgwrapper.mpgcpp.targets_tensorflow_float_cxx(
        X.numpy().astype(np.float32).tolist(), string_tensor_to_string(heuristic), string_tensor_to_string(target),
        bool(flatten)
    ))


def generate_target(X, target, weight_type, flatten: bool):
    """
Generates a target for the given heuristic and target.

The target is generated using the C++ implementation of the arc consistency algorithm
:param X: The input
:param heuristic: The used heuristic
:param target: The target to generate. One of "winners", "strategy", "all"
:param flatten: Whether the input is flattened
    """
    args = [X, "dense", target, flatten]
    _generate_target_impl = _generate_target_impl_int if weight_type == tf.int32 or weight_type == tf.int64 else _generate_target_impl_float
    return tf.py_function(_generate_target_impl,
                          inp=args,
                          Tout=tf.int32 if weight_type == tf.int32 or weight_type == tf.int64 else tf.float32)


def get_target(target_dataset,filename, target, flatten: bool):
    """
    Gets the target from the dataset
    """
    match target:
        case "strategy":
            R= np.array(target_dataset.loc[filename,["max_strategy","min_strategy"]].to_list()).T
        case "winners":
            return np.array(target_dataset.loc[filename,["winners_max","winners_min"]].to_list()).T
        case "both":
            A=np.array(target_dataset.loc[filename,["winners_max","winners_min"]].to_list()).T
            B=np.array(target_dataset.loc[filename,["max_strategy","min_strategy"]].to_list()).T
            R= np.array([A,B])
        case _:
            raise ValueError("Invalid target")
    if flatten:
        R=R.flatten()
    return R

def _target_ensure_shape(output, target_name, target_value, n):
    """
Ensures that the shape of the target is correct
:param output: The output of the model
:param target_name: The name of the target
:param target_value: The value of the target
:param n: The number of nodes
"""
    if target_name == "strategy":
        return tf.ensure_shape(target_value, (2, n))
    elif target_name == "winners":
        return tf.ensure_shape(target_value, (2, n))
    elif target_name == "all":
        return tf.ensure_shape(target_value, (2, 2, n))
    else:
        raise ValueError("Invalid target name")


def generate_dense_gnp_instance(n, p, seeder, cardinality: int, target: str, generated_input: str, flatten: bool,
                                weight_distribution: tfp.distributions.Distribution, weight_type):
    """
Generates a dense G(n,p) instance with tensorflow
:param n: The number of nodes
:param p: The probability of an edge
:param seeder: The seeder to generate PRNG seeds
:param cardinality: The cardinality of the instance
:param target: The target to generate. One of "winners", "strategy", "all", "none"
:param generated_input: The input to generate. One of "adjacency", "weight", "both"
:param flatten: Whether to flatten the output
:param weight_distribution: The distribution to use for the weights
:param weight_type: The type of the weights
"""
    shape = (n, n) if not flatten else (n * n,)
    W = weight_distribution.sample(shape, seed=seeder())
    dtype = weight_distribution.dtype
    A = tf.numpy_function(gnp_adjacency_matrix, inp=[n, p], Tout=tf.uint8, stateful=True)
    A = tf.ensure_shape(A, shape=(n, n))
    if flatten:
        A = tf.reshape(A, shape=(n * n,))
    A = tf.cast(A, dtype=dtype)
    W = tf.multiply(A, W)

    if flatten:
        if generated_input == "both":
            output = tf.concat(cast_all(dtype, A, W), axis=0)
        elif generated_input == "adjacency":
            output = cast_all(dtype, A)
        else:
            output = cast_all(dtype, W)
        if target != "none":
            target_value = generate_target(output, target, weight_type, flatten)
            target_value = _target_ensure_shape(output, target, target_value, n)
            return (tf.cast(output, dtype=tf.float32), tf.cast(target_value, dtype=tf.float32))
        return output
    else:
        if generated_input == "both":
            output = tf.cast(tf.stack([A, W], axis=0), dtype=tf.float32)
        elif generated_input == "adjacency":
            output = tf.cast(A, dtype=tf.float32)
        else:
            output = tf.cast(W, dtype=tf.float32)
        if target != "none":
            target_value = generate_target(output, target, weight_type, flatten)
            target_value = _target_ensure_shape(output, target, target_value, n)
            return (output, tf.cast(target_value, dtype=tf.float32))
        return output

def generate_sparse_gnp_instance(n, p, seeder, cardinality: int, target: str, generated_input: str, flatten: bool,
                                weight_distribution: tfp.distributions.Distribution, weight_type):
    """
Generates a dense G(n,p) instance with tensorflow
:param n: The number of nodes
:param p: The probability of an edge
:param seeder: The seeder to generate PRNG seeds
:param cardinality: The cardinality of the instance
:param target: The target to generate. One of "winners", "strategy", "all", "none"
:param generated_input: The input to generate. One of "adjacency", "weight", "both"
:param flatten: Whether to flatten the output
:param weight_distribution: The distribution to use for the weights
:param weight_type: The type of the weights
"""
    shape = (n, n) if not flatten else (n * n,)
    dtype = weight_distribution.dtype
    I,entries_count = tf.numpy_function(gnp_adjacency_list, inp=[n, p], Tout=[tf.int64,tf.int64], stateful=True)
    A = tf.sparse.SparseTensor(indices=I, values=tf.ones(entries_count, dtype=tf.float32), dense_shape=(n, n))
    W=tf.sparse.SparseTensor(indices=I, values=weight_distribution.sample(entries_count), dense_shape=(n, n))
    if flatten:
        A = tf.reshape(A, shape=(n * n,))
    A = tf.cast(A, dtype=dtype)
    W=tf.cast(W, dtype=dtype)

    if flatten:
        if generated_input == "both":
            output = tf.sparse.concat(0,cast_all(dtype, A, W))
        elif generated_input == "adjacency":
            output = cast_all(dtype, A)
        else:
            output = cast_all(dtype, W)
        if target != "none":
            target_value = generate_target(output, target, weight_type, flatten)
            target_value = _target_ensure_shape(output, target, target_value, n)
            return (tf.cast(output, dtype=tf.float32), tf.cast(target_value, dtype=tf.float32))
        return output
    else:
        if generated_input == "both":
            output = tf.cast(tf.sparse.concat(0,[A, W]), dtype=tf.float32)
        elif generated_input == "adjacency":
            output = tf.cast(A, dtype=tf.float32)
        else:
            output = tf.cast(W, dtype=tf.float32)
        if target != "none":
            target_value = generate_target(output, target, weight_type, flatten)
            target_value = _target_ensure_shape(output, target, target_value, n)
            return (output, tf.cast(target_value, dtype=tf.float32))
        return output


def vector_permutation(vector, permutation):
    return tf.gather(vector, permutation, axis=-1)


def matrix_permutation(matrix, permutation):
    return tf.gather(tf.gather(matrix, permutation, axis=-2), permutation, axis=-1)


def dim_mul(*b):
    if any([x is None for x in b]):
        return None
    return reduce(lambda x, y: x * y, b, 1)


def get_generator_signature(generated_input: str, n: int, target: str, flatten: bool):
    shape = None
    if flatten:
        if generated_input == "both":
            shape = (dim_mul(2, n, n),)
        else:
            shape = (dim_mul(n, n),)
        signature = (tf.TensorSpec(shape=shape, dtype=tf.float32),)
    else:
        if generated_input == "both":
            shape = (2, n, n)
        else:
            shape = (n, n)
        signature = (tf.TensorSpec(shape=shape, dtype=tf.float32), tf.TensorSpec(shape=(2,), dtype=tf.int32))
    if target != "none":
        if target == "both":
            shape = (2, 2, n)
        else:
            shape = (2, n)
        signature = (*signature, tf.TensorSpec(shape=shape))

    return signature

def get_reader_signature(generated_input: str, n: int, target: str, flatten: bool):
    shape = None
    if flatten:
        if generated_input == "both":
            shape = (dim_mul(2, n, n),)
        else:
            shape = (dim_mul(n, n),)
        signature = (tf.TensorSpec(shape=shape, dtype=tf.float32),)
    else:
        if generated_input == "both":
            shape = (2, n, n)
        else:
            shape = (n, n)
        signature = (tf.TensorSpec(shape=shape, dtype=tf.float32),)
    if target != "none":
        if target == "both":
            shape = (2, 2, n)
        else:
            shape = (2, n)
        signature = (*signature, tf.TensorSpec(shape=shape))
    if len(signature) == 1:
        return signature[0]
    return signature

def get_sparse_signature(generated_input: str, n: int, target: str, flatten: bool):
    shape = None
    if flatten:
        if generated_input == "both":
            shape = (dim_mul(2, n, n),)
        else:
            shape = (dim_mul(n, n),)
        signature = (tf.SparseTensorSpec(shape=shape, dtype=tf.float32),)
    else:
        if generated_input == "both":
            shape = (2, n, n)
        else:
            shape = (n, n)
        signature = (tf.SparseTensorSpec(shape=shape, dtype=tf.float32), tf.SparseTensorSpec(shape=(2,), dtype=tf.int32))
    if target != "none":
        if target == "both":
            shape = (2, 2, n)
        else:
            shape = (2, n)
        signature = (*signature, tf.SparseTensorSpec(shape=shape))
    return signature


def get_weight_type(weight_type: Union[str, tf.DType]):
    if weight_type == "int":
        weight_type = tf.int32
    elif weight_type == "float":
        weight_type = tf.float32
    elif weight_type == "double":
        weight_type = tf.float64
    elif not isinstance(weight_type, tf.DType):
        raise ValueError("weight_type must be a string or a tf.DType")
    return weight_type
