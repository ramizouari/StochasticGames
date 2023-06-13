import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import mpg.mpgio
from mpg.ml.dataset import utils


def read_mpg(file:str,target:str,target_data,generated_input: str, flatten:bool,n,weight_type,dense:bool=False):
    """
    Reads a dataset from a file
    :param file: The path to the file
    :param target: The target to generate. One of "winners", "strategy", "all", "none"
    :param target_data: The target data. If None, the target will be calculated from the input
    :param generated_input: The input to generate. One of "adjacency", "weight", "both"
    :param flatten: Whether to flatten the input
    :param n: The number of nodes
    :param weight_type: The type of the weights
    :param dense: Whether to return dense matrices
    :return: The dataset
    """
    game=mpg.mpgio.read_mpg(file,edgetype=float)
    I=[]
    W=[]
    A=[]
    for u,v in game.edges:
        I.append([u,v])
        W.append(game.edges[u,v])
        A.append(1)
    I=tf.constant(I)
    W=tf.sparse.SparseTensor(I,tf.constant(W),[n,n])
    A=tf.sparse.SparseTensor(I,tf.constant(A),[n,n])
    if dense:
        W=tf.sparse.to_dense(W)
        A=tf.sparse.to_dense(A)
    dtype=weight_type
    if flatten:
        if generated_input == "both":
            output = tf.concat(utils.cast_all(dtype, A, W), axis=0)
        elif generated_input == "adjacency":
            output = utils.cast_all(dtype, A)
        else:
            output = utils.cast_all(dtype, W)
        if target != "none":
            if target_data is None:
                target_value = utils.generate_target(output, target, weight_type, flatten)
            else:
                target_value = target_data[file]
            target_value = utils._target_ensure_shape(output, target, target_value, n)
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
            target_value = utils.generate_target(output, target, weight_type, flatten)
            target_value = utils._target_ensure_shape(output, target, target_value, n)
            return (output, tf.cast(target_value, dtype=tf.float32))
        return output

class MPGGraphReader(tf.data.Dataset):
    def __new__(cls, path:str,target:str=None, target_path:str=None,generated_input: str = "both", flatten=False,n=None,weight_type: str = "int"):
        """
        Reads a dataset from a file
        :param path: The path to the file
        :param target: The target to generate. One of "winners", "strategy", "all", "none"
        """
        if target is None or target == False:
            target = "none"
        elif target == True:
            target = "both"
        if target not in ("winners", "strategy", "all", "none"):
            raise ValueError("target must be one of 'winners', 'strategy', 'all', 'none'")
        if generated_input not in ("adjacency", "weight", "both"):
            raise ValueError("generated_input must be one of 'adjacency', 'weight', 'both'")
        options=tf.data.Options()
        if target_path is not None:
            target_data=pd.read_csv(target_path)
        options.experimental_deterministic=False
        signature=utils.get_signature(generated_input=generated_input, flatten=flatten, target=target, n=n)
        return tf.data.Dataset.list_files(path).map(lambda path: read_mpg(path, target,target_path, generated_input, flatten, n, weight_type),num_parallel_calls=12)\
            .with_options(options)