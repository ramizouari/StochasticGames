import os

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import mpg.mpgio
from mpg.ml.dataset import utils


def _read_mpg_to_tensor_sparse(file):
    game=mpg.mpgio.read_mpg(file,edgetype=float)
    I=[]
    W=[]
    A=[]
    n=game.number_of_nodes()
    for u,v in game.edges:
        I.append([u,v])
        W.append(game.edges[u,v]["weight"])
        A.append(1)
    return I,A,W,n
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
    I,A,W,N=_read_mpg_to_tensor_sparse(file)
    I=tf.cast(tf.ensure_shape(I,[None,2]),dtype=tf.int64)
    A=tf.ensure_shape(A,[None])
    W=tf.ensure_shape(W,[None])
    W=tf.sparse.SparseTensor(I,tf.cast(tf.constant(W),dtype=tf.float32),[N,N])
    A=tf.sparse.SparseTensor(I,tf.cast(tf.constant(A),dtype=tf.float32),[N,N])
    W=tf.sparse.reorder(W)
    A=tf.sparse.reorder(A)
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


class FileReaderIterator:
    def __init__(self,path:str,target:str=None, target_path:str=None,generated_input: str = "both", flatten=False,n=None,weight_type: str = "int",
                 filter:callable=None):
        self.path=path
        if target is None or target == False:
            target = "none"
        elif target == True:
            target = "both"
        if target not in ("winners", "strategy", "all", "none"):
            raise ValueError("target must be one of 'winners', 'strategy', 'all', 'none'")
        if generated_input not in ("adjacency", "weight", "both"):
            raise ValueError("generated_input must be one of 'adjacency', 'weight', 'both'")
        if target_path is not None:
            target_data = pd.read_csv(target_path)
        else:
            target_data=None
        self.target_data=target_data
        self.target=target
        self.generated_input=generated_input
        self.flatten=flatten
        self.n=n
        self.weight_type=weight_type
        if type(path) is str:
            self.path_iter=iter(os.listdir(path))
        else:
            self.path_iter=iter(path)
        self.filter=filter

    def __iter__(self):
        return self

    def __next__(self):
        file = next(self.path_iter)
        if self.filter is not None:
            while not self.filter(file):
                file = next(self.path_iter)
        return read_mpg(os.path.join(self.path,file), self.target, self.target_data, self.generated_input, self.flatten, self.n,
                        self.weight_type, dense=True)

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
        signature=utils.get_reader_signature(generated_input=generated_input, flatten=flatten, target=target, n=n)
        file_reader=FileReaderIterator(path,target,target_path,generated_input,flatten,n,weight_type)
        print(signature)
        def generator():
            for input in file_reader:
                yield input
        return tf.data.Dataset.list_files(path).from_generator(generator,output_signature=signature).with_options(options)