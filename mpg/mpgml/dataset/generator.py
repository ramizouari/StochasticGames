import numpy as np
import tensorflow as tf
import mpg.graph.random_graph
import networkx as nx


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


def _generate_instances(n,p,seed, cardinality: int, target: bool, as_graph: bool,
                        adj_matrix: bool, weight_matrix: bool, as_dense: bool):
    generator=np.random.Generator(np.random.MT19937(seed))
    graph = mpg.graph.random_graph.gnp_random_mpg(n=n,p=p, seed=seed, method="fast", loops=True,
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
                output = tf.cast(_stack_sparse_tensors((2, n, n), A, W),dtype=tf.float32)

        elif adj_matrix:
            output = tf.cast(_as_tensor(nx.adjacency_matrix(graph, weight=None), as_dense=as_dense, shape_hint=(n, n)),dtype=tf.float32)
        elif weight_matrix:
            output = tf.cast(_as_tensor(nx.adjacency_matrix(graph, weight="weight"), as_dense=as_dense, shape_hint=(n, n)),dtype=tf.float32)
    starting=tf.constant([generator.integers(0,n),generator.integers(0,1,endpoint=True)])
    if target:
        # TODO: Add target
        return (output,starting,1)
    else:
        return (output,1)
def _generate_dense_instances(n,p,seed, cardinality: int, target: bool, weight_matrix: bool,flatten:bool):

    generator=np.random.Generator(np.random.MT19937(seed))
    shape=(n,n) if not flatten else (n*n,)
    W=generator.integers(-10,10,endpoint=True,size=shape)
    A=generator.integers(0,2,endpoint=True,size=shape)
    W=np.multiply(A,W)
    vertex=generator.integers(0,n)
    player=generator.integers(0,2,endpoint=True)
    if flatten:
        if weight_matrix:
            output=tf.cast(tf.concat([A,W,(vertex,player)],axis=0),dtype=tf.float32)
        else:
            output=tf.cast(tf.concat([A,(vertex,player)],axis=0),dtype=tf.float32)
        if target:
            return (output,1)
        return output
    else:
        if weight_matrix:
            output=tf.cast(tf.stack([A,W],axis=0),dtype=tf.float32)
        else:
            output=tf.cast(A,dtype=tf.float32)
        if target:
            return (output,tf.constant([vertex,player]),1)
        return (output,tf.constant([vertex,player]))
class MPGGeneratedDenseDataset(tf.data.Dataset):
    def _generator(n,p,cardinality: int, target: bool, weight_matrix: bool,flatten):
        if cardinality == tf.data.INFINITE_CARDINALITY:
            seed = 0
            while True:
                yield _generate_dense_instances(n,p,seed, cardinality, target, weight_matrix, flatten)
                seed += 1
        else:
            for sample_idx in range(cardinality):
                yield _generate_dense_instances(n,p,sample_idx, cardinality, target, weight_matrix, flatten)


    def __new__(cls,n,p, cardinality=tf.data.INFINITE_CARDINALITY,
                target: bool = False,weight_matrix: bool =True,flatten =False):
        shape = None
        if flatten:
            if weight_matrix:
                shape=(2*n*n+2,)
            else:
                shape=(n*n+2,)
            signature = (tf.TensorSpec(shape=shape, dtype=tf.float32),)
        else:
            if weight_matrix:
                shape=(2,n,n)
            else:
                shape=(n,n)
            signature = (tf.TensorSpec(shape=shape, dtype=tf.float32),tf.TensorSpec(shape=(2,), dtype=tf.int32))
        if target:
            signature = (*signature, tf.TensorSpec(shape=()))
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(n,p,cardinality, target, weight_matrix,flatten),
            output_signature=signature
        )

    def __init__(self,n,p):
        pass


class MPGGeneratedDataset(tf.data.Dataset):
    def _generator(n,p,cardinality: int, target: bool, as_graph: bool,
                   adj_matrix: bool, weight_matrix: bool, as_dense: bool):
        if cardinality == tf.data.INFINITE_CARDINALITY:
            seed = 0
            while True:
                yield _generate_instances(n,p,seed, cardinality, target, as_graph, adj_matrix, weight_matrix, as_dense)
                seed += 1
        else:
            for sample_idx in range(cardinality):
                yield _generate_instances(n,p,sample_idx, cardinality, target, as_graph, adj_matrix, weight_matrix,
                                          as_dense)

    def __new__(cls,n,p, cardinality=tf.data.INFINITE_CARDINALITY, target: bool = False, as_graph: bool = False,
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
        signature = (TensorSpec(shape=shape),tf.TensorSpec(shape=(2,), dtype=tf.int32))
        if target:
            signature = (*signature, tf.TensorSpec(shape=()))
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(n,p,cardinality, target, as_graph, adj_matrix, weight_matrix, as_dense),
            output_signature=signature
        )

    def __init__(self,n,p):
        pass
