import abc
from collections.abc import Iterable
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class DatasetTransform(tf.Module,abc.ABC):

    def __init__(self,name=None):
        super().__init__(name=name)

    def call(self,dataset) -> tf.data.Dataset:
        return dataset.map(self.mapper)

    def transform(self,dataset) -> tf.data.Dataset:
        return self.call(dataset)

    def __call__(self,dataset) -> tf.data.Dataset:
        return self.call(dataset)

    def mapper(self,*args):
        raise NotImplementedError()


class DatasetStackTransforms(DatasetTransform):
    def __init__(self,transforms,name=None):
        super().__init__(name=name)
        self.transforms=transforms

    def call(self,dataset) -> tf.data.Dataset:
        for transform in self.transforms:
            dataset=transform(dataset)
        return dataset


class WithStartingPosition(DatasetTransform):
    def __init__(self, starting_vertex:Union[str,int],starting_turn:Union[str,int,bool], reduce_turn=True,seed=None,name=None):
        super().__init__(name=name)
        self.starting_vertex = starting_vertex
        self.starting_turn = starting_turn
        self.reduce_turn = reduce_turn
        if seed is None:
            seed = np.random.randint(0, 1 << 32)
        self.seed=seed
        self.seeder=tfp.util.SeedStream(seed, "starting_position_generator")

    def _get_starting_position(self,x):
        if self.starting_vertex == "random" or self.starting_vertex is None:
            starting_vertex = tf.random.uniform(shape=(), minval=0, maxval=tf.shape(x)[1], dtype=tf.int32, seed=self.seeder())
        else:
            starting_vertex = self.starting_vertex
        if self.starting_turn == "random" or self.starting_turn is None:
            starting_turn = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32, seed=self.seeder())
        else:
            starting_turn = self.starting_turn
        return starting_vertex,starting_turn

    def _transform_valuation(self,y):
        a=tf.constant([1,2],dtype=tf.float32)
        b=tf.constant([0,1],dtype=tf.float32)
        return a*y -b

    def mapper(self,*args):
        x,y,*z=args
        starting_vertex,starting_turn=self._get_starting_position(x)
        y=y[...,starting_turn,starting_vertex]
        y=self._transform_valuation(y)
        if self.reduce_turn:
            k=1 if starting_turn else -1
            K=tf.constant([1,k],dtype=tf.float32)
            return (K*x,starting_vertex),K*y,*z
        else:
            return (x,starting_vertex,starting_turn),y,*z





class WithRandomStartingPositions(DatasetTransform):
    def __init__(self,repeats:Union[float,int], reduce_turn=True,seed=None,name=None):
        super().__init__(name=name)
        self.repeats = repeats
        self.reduce_turn = reduce_turn
        if seed is None:
            seed = np.random.randint(0, 1 << 32)
        self.seed=seed
        self.seeder=tfp.util.SeedStream(seed, "starting_position_generator")

    def _get_repeats(self,X):
        if isinstance(self.repeats,float):
            return tf.cast(tf.math.ceil(self.repeats*tf.cast(tf.shape(X)[0],tf.float32)),tf.int32)
        return tf.fill(tf.shape(X)[0],self.repeats)

    def _get_starting_positions(self,x):
        repeats=self._get_repeats(x)
        starting_vertex = tf.random.uniform(shape=repeats, minval=0, maxval=tf.shape(x)[1], dtype=tf.int32, seed=self.seeder())
        starting_turn = tf.random.uniform(shape=repeats, minval=0, maxval=2, dtype=tf.int32, seed=self.seeder())
        return starting_vertex,starting_turn


    def call(self,dataset):
        raise NotImplementedError

class StrategyEncoder(DatasetTransform):
    def __init__(self, name=None):
        super().__init__(name=name)

    def mapper(self, *args):
        x, y, *z = args
        if isinstance(x,tuple) and len(x)>1:
            x,starting_vertex,*v=x
            size=tf.shape(x)[1]
            policy=tf.one_hot(tf.cast(y[0],dtype=tf.int32),depth=size)
            value=y[1]
            return (x,starting_vertex,*v), (value,policy), *z
        else:
            raise NotImplementedError("For now, StrategyEncoder only works with WithStartingPosition transforms")
            pass


def unrag_tensor(X):
    if isinstance(X,tf.RaggedTensor):
        return X.to_tensor()
    else:
        return X

def recursive_unrag_tensor(X):
    return tf.nest.map_structure(unrag_tensor,X)

class BatchDatasetTransform(DatasetTransform):
    def __init__(self,batch_size,pad=True,name=None):
        super().__init__(name=name)
        self.batch_size=batch_size
        self.pad=pad


    def call(self,dataset) -> tf.data.Dataset:
        return dataset.ragged_batch(self.batch_size).map(self.mapper)

    def mapper(self,*args):
        if self.pad:
            return recursive_unrag_tensor(args)
        else:
            return args


class Transposer(DatasetTransform):
    def __init__(self,name=None):
        super().__init__(name=name)


    def _transpose(self,X):
        shape=tf.shape(X)
        rank=tf.rank(X)
        base_permutation=tf.constant([1,2,0])
        permutation=tf.concat([shape[:-3],base_permutation + rank -3],axis=0)
        return tf.transpose(X,permutation)

    def mapper(self,*args):
        X,y,*z=args
        if isinstance(X,tuple):
            X,*v=X
            return (self._transpose(X),*v),y,*z
        else:
            return self._transpose(X),y,*z


class WeightNormalizer(DatasetTransform):
    def __init__(self,order="channels_last",name=None):
        super().__init__(name=name)
        self.order=order
        if order not in ["channels_first","channels_last"]:
            raise ValueError("order must be one of 'channels_first' or 'channels_last'")
        raise NotImplementedError("WeightNormalizer is not implemented yet")
        #self.weight_normalizer=tf.keras.layers.LayerNormalization(axis=-1 if order=="channels_last" else 1)

    def mapper(self,*args):
        X,*y= args
        if isinstance(X,tuple):
            X,*v=X
            return (self.weight_normalizer(X),*v),*y
        else:
            return self.weight_normalizer(X),*y


class RandomPermutation(DatasetTransform):
    def __init__(self,seed=None,name=None):
        super().__init__(name=name)
        if seed is None:
            seed = np.random.randint(0, 1 << 32)
        self.seed=seed
        self.seeder=tfp.util.SeedStream(seed, "random_permutation")

    def mapper(self,*args):
        raise NotImplementedError("RandomPermutation is not implemented yet")

class EdgeConnectionNoise(DatasetTransform):
    def __init__(self,p:float,seed=None,name=None):
        super().__init__(name=name)
        self.p=p
        if seed is None:
            seed = np.random.randint(0, 1 << 32)
        self.seed=seed
        self.seeder=tfp.util.SeedStream(seed, "edge_connection_noise")
        self.bernoulli=tfp.distributions.Bernoulli(probs=p)

    def mapper(self,*args):
        X,*y=args
        if isinstance(X,tuple):
            X,*v=X
            raise NotImplementedError("EdgeConnectionNoise is not implemented yet")
        else:
            raise NotImplementedError("EdgeConnectionNoise is not implemented yet")



class PolicyNoise(DatasetTransform):
    def __init__(self,noise:tfp.distributions = None,activation = None,seed=None,name=None):
        super().__init__(name=name)
        if noise is None:
            noise=tfp.distributions.Uniform(loc=0,scale=1)
        self.noise=noise
        self.activation=activation
        if seed is None:
            seed = np.random.randint(0, 1 << 32)
        self.seed=seed
        self.seeder=tfp.util.SeedStream(seed, "policy_noise")

    def mapper(self,*args):
        X,y,*z=args
        if isinstance(y,tuple) and len(y)>=2:
            v,pi,*t=y
            n=tf.shape(pi)[-1]
            noise=self.noise.sample(tf.shape(pi),seed=self.seeder()) / n
            pi=pi+noise
            if self.activation is not None:
                pi=self.activation(pi)
            pi=pi/tf.reduce_sum(pi,axis=-1,keepdims=True)
            return X,(v,pi),*z
        else:
            raise ValueError("PolicyNoise only supports y being a tuple of (value,policy,...)")
