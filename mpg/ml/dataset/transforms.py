import abc
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

    def mapper(self,*args):
        x,y,*z=args
        starting_vertex,starting_turn=self._get_starting_position(x)
        if self.reduce_turn:
            k=1 if starting_turn else -1
            return (k*x,starting_vertex),y[...,starting_turn,starting_vertex],*z
        else:
            return (x,starting_vertex,starting_turn),y[...,starting_turn,starting_vertex],*z





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

