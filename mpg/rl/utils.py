import abc

import cachetools
import tf_agents as tfa
import tensorflow as tf
from . import environment


def environment_representation(env: "environment.MPGEnvironment", representation="both"):
    """
    Extracts a representation of the environment

    """
    if representation is None:
        return env
    match representation:
        case "adjacency":
            return env.graph.adjacency_matrix
        case "weights":
            return env.graph.weights_matrix
        case "both":
            return env.graph.tensor_representation
        case "direct":
            return env
        case "graph":
            return env.graph
        case "gnn":
            raise NotImplementedError("GNN representation not implemented yet")


def encode_fully_observable_state(env: "environment.MPGEnvironment", state, representation="both"):
    return dict(state=state, environment=environment_representation(env, representation=representation))


def encode_fully_observable_state_specs(env: "environment.MPGEnvironment", env_specs):
    return dict(state=tfa.specs.ArraySpec(shape=()), environment=tfa.specs.ArraySpec(env_specs))


class ObjectWrapper(object):
    """Wrapper class that provides proxy access to some internal instance."""

    __wraps__ = None
    __ignore__ = "class mro new init setattr getattr getattribute"

    def __init__(self, obj):
        if self.__wraps__ is None:
            raise TypeError("base class Wrapper may not be instantiated")
        elif isinstance(obj, self.__wraps__):
            self._obj = obj
        else:
            raise ValueError("wrapped object must be of %s" % self.__wraps__)

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._obj, name)

    # create proxies for wrapped object's double-underscore attributes
    class __metaclass__(type):
        def __init__(cls, name, bases, dct):

            def make_proxy(name):
                def proxy(self, *args):
                    return getattr(self._obj, name)

                return proxy

            type.__init__(cls, name, bases, dct)
            if cls.__wraps__:
                ignore = set(f"__{n}__" for n in cls.__ignore__.split())
                for name in dir(cls.__wraps__):
                    if name.startswith("__"):
                        if name not in ignore and name not in dct:
                            setattr(cls, name, property(make_proxy(name)))


def normal_splitter(observation):
    if isinstance(observation["state"], tfa.specs.TensorSpec):
        return observation, tf.TensorSpec(shape=(tf.cast(observation["state"].maximum, dtype=tf.int32) + 1,))
    batch_rank = tf.rank(observation["state"])
    I=tf.stack([tf.zeros_like(observation["state"]), tf.cast(observation["state"], dtype=tf.int32)],axis=-1)
    C = tf.gather_nd(observation["environment"],indices=I,batch_dims=batch_rank)
    return observation, C


class AbstractMPGActionConstraintSplitter(tf.Module):

    def __init__(self, env: "environment.MPGEnvironment" = None, name="MPGActionConstraintSplitter"):
        self.env = env
        super().__init__(name=name)
        if self.env is not None:
            self.count_vertices = tf.constant(env.count_vertices, dtype=tf.int32)
        else:
            self.count_vertices = tf.constant(0, dtype=tf.int32)
        self.built = False

    def build(self, observation_shape):
        if self.env is None:
            self.count_vertices.assign = tf.cast(observation_shape["state"].maximum,dtype=tf.int32) + 1
        self.built = True
        constraint_shape=observation_shape["state"].shape[:-1]+ [self.count_vertices]
        return observation_shape, tf.TensorSpec(
            shape=constraint_shape)

    @abc.abstractmethod
    def call(self, observation):
        pass

    def __call__(self,observation):
        if not self.built:
            return self.build(observation)
        return self.call(observation)

class MPGActionConstraintSplitter(AbstractMPGActionConstraintSplitter):
    def call(self, observation):
        batch_rank=tf.rank(observation["state"])
        I = tf.stack([tf.zeros_like(observation["state"]), tf.cast(observation["state"], dtype=tf.int32)], axis=-1)
        C = tf.gather_nd(observation["environment"], indices=I, batch_dims=batch_rank)
        return observation, C


FullyObservableMPGActionConstraintSplitter = MPGActionConstraintSplitter

# TODO: Implement this class
class PartiallyObservableMPGActionConstraintSplitter(AbstractMPGActionConstraintSplitter):
    def __init__(self,observation_shape,env: "environment.MPGEnvironment" = None, name="MPGActionConstraintSplitter"):
        if env is None:
            raise ValueError("env must be specified in a partially observable environment")
        super().__init__(env=env,name=name)
        self.environment_tensor=tf.constant(env.graph.tensor_representation,dtype=tf.float32)
    def call(self, observation):
        batch_rank=tf.rank(observation)
        I = tf.stack([tf.zeros_like(observation), tf.cast(observation, dtype=tf.int32)], axis=-1)
        C = tf.gather_nd(self.environment_tensor, indices=I, batch_dims=batch_rank)
        return observation["state"], tf.reshape(C, shape=(1, -1))
