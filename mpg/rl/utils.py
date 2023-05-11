import cachetools
import tf_agents as tfa
from . import environment


def environment_representation(env: environment.MPGEnvironment, representation="both"):
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


def encode_fully_observable_state(env: environment.MPGEnvironment, state, representation="both"):
    return dict(state=state, environment=environment_representation(env,representation=representation))


def encode_fully_observable_state_specs(env: environment.MPGEnvironment, env_specs):
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