import numba
import numpy as np
import tensorflow as tf
import mpg.graph.random_graph
import networkx as nx
import itertools
import tensorflow_probability as tfp
import mpg.wrapper as mpgwrapper
from . import utils


class MPGGeneratedDenseDataset(tf.data.Dataset):
    """
    A dataset that generates dense MPG graphs following a G(n,p) distribution.
    """

    def __new__(cls, n, p, cardinality=tf.data.INFINITE_CARDINALITY,
                target: str = None, generated_input: str = "both", flatten=False, seed=None,
                weights_distribution: tfp.distributions.Distribution = None,
                weight_type: str = "int", num_parallel_calls=tf.data.experimental.AUTOTUNE):
        """
Generates a dense G(n,p) dataset.
:param n: The number of nodes
:param p: The probability of an edge
:param cardinality: The cardinality of the dataset. If it is equal to tf.data.INFINITE_CARDINALITY, the dataset will be infinite
:param target: The target to generate. One of "winners", "strategy", "all", "none"
:param weight_matrix: Whether to generate a weight matrix
:param flatten: Whether to flatten the input
:param seed: The seed to use for the PRNG
:param weights_distribution: The distribution to use for the weights
:param weight_type: The type of the weights
        """
        if generated_input not in ("adjacency", "weight", "both"):
            raise ValueError("generated_input must be one of 'adjacency', 'weight', 'both'")
        if target is None or target == False:
            target = "none"
        elif target == True:
            target = "all"
        weight_type= utils.get_weight_type(weight_type)

        if weights_distribution is None:
            weights_distribution = tfp.distributions.Uniform(low=-1, high=1)
        if seed is None:
            seed = np.random.randint(0, 1 << 32)

        seeder = tfp.util.SeedStream(seed, "seeding_generator")

        signature=utils.get_signature(generated_input=generated_input, flatten=flatten, target=target, n=n)

        generated: tf.data.Dataset
        options = tf.data.Options()
        options.experimental_deterministic = False
        if cardinality == tf.data.INFINITE_CARDINALITY:
            generated = tf.data.Dataset.counter(start=seed, step=1).with_options(options)
        else:
            generated = tf.data.Dataset.range(seed, seed + cardinality).with_options(options)
        return generated.map(
            lambda seed: utils.generate_dense_gnp_instance(n, p, seeder, cardinality, target, generated_input, flatten,
                                                           weights_distribution, weight_type),
            num_parallel_calls=num_parallel_calls
        ).with_options(options)
        #    range,
        #    args=(n,p,cardinality, target, weight_matrix,flatten),
        #    output_signature=signature
        # )

    def __init__(self, n, p, cardinality=tf.data.INFINITE_CARDINALITY,
                 target: bool = False, generated_input: str = "both", flatten=False, seed=None,
                 weights_distribution: tfp.distributions.Distribution = None,
                 weight_type: str = "int"):

        self.n = n
        self.p = p
        self.cardinality = cardinality
        self.target = target
        self.generated_input = generated_input
        self.flatten = flatten
        self.seed = seed
        self.weights_distribution = weights_distribution

    def _permutation(self, x, P):
        if self.flatten:
            S = tf.concat(tf.reshape(tf.tensordot(P, P, axes=None), shape=(-1,)))
            return tf.concat([tf.gather(x, S, axis=0), P[x[-2]], x[-1]])

    def permutation(self, P):
        return self.map(lambda x: self._permutation(x, P))

    def with_starting_position(self, starting_position=None):
        if starting_position == "all":
            self.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        if starting_position == "random" or not starting_position:
            starting_position = tf.random.uniform(shape=(self.n,), minval=0, maxval=2, dtype=tf.int32)
        return self.map(lambda x: tf.concat([x, starting_position]))




class MPGSparseGeneratedDataset(tf.data.Dataset):
    def __new__(cls, n, p, cardinality=tf.data.INFINITE_CARDINALITY,
                target: str = None, generated_input: str = "both", flatten=False, seed=None,
                weights_distribution: tfp.distributions.Distribution = None,
                weight_type: str = "int"):
        """
Generates a potentially sparse G(n,p) dataset.
:param n: The number of nodes
:param p: The probability of an edge
:param cardinality: The cardinality of the dataset. If it is equal to tf.data.INFINITE_CARDINALITY, the dataset will be infinite
:param target: The target to generate. One of "winners", "strategy", "all", "none"
:param weight_matrix: Whether to generate a weight matrix
:param flatten: Whether to flatten the input
:param seed: The seed to use for the PRNG
:param weights_distribution: The distribution to use for the weights
:param weight_type: The type of the weights
        """
        if generated_input not in ("adjacency", "weight", "both"):
            raise ValueError("generated_input must be one of 'adjacency', 'weight', 'both'")
        if target is None or target == False:
            target = "none"
        elif target == True:
            target = "all"
        weight_type = utils.get_weight_type(weight_type)

        if weights_distribution is None:
            weights_distribution = tfp.distributions.Uniform(low=-1, high=1)
        if seed is None:
            seed = np.random.randint(0, 1 << 32)

        seeder = tfp.util.SeedStream(seed, "seeding_generator")

        signature = utils.get_sparse_signature(generated_input=generated_input, flatten=flatten, target=target, n=n)

        generated: tf.data.Dataset
        options = tf.data.Options()
        options.experimental_deterministic = False
        if cardinality == tf.data.INFINITE_CARDINALITY:
            generated = tf.data.Dataset.counter(start=seed, step=1).with_options(options)
        else:
            generated = tf.data.Dataset.range(seed, seed + cardinality).with_options(options)
        return generated.map(
            lambda seed: utils.generate_sparse_gnp_instance(n, p, seeder, cardinality, target, generated_input, flatten,
                                                           weights_distribution, weight_type),
            num_parallel_calls=12
        ).with_options(options)
        #    range,
        #    args=(n,p,cardinality, target, weight_matrix,flatten),
        #    output_signature=signature
        # )

    def __init__(self, n, p, cardinality=tf.data.INFINITE_CARDINALITY,
                 target: bool = False, generated_input: str = "both", flatten=False, seed=None,
                 weights_distribution: tfp.distributions.Distribution = None,
                 weight_type: str = "int"):

        self.n = n
        self.p = p
        self.cardinality = cardinality
        self.target = target
        self.generated_input = generated_input
        self.flatten = flatten
        self.seed = seed
        self.weights_distribution = weights_distribution

    def _permutation(self, x, P):
        if self.flatten:
            S = tf.concat(tf.reshape(tf.tensordot(P, P, axes=None), shape=(-1,)))
            return tf.concat([tf.gather(x, S, axis=0), P[x[-2]], x[-1]])

    def permutation(self, P):
        return self.map(lambda x: self._permutation(x, P))

    def with_starting_position(self, starting_position=None):
        if starting_position == "all":
            self.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        if starting_position == "random" or not starting_position:
            starting_position = tf.random.uniform(shape=(self.n,), minval=0, maxval=2, dtype=tf.int32)
        return self.map(lambda x: tf.concat([x, starting_position]))

