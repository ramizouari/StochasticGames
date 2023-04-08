import tensorflow as tf

from . import utils


def assume_starting_position(dataset, starting_vertex, starting_turn):
    """
    Assume a starting position for the dataset
    :param dataset: The data to assume the starting position for
    :param starting_vertex: The starting vertex
    :param starting_turn: The starting turn
    :return: The dataset with the starting position assumed
    """
    return dataset.map(lambda x, y: (x, y[..., starting_turn, starting_vertex]))


def graph_isomorphism(dataset, P):
    """
    Apply a graph isomorphism to the dataset
    :param dataset: The dataset to apply the isomorphism to
    :param P: The permutation matrix
    :return: The dataset with the isomorphism applied
    """
    return dataset.map(lambda x, y: (utils.matrix_permutation(x, P), utils.vector_permutation(y, P)))


def random_graph_isomorphism(dataset, seed=None):
    """
    Apply a random graph isomorphism to the dataset
    :param dataset: The dataset to apply the isomorphism to
    :return: The dataset with the isomorphism applied
    """

    if seed is None:
        seed = tf.random.uniform(shape=(), minval=0, maxval=2 ** 30, dtype=tf.int32)
    P = tf.random.shuffle(tf.range(dataset.element_spec[0].shape[-1]), seed=seed)
    return dataset.map(lambda x, y: (utils.matrix_permutation(x, P), utils.vector_permutation(y, P)))
