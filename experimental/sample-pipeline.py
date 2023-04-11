import mpg.mpgml.dataset.generator
from mpg.mpgml.dataset.generator import MPGGeneratedDenseDataset
import tensorflow as tf
import tensorflow_probability as tfp
import mpg.mpgml.dataset.utils as utils
import mpg.mpgml.dataset.augmentation as ds_aug
import mpg.mpgml.layers.padding as mpgml_pad
import mpg.mpgml.layers.normalisation as mpgml_norm
import mpg.mpgml.layers.augmentation as mpgml_augm
from tensorflow import keras


def assume_starting_position(dataset, starting_vertex, starting_turn):
    return dataset.map(lambda x, y: (x, y[..., starting_turn, starting_vertex]))


if __name__ == "__main__":
    n = 10
    dataset = MPGGeneratedDenseDataset(n, 0.5, seed=25, target="winners", generated_input="both", flatten=False,
                                       weights_distribution=tfp.distributions.Uniform(-100, 100), weight_type="int")
    validation_dataset = MPGGeneratedDenseDataset(8, 0.5, target="winners", generated_input="both", flatten=False,
                                                  weights_distribution=tfp.distributions.Uniform(-1, 1),
                                                  weight_type="float")

    transformed = assume_starting_position(ds_aug.random_graph_isomorphism(dataset.batch(64).take(1024).cache(), 10), 0,
                                           0).repeat()
    validation_dataset = assume_starting_position(validation_dataset, 0, 0).batch(64).take(12).cache()
    model = keras.Sequential([
        mpgml_pad.GraphPaddingLayer(n),
        #mpgml_augm.GraphIsomorphismLayer(n),
        mpgml_norm.MPGNormalisationLayer(edges_matrix=1),
        mpgml_augm.EdgeWeightsNoiseLayer(noise_layer=keras.layers.GaussianNoise(stddev=0.01), edges_matrix=1),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(15, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(15, activation="relu"),
        keras.layers.Dense(1, "sigmoid")
    ])

    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(x=transformed,
              epochs=250,
              verbose='auto',
              steps_per_epoch=64,
              shuffle=False,
              validation_data=validation_dataset
              )
