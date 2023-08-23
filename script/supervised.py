import abc
import sys

import tensorflow as tf
import mpg.ml.model.gnn as gnn
import mpg.ml.layers.augmentation as aug
import mpg.ml.dataset.reader as reader
import mpg.ml.dataset.generator as generator
import mpg.ml.dataset.utils as utils
import mpg.ml.dataset.transforms as transforms
import argparse


class LossTransformation(tf.keras.losses.Loss):
    def __init__(self, loss, name=None):
        super(LossTransformation, self).__init__(name=name)
        self.loss = loss

    @abc.abstractmethod
    def call(self, y_true, y_pred):
        pass


class Tanh2Sigmoid(LossTransformation):
    def __init__(self, name=None):
        super(Tanh2Sigmoid, self).__init__(tf.keras.losses.BinaryCrossentropy(name=name), name=name)

    def call(self, y_true, y_pred):
        return self.loss((y_true + 1) / 2, (tf.sigmoid(y_pred) + 1) / 2)


def main(argv=()):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,required=True,help="Path to dataset")
    parser.add_argument("--target", type=str,required=True,help="Path to dataset annotations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--random_connection_probability", type=float, default=0.05)
    parser.add_argument("--weight_noise", type=float, default=0.01)
    parser.add_argument("--weight_normalisation", type=str, required=False)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--total_batches", type=int, default=100)
    args = parser.parse_args(argv)
    target_path = args.target
    dataset_path = args.dataset
    # target_dataset="/run/media/ramizouari/6cf4ec5b-ccaf-4b2a-a40f-75ddfe64af76/HPC/dataset/targets_dense.json"
    gen = reader.MPGGraphReader(dataset_path,
                                target="all", target_path=target_path, debug=False)
    model = gnn.ResGNN(conv_layers=[32, 16, 1, 32, 16, 1], residual_connections={3: [0], 6: [3, 0]},
                       graph_normalisation=True,
                       masked_softmax=True, ragged_batch=False, weight_normalisation=args.weight_normalisation,
                       random_connection_probability=0.05,
                       weight_noise_layer=aug.UniformNoise(-args.weight_noise,args.weight_noise, name="uniform_noise"))
    sub_model = tf.keras.Model(inputs=model.inputs, outputs=(model.outputs[0] + 1) / 2)

    sub_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss="binary_crossentropy",
                      metrics=["accuracy"])
    sub_model.summary()
    T = transforms.DatasetStackTransforms(
        [transforms.Transposer(), transforms.WithStartingPosition(0,0),
         transforms.BatchDatasetTransform(pad=True, batch_size=args.batch_size)])
    dataset = T(gen.take(16).cache()).map(lambda X, Y: (X, (Y[0] + 1) / 2)).repeat()
    return sub_model, dataset

if __name__ == "__main__":
    main(sys.argv[1:])