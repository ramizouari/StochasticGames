from collections import ChainMap
import tensorflow as tf

WEIGHT_MATRIX_AXIS = 1
ADJACENCY_MATRIX_AXIS = 0


def build_layer_from_args(layer_class, layer_args, **additional_kwargs):
    if type(layer_args) is dict:
        # ChainMap is used to combine the layer_args and additional_kwargs dicts
        # into a single dict. The layer_args dict takes precedence over the additional_kwargs dict.
        kwargs = ChainMap(layer_args, additional_kwargs)
        return layer_class(**kwargs)
    elif type(layer_args) is list:
        return layer_class(*layer_args, **additional_kwargs)
    else:
        return layer_class(layer_args, **additional_kwargs)


class UnragTensor(tf.keras.layers.Layer):
    def __init__(self, axis=0, name=None):
        super().__init__(name=name)
        self.axis = axis

    def call(self, inputs, **kwargs):
        if isinstance(inputs,tf.RaggedTensor):
            return inputs.to_tensor()
        else:
            return inputs

class RagTensor(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, inputs, **kwargs):
        if isinstance(inputs,tf.RaggedTensor):
            return inputs
        else:
            return tf.RaggedTensor.from_tensor(inputs)


class RagPolicy(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, inputs, **kwargs):
        environment,policy_dense, row_lengths = inputs
        if isinstance(environment,tf.RaggedTensor):
            return tf.RaggedTensor.from_tensor(policy_dense, row_lengths,ragged_rank=1)
        else:
            return policy_dense
