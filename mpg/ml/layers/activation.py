import tensorflow as tf


class MaskedSoftmax(tf.keras.layers.Softmax):

    def __init__(self, axis=-1, **kwargs):
        super().__init__(axis=axis, **kwargs)
    def call(self,inputs,mask=None):
        logits,mask=inputs
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), float("-inf")))
        return super().call(logits)


# Not sure if this is necessary
class DifferentiableMaskedSoftmax(tf.keras.layers.Layer):
    def __init__(self,neg_inf=-1e6,name=None):
        super().__init__(name=name)
        self.neg_inf=neg_inf

    @tf.custom_gradient
    def call(self, inputs, *args, **kwargs):
        def gradient(dy):
            logits, mask = inputs
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), self.neg_inf))
            probs = tf.nn.softmax(logits)
            return dy * probs, None
        logits, mask = inputs
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), self.neg_inf))
        probs = tf.nn.softmax(logits)
        return probs, gradient

