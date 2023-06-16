import tensorflow as tf


class MaskedSoftmax(tf.keras.layers.Softmax):

    def call(self,inputs,mask=None):
        logits,mask=inputs
        logits=tf.where(mask,logits,tf.fill(tf.shape(logits),float("-inf")))
        return super().call(logits)


# Not sure if this is necessary
class DifferentiableMaskedSoftmax(tf.keras.layers.Layer):
    def __init__(self,name=None):
        super().__init__(name=name)

    @tf.custom_gradient
    def call(self, inputs, *args, **kwargs):
        def gradient(dy):
            logits, mask = inputs
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), float("-inf")))
            probs = tf.nn.softmax(logits)
            return dy * probs, None
        logits, mask = inputs
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), float("-inf")))
        probs = tf.nn.softmax(logits)
        return probs, gradient

