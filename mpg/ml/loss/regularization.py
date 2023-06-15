import tensorflow as tf

class L2Loss(tf.Module):
    def __init__(self, weights, alpha:float=1, name=None):
        super().__init__(name=name)
        self.weights = weights
        self.alpha=alpha

    @tf.function
    def __call__(self):
        return self.alpha*tf.add_n([tf.nn.l2_loss(v) for v in self.weights])
def l2_loss(model,alpha:float=1):
    @tf.function
    def l2_loss_implementation():
        return alpha*tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
    return l2_loss_implementation

class L2LossHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, weights,regularization=0.01):
        super().__init__()
        self.weights = weights
        self.l2_loss=L2Loss(self.weights,alpha=regularization, name="l2_loss_callback")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["l2_loss"] = float(self.l2_loss())


