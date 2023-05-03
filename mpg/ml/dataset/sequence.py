import numpy as np
import tensorflow as tf
from tensorflow import keras
import mpg.games.mpg as mpg

def _generate_dense_batch(batch_size,n,p,seed, cardinality: int, target: bool, weight_matrix: bool,flatten:bool):

    generator=np.random.Generator(np.random.MT19937(seed))
    shape=(batch_size,n,n) if not flatten else (batch_size,n*n)
    W=generator.integers(-10,10,endpoint=True,size=shape)
    A=generator.binomial(1,p,size=shape)
    W=np.multiply(A,W)
    vertex=generator.integers(0,n,size=(batch_size,1))
    player=generator.integers(0,1,endpoint=True,size=(batch_size,1))
    if flatten:
        if weight_matrix:
            output=tf.cast(tf.concat([A,W,vertex,player],axis=1),dtype=tf.float32)
        else:
            output=tf.cast(tf.concat([A,vertex,player],axis=1),dtype=tf.float32)
        if target:
            return (output,np.ones(shape=(batch_size,)))
        return output
    else:
        if weight_matrix:
            output=tf.cast(tf.stack([A,W],axis=1),dtype=tf.float32)
        else:
            output=tf.cast(A,dtype=tf.float32)
        if target:
            return (output,tf.constant([vertex,player]),np.ones(shape=(batch_size,)))
        return (output,tf.constant([vertex,player]))
class MPGGeneratedDenseDataset(keras.utils.Sequence):
    def __init__(self, n, p,batch_size, batches=64,
                 target: bool = False, weight_matrix: bool = True, flatten=False,seed=0):
        self.n = n
        self.p = p
        self.batches = batches
        self.target = target
        self.weight_matrix = weight_matrix
        self.flatten = flatten
        self.batch_size = batch_size
        self.generator=np.random.Generator(np.random.MT19937(seed))
        self.seed=seed
    def __getitem__(self, item):
        return _generate_dense_batch(self.batch_size,self.n,self.p,self.seed+item, self.batches, self.target, self.weight_matrix, self.flatten)

    def __len__(self):
        return self.batches

    def on_epoch_end(self):
        self.seed=int.from_bytes(self.generator.bytes(4),byteorder='big')
        pass


