import tensorflow as tf
from numpy.random import RandomState as R

seed = 42
ds_size = 9*512

def give(dim, n, channels, out_size=10, ds_size = ds_size):
    if dim == 1:
        x = R(seed).random((ds_size, n, channels))
        x = x.reshape(x.shape[0], n, channels)
    else:
        x = R(seed).random((ds_size, n, n, channels))
        x = x.reshape(x.shape[0], n, n, channels)
    
    y = R(seed).randint(0,out_size,ds_size)
    y = tf.keras.utils.to_categorical(y, out_size)
    
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    return dataset
