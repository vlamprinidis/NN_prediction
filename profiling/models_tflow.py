import sys
import os
import numpy as np
from numpy.random import RandomState as R

# Tensorflow

import tensorflow as  tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, Conv1D, Conv2D, Dense, Dropout, ELU, Embedding, Flatten, GaussianDropout, GaussianNoise, MaxPool1D, MaxPool2D, ReLU, Softmax

opt = tf.keras.optimizers.SGD(learning_rate=0.01)
loss = 'categorical_crossentropy'
metric = 'accuracy'

# def dataset(dim=2):
#     def pad(arr):
#         return np.pad(arr, ((0,0),(2,2),(2,2)))

#     mnist = tf.keras.datasets.mnist
#     (x_train, y_train), _ = mnist.load_data()
#     x_train = x_train/255.0
#     x = pad(x_train)

#     rows, cols = 32, 32
#     if dim==2:
#         x = x.reshape(x.shape[0], rows, cols, 1)
#         input_shape = (rows, cols, 1)
#     else: # dim==1
#         x = x.reshape(x.shape[0], rows, cols)
#         input_shape = (rows, cols)
#     x = x.astype('float32')
#     # one-hot encode the labels
#     y = tf.keras.utils.to_categorical(y_train, 10)
    
#     return x, y

def dummy(dim, n):
    ds_size = 5000
    out_size = 10
    if dim == 1:
        x = R(42).random((ds_size, n))
        x = x.reshape(x.shape[0], n, 1)
    else:
        x = R(42).random((ds_size, n, n))
        x = x.reshape(x.shape[0], n, n, 1)
    
    y = R(42).randint(0,out_size,ds_size)
    y = tf.keras.utils.to_categorical(y, out_size)
    
    return x,y
    
def base(layer):    
    model = Sequential()
    model.add( layer )
    model.add( Flatten() )
    model.add( Dense(units = 10) )
    model.compile(loss=loss, optimizer=opt,
                 metrics=[metric])
    
    return model

class conv1d:
    def __init__(self, n):
        self.x, self.y = dummy(1,n)
    
    def create(self):
        print('\n\nThis is tflow-conv1d \n\n')
        self.model = base(Conv1D(filters = 6, kernel_size = 5))

class conv2d:
    def __init__(self, n):
        self.x, self.y = dummy(2,n)
        
    def create(self):
        print('\n\nThis is tflow-conv2d \n\n')
        self.model = base(Conv2D(filters = 6, kernel_size = 5))

class avg1d:
    def __init__(self, n):
        self.x, self.y = dummy(1,n)
    
    def create(self):
        print('\n\nThis is tflow-avg1d \n\n')
        self.model = base(AveragePooling1D(pool_size = 2))
    
    
class avg2d:
    def __init__(self, n):
        self.x, self.y = dummy(2,n)
        
    def create(self):
        print('\n\nThis is tflow-avg2d \n\n')
        self.model = base(AveragePooling2D(pool_size = 2))

class max1d:
    def __init__(self, n):
        self.x, self.y = dummy(1,n)
        
    def create(self):
        print('\n\nThis is tflow-max1d \n\n')
        self.model = base( MaxPool1D(pool_size = 2) )
    
class max2d:
    def __init__(self, n):
        self.x, self.y = dummy(2,n)
        
    def create(self):
        print('\n\nThis is tflow-max2d \n\n')
        self.model = base( MaxPool2D(pool_size = 2) )

class dense:
    def __init__(self, n):
        self.x, self.y = dummy(1,n)
    
    def create(self):
        print('\n\nThis is tflow-dense \n\n')
        model = Sequential()
        model.add( Dense(units = 10) )
        model.compile(loss=loss, optimizer=opt, metrics=[metric])
        self.model = model

mapp = {
    'avg1d': avg1d,
    'avg2d': avg2d,
    'conv1d': conv1d,
    'conv2d': conv2d,
    'max1d': max1d,
    'max2d': max2d,
    'dense': dense
}