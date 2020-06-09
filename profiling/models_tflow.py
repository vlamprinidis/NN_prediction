import sys
import os
import numpy as np

# Tensorflow

import tensorflow as  tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, Conv1D, Conv2D, Dense, Dropout, ELU, Embedding, Flatten, GaussianDropout, GaussianNoise, MaxPool1D, MaxPool2D, ReLU, Softmax

opt = tf.keras.optimizers.SGD(learning_rate=0.01)
loss = 'categorical_crossentropy'
metric = 'accuracy'

def dataset(dim=2):
    def pad(arr):
        return np.pad(arr, ((0,0),(2,2),(2,2)))

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train/255.0
    x = pad(x_train)

    rows, cols = 32, 32
    if dim==2:
        x = x.reshape(x.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)
    else: # dim==1
        x = x.reshape(x.shape[0], rows, cols)
        input_shape = (rows, cols)
    x = x.astype('float32')
    # one-hot encode the labels
    y = tf.keras.utils.to_categorical(y_train, 10)
    
    return x, y

def base(layer):    
    model = Sequential()
    model.add(layer)
    model.add(Flatten())
    model.add(Dense(units=10))
    model.compile(loss=loss, optimizer=opt,
                 metrics=[metric])
    
    return model

def conv1d(numf):
    print('\n\nThis is tflow-conv1d \n\n')
    
    model = base(Conv1D(filters = numf, 
                       kernel_size = 5))

    return model
    
def conv2d(numf):
    print('\n\nThis is tflow-conv2d \n\n')
    
    model = base(Conv2D(filters = numf, 
                       kernel_size = 5))

    return model

def avg1d(numf):
    print('\n\nThis is tflow-avg1d \n\n')
    
    model = base(AveragePooling1D(pool_size = numf))
    
    return model
    
def avg2d(numf):
    print('\n\nThis is tflow-avg2d \n\n')
    
    model = base(AveragePooling2D(pool_size = numf))
    
    return model


# def tf_
# PyTorch

# import torch

# from torch.nn import Conv1d, Conv2d, MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, ReLU, ELU, Softmax, Linear, Dropout


# import torchvision
# import torchvision.transforms as transforms
