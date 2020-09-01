import numpy as np
import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten

import tflow_lib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-nodes', type = int, required = True)
parser.add_argument('-batch', type = int, required = True)
parser.add_argument('-epochs', type = int, required = True)
args = parser.parse_args()

opt = tf.keras.optimizers.SGD(learning_rate=0.01)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 32x32 input
class LeNet5(Sequential):        
    def __init__(self):
        super().__init__()
        self.add( 
            layers.Conv2D(filters = 6, kernel_size = 5, activation = 'tanh')
        )
        self.add(
            layers.AveragePooling2D(pool_size = 2, strides = 2)
        )
        self.add( 
            layers.Conv2D(filters = 16, kernel_size = 5, activation = 'tanh')
        )
        self.add(
            layers.AveragePooling2D(pool_size = 2, strides = 2)
        )
        self.add( Flatten() )
        self.add( Dense(units = 120, activation = 'tanh') )
        
        self.add( Flatten() )
        self.add( Dense(units = 84, activation = 'tanh') )
        
        self.add( Flatten() )
        self.add( Dense(units = 10) )
        
        self.compile(loss = loss, optimizer = opt, metrics=['accuracy'])
        
Model = LeNet5
if args.nodes > 1:
    model = tflow_lib.distribute(strategy, Model, args.nodes)
else:
    model = Model()

# dataset = tflow_lib.give2d(ds_size=1024, numf=32, channels=3, out_size=10)
(x,y),_ = tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)

x = x.astype(np.float32)
x = x/255
x = x.reshape(x.shape[0], 28, 28, 1)

# y = y.astype(np.float32)
y = tf.keras.utils.to_categorical(y, 10)

dataset = tf.data.Dataset.from_tensor_slices((x, y))

tflow_lib.train(model, dataset, args.batch, args.epochs)