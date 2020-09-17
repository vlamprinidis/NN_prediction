import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten
import argparse
from tf_data import give
import lib_tflow
from lib_tflow import distribute

import sys
sys.path.append('/home/ubuntu/profile')
import lib

parser = argparse.ArgumentParser()
parser = lib.arg_all(parser)
parser = lib.arg_conv(parser)
args = parser.parse_args()

DIM = args.dim
NAME = 'CONV{}D'.format(DIM)
RESULT = 'conv{}d.tflow'.format(DIM)
conv = layers.Conv1D if DIM==1 else layers.Conv2D

class Conv:        
    def create(self):
        model = Sequential()
        model.add( 
            conv(filters = args.filters, kernel_size = args.kernel, strides = args.stride,
                                             name = NAME, activation = 'tanh')
        )
        model.add( Flatten(name='FLATTEN') )
        model.add( Dense(units = 10, name='FINAL_DENSE') )
        model.compile(loss = lib_tflow.loss, optimizer = lib_tflow.opt, metrics=['accuracy'])
        self.model = model
        
Model = Conv()
if args.nodes > 1:
    distribute(strategy, Model, args.nodes)
else:
    Model.create()

dataset = give(DIM, args.numf, args.channels)

dataset = dataset.batch(args.batch)

if args.nodes > 1:
    dataset = strategy.experimental_distribute_dataset(dataset)
    
steps = 9*512//args.batch//args.nodes

time = lib_tflow.profile([NAME], Model.model, dataset, steps, args.epochs)

import numpy as np

data = np.array([[
    args.epochs, 9*512, # dataset size
    args.numf,
    args.channels,
    args.batch,
    args.nodes,
    args.kernel,
    args.stride,
    args.filters,
    time
]])
with open('conv{}d_tanh.tflow'.format(DIM),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
    