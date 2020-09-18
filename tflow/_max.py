import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten
import argparse
from tf_data import give, ds_size
import lib_tflow
from lib_tflow import distribute

import sys
sys.path.append('/home/ubuntu/profile')
import lib

parser = argparse.ArgumentParser()
parser = lib.arg_all(parser)
parser = lib.arg_pool(parser)
args = parser.parse_args()

DIM = args.dim
RESULT = '__max{}d.tflow'.format(DIM)
NAME = 'MAX{}D'.format(DIM)
max_pool = layers.MaxPool1D if DIM==1 else layers.MaxPool2D

class Max:
    def create(self):
        model = Sequential()
        model.add( 
            max_pool(pool_size = args.pool, strides=args.stride, name = NAME)
        )
        model.add( Flatten(name='FLATTEN') )
        model.add( Dense(units = 10, name='FINAL_DENSE') )
        model.compile(loss = lib_tflow.loss, optimizer = lib_tflow.opt, metrics=['accuracy'])
        self.model = model

Model = Max()
if args.nodes > 1:
    distribute(strategy, Model, args.nodes)
else:
    Model.create()

dataset = give(DIM, args.numf, args.channels)

dataset = dataset.batch(args.batch)

if args.nodes > 1:
    dataset = strategy.experimental_distribute_dataset(dataset)
    
steps = ds_size//args.batch//args.nodes

the_typs = ['MaxPool']

time = lib_tflow.profile(the_typs, None, Model.model, dataset, steps, args.epochs)

import numpy as np

data = np.array([[
    args.epochs, ds_size, # dataset size
    args.numf,
    args.channels,
    args.batch,
    args.nodes,
    args.pool,
    args.stride,
    time
]])
with open('max{}d.tflow'.format(DIM),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")