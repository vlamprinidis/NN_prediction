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
args = parser.parse_args()

DIM = args.dim

class Norm:
    def create(self):
        model = Sequential()
        model.add( 
            layers.BatchNormalization()
        )
        model.add( Flatten(name='FLATTEN') )
        model.add( Dense(units = 10, name='FINAL_DENSE') )
        model.compile(loss = lib_tflow.loss, optimizer = lib_tflow.opt, metrics=['accuracy'])
        self.model = model

Model = Norm()
if args.nodes > 1:
    distribute(strategy, Model, args.nodes)
else:
    Model.create()

dataset = give(DIM, args.numf, args.channels)

dataset = dataset.batch(args.batch)

if args.nodes > 1:
    dataset = strategy.experimental_distribute_dataset(dataset)
    
steps = ds_size//args.batch//args.nodes

the_typs, the_ops = (['SquaredDifference', 'Mean'], None) if DIM == 1 else (
    ['FusedBatchNormV3', 'FusedBatchNormGradV3'], None)

time = lib_tflow.profile(the_typs, the_ops, Model.model, dataset, steps, args.epochs)

import numpy as np

data = np.array([[
    args.epochs, ds_size, # dataset size
    args.numf,
    args.channels,
    args.batch,
    args.nodes,
    time
]])
with open('norm{}d.tflow'.format(DIM),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
    