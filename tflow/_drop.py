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
parser.add_argument('-drop', type = float, required = True)
args = parser.parse_args()

DIM = args.dim
RESULT = '__drop{}d.tflow'.format(DIM)
NAME = 'DROP{}D'.format(DIM)

class Drop:
    def create(self):
        model = Sequential()
        model.add( 
            layers.Dropout(rate = args.drop, name = NAME)
        )
        model.add( Flatten(name='FLATTEN') )
        model.add( Dense(units = 10, name='FINAL_DENSE') )
        model.compile(loss = lib_tflow.loss, optimizer = lib_tflow.opt, metrics=['accuracy'])
        self.model = model

Model = Drop()
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
    args.drop,
    time
]])
with open('drop{}d.tflow'.format(DIM),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
    