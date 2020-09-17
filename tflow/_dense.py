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
parser.add_argument('-units', type = int, required = True)
args = parser.parse_args()

DIM = args.dim
RESULT = '__dense{}d.tflow'.format(DIM)
NAME = 'DENSE{}D'.format(DIM)

class MyDense:
    def create(self):
        model = Sequential()
        model.add( 
            Dense(units = args.units, name = NAME)
        )
        model.add( Flatten(name='FLATTEN', activation = 'tanh') )
        model.add( Dense(units = 10, name='FINAL_DENSE') )
        model.compile(loss = lib_tflow.loss, optimizer = lib_tflow.opt, metrics=['accuracy'])
        self.model = model

Model = MyDense()
if args.nodes > 1:
    distribute(strategy, Model, args.nodes)
else:
    Model.create()
    
dataset = give(DIM, args.numf, args.channels)

time = lib_tflow.profile([NAME], Model.model, dataset, args.batch, args.epochs)

import numpy as np

data = np.array([[
    args.numf,
    args.channels,
    args.batch,
    args.nodes,
    args.units,
    time
]])
with open('dense{}d.tflow'.format(DIM),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
