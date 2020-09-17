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

parser.add_argument('-numf', type = int, required = True )
parser.add_argument('-batch', type = int, required = True, 
                           choices = lib.batch_ls )
parser.add_argument('-nodes', type = int, required = True,
                           choices = lib.nodes_ls )
parser.add_argument('-epochs', type = int, required = True)

parser.add_argument('-units', type = int, required = True)
args = parser.parse_args()

NAME = 'DENSE'

class MyDense:
    def create(self):
        model = Sequential()
        model.add( 
            Dense(units = args.units, name = NAME, activation = 'tanh')
        )
        model.add( Flatten(name='FLATTEN') )
        model.add( Dense(units = 10, name='FINAL_DENSE') )
        model.compile(loss = lib_tflow.loss, optimizer = lib_tflow.opt, metrics=['accuracy'])
        self.model = model

Model = MyDense()
if args.nodes > 1:
    distribute(strategy, Model, args.nodes)
else:
    Model.create()
    
dataset = give(1, args.numf, 1)

dataset = dataset.batch(args.batch)

if args.nodes > 1:
    dataset = strategy.experimental_distribute_dataset(dataset)
    
steps = 9*512//args.batch//args.nodes

time = lib_tflow.profile([NAME], Model.model, dataset, steps, args.epochs)

import numpy as np

data = np.array([[
    args.numf,
    args.batch,
    args.nodes,
    args.units,
    time
]])
with open('dense.tflow','a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
