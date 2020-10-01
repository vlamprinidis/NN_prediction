import numpy as np
import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

import tflow_lib
import tflow_models

import tf_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-model', type = str, required = True)
parser.add_argument('-ds', type = int, required = True)
parser.add_argument('-numf', type = int, required = True)
parser.add_argument('-ch', type = int, required = True)
parser.add_argument('-out', type = int, required = True)

parser.add_argument('-nodes', type = int, required = True)
parser.add_argument('-batch', type = int, required = True)
parser.add_argument('-epochs', type = int, required = True)
args = parser.parse_args()

print('model:',args.model,'dataset size:',args.ds,'numf:',args.numf,'channels:',ch,'out:',args.out)
        
Model = getattr(tflow_models, args.model)()
if args.nodes > 1:
    model = tflow_lib.distribute(strategy, Model, args.nodes)
else:
    model = Model.create()

ds_size = args.ds
dataset = tf_data.give(2, args.numf, args.ch, out_size=args.out, ds_size = ds_size)

dataset = dataset.batch(args.batch)

if args.nodes > 1:
    dataset = strategy.experimental_distribute_dataset(dataset)

steps = max(ds_size/args.batch/args.nodes,1)

the_time = tflow_lib.profile(model, dataset, steps, args.epochs)

import socket
host = socket.gethostname()

import numpy as np

data = np.array([[
    args.epochs, ds_size, # dataset size
    args.numf,
    args.ch,
    args.batch,
    args.nodes,
    the_time
]])
with open('{}_{}.tflow'.format(args.model, host),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
