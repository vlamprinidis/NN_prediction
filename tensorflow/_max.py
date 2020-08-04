import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten
import argparse
from tf_data import give
import funs_tflow
from funs_tflow import distribute
import funs

parser = argparse.ArgumentParser()
parser.add_argument('-numf', type = int, required = True,
                           choices = funs.numf_ls )
parser.add_argument('-batch', type = int, required = True, 
                           choices = funs.batch_ls )
parser.add_argument('-nodes', type = int, required = True,
                           choices = funs.nodes_ls )
parser.add_argument('-epochs', type = int, required = True)
parser.add_argument('-channels', type = int, required = True)
parser.add_argument('-pool', type = int, required = True)
parser.add_argument('-stride', type = int, required = True)
parser.add_argument('-dim', type = int, required = True)
args = parser.parse_args()

DIM = args.dim
RESULT = '__max{}d.tflow'.format(DIM)
NAME = 'MAX{}D'.format(DIM)
max_pool = layers.MaxPool1D if DIM==1 else layers.MaxPool2D

model = Sequential()
model.add( 
    max_pool(pool_size = args.pool, strides=args.stride, name = NAME)
)
model.add( Flatten(name='FLATTEN') )
model.add( Dense(units = 10, name='FINAL_DENSE') )

if args.nodes > 1:
    model = distribute(strategy, model, args.nodes)
else:
    model.compile(loss = funs_tflow.loss, optimizer = funs_tflow.opt, metrics=['accuracy'])

x,y = give(DIM, args.numf, args.channels)

prof = funs_tflow.profile(model, x, y, args.batch, args.epochs)

if prof != None:
    key = funs.my_key({
            'numf':args.numf,
            'batch':args.batch,
            'nodes':args.nodes,
            'channels':args.channels,
            'stride':args.stride,
            'pool':args.pool
        })
    value = funs_tflow.get_ops(prof)
    funs.update(key, value, RESULT)