import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten
import argparse
# from tf_data import give
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
parser.add_argument('-filters', type = int, required = True)
parser.add_argument('-kernel', type = int, required = True)
parser.add_argument('-stride', type = int, required = True)
parser.add_argument('-dim', type = int, required = True)
args = parser.parse_args()

DIM = args.dim
NAME = 'CONV{}D'.format(DIM)
RESULT = '__conv{}d.tflow'.format(DIM)
conv = layers.Conv1D if DIM==1 else layers.Conv2D
    
model = Sequential()
model.add( 
    conv(filters = args.filters, kernel_size = args.kernel, 
                                     name = NAME)
)
model.add( Flatten(name='FLATTEN') )
model.add( Dense(units = 10, name='FINAL_DENSE') )

if args.nodes > 1:
    model = distribute(strategy, model, args.nodes)
else:
    model.compile(loss = funs_tflow.loss, optimizer = funs_tflow.opt, metrics=['accuracy'])

from numpy.random import RandomState as R
def give(dim, n, channels):
    ds_size = 1024
    out_size = 10
    if dim == 1:
        x = R(seed).random((ds_size, n, channels))
        x = x.reshape(x.shape[0], n, channels)
    else:
        x = R(seed).random((ds_size, n, n, channels))
        x = x.reshape(x.shape[0], n, n, channels)
    
    y = R(seed).randint(0,out_size,ds_size)
    y = tf.keras.utils.to_categorical(y, out_size)
    
    return x,y

x,y = give(DIM, args.numf, args.channels)

prof = funs_tflow.profile(model, x, y, args.batch, args.epochs)

if prof != None:
    key = funs.my_key({
            'numf':args.numf,
            'batch':args.batch,
            'nodes':args.nodes,
            'channels':args.channels,
            'filters':args.filters,
            'kernel':args.kernel,
            'stride':args.stride
        })
    value = funs_tflow.get_ops(prof)
    funs.update(key, value, RESULT)