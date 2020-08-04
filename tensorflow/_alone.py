import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten
import argparse
from tf_data import give
import funs_tflow
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
parser.add_argument('-dim', type = int, required = True)
args = parser.parse_args()

DIM = args.dim
RESULT = '__alone{}d.tflow'.format(DIM)

model = Sequential()
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
        })
    value = funs_tflow.get_ops(prof)
    funs.update(key, value, RESULT)