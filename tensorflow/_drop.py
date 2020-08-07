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
parser = funs.arg_all(parser)
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
        model.compile(loss = funs_tflow.loss, optimizer = funs_tflow.opt, metrics=['accuracy'])
        self.model = model

Model = Drop()
if args.nodes > 1:
    distribute(strategy, Model, args.nodes)
else:
    Model.create()

x,y = give(DIM, args.numf, args.channels)

prof = funs_tflow.profile(Model.model, x, y, args.batch, args.epochs)

if prof != None:
    key = funs.my_key({
            'numf':args.numf,
            'batch':args.batch,
            'nodes':args.nodes,
            'channels':args.channels,
            'drop':args.drop
        })
    value = funs_tflow.get_ops(prof)
    funs.update(key, value, RESULT)