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
RESULT = '__relu{}d.tflow'.format(DIM)
NAME = 'RELU{}D'.format(DIM)

model = Sequential()
model.add( 
    layers.ReLU(name = NAME)
)
model.add( Flatten(name='FLATTEN') )
model.add( Dense(units = 10, name='FINAL_DENSE') )

opt = tf.keras.optimizers.SGD(learning_rate=0.01)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

nodes = args.nodes
if nodes > 1:
    workers = []
    if nodes == 2:
        workers = ["10.0.1.121:8890", "10.0.1.104:8890"]
    else:
        workers = ["10.0.1.121:8890", "10.0.1.104:8890", "10.0.1.46:8890"]
    import json
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': workers
        },
        'task': {'type': 'worker', 'index': funs.rank}
    })

    with strategy.scope():
        model.compile(loss=loss, optimizer=opt,
                  metrics=['accuracy'])
else:
    model.compile(loss=loss, optimizer=opt,
                      metrics=['accuracy'])

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