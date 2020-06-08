# before running this script run:
# tensorboard --logdir /home/ubuntu/tf_logs --bind_all

import tensorflow as tf
import numpy as np
import os
import sys

import funs
import models_tflow as mod

# dict_ = { (<layer>,<numf>,<batch>,<nodes>,<it>) : <dataframe> }
mapp = {
    'avg1d':mod.avg1d,
    'avg2d':mod.avg2d,
    'conv1d':mod.conv1d,
    'conv2d':mod.conv2d
}

# Distribute tensorflow if needed
def tf_distribute(build_func, numf, rank, nodes):
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
            'task': {'type': 'worker', 'index': rank}
        })
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            model = build_func(numf)
    else:
        model = build_func(numf)
    
    return model

layer, numf, batch, rank, nodes, it = funs.parse(sys.argv)

model,x,y = tf_distribute(mapp[layer], numf, rank, nodes)

epochs = 1
if rank == 0:
    logdir = '/home/ubuntu/tf_logs'
    os.system('rm -rf {}'.format(logdir))

    with tf.profiler.experimental.Profile(logdir):
        model.fit(x, y, batch_size = batch, epochs = epochs)
        pass

    df = funs.get_tf_ops(logdir)
    numf, batch, nodes, it = str(numf), str(batch), str(nodes), str(it)
    funs.update(key=(layer, 'feat_' + numf, 'batch_' + batch, 'nodes_' + nodes, 'it_' + it), df=df, fname='tf.pkl')
    
else:
    model.fit(x, y, batch_size = batch, epochs = epochs)