import tensorflow as tf
import numpy as np
import pandas as pd
import os
import wget
import funs as h

# This can overwrite the file, don't use outside funs_tflow
def _save(logdir, target):
    host = h.host
    dire = '{}/plugins/profile'.format(logdir)
    [entry] = os.listdir(dire)

    os.rename(os.path.join(dire,entry),(os.path.join(dire, 'my')))
    url = 'http://localhost:6006/data/plugin/profile/data?run=my&tag=tensorflow_stats&host={}&tqx=out:csv;'.format(host)
    out = target
    if os.path.exists(out):
        os.remove(out)
    wget.download(url,out)

def get_ops(source):
    df = pd.read_csv(source, index_col=0)
    df = df[['Type', 'Operation', '#Occurrences', 'Avg. self-time (us)']]
    df = df.sort_values(by = ['Type', 'Operation']).reset_index(drop=True)
    
    return df

def prepare(model_class, nodes):
    rank = h.rank
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
            model_class.create()
    else:
        model_class.create()

def profile(model_class, batch, epochs):
    model, x, y = model_class.model, model_class.x, model_class.y
    prof_file = './out_tflow.csv'

    logdir = '/home/ubuntu/logs_tflow'
    os.system('rm -rf {}'.format(logdir))

    with tf.profiler.experimental.Profile(logdir):
        model.fit(x, y, batch_size = batch, epochs = epochs)
        pass

    _save(logdir, prof_file)

    return prof_file


### this is for creating the graph
#         tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
#                                              profile_batch=3)

#         model.fit(x, y, batch_size = batch, steps_per_epoch=3, epochs = 1, callbacks=[tb_callback])