import tensorflow as tf
import numpy as np
import pandas as pd
import os
import wget
import funs as h

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

def prepare(build_func, x, y, numf, nodes):
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
            model = build_func(numf)
    else:
        model = build_func(numf)
    
    return model

def profile(model, x, y, batch, epochs, nodes, use_prof):
    if use_prof:
        logdir = '/home/ubuntu/logs_tflow'
        os.system('rm -rf {}'.format(logdir))

        with tf.profiler.experimental.Profile(logdir):
            model.fit(x, y, batch_size = batch, epochs = epochs)
            pass

        _save(logdir, './out_tflow.csv')
#         numf, batch, nodes, it = str(numf), str(batch), str(nodes), str(it)
#         h.update(key=(layer, 'feat_' + numf, 'batch_' + batch, 'nodes_' + nodes, 'it_' + it), df=df, fname='tf.pkl')

    else:
        model.fit(x, y, batch_size = batch, epochs = epochs)