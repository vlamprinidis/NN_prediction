import tensorflow as tf
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

import numpy as np
import pandas as pd
import os
import wget

import socket
host = socket.gethostname()
print(host)
ranks = {
    'vlas-1':0,
    'vlas-2':1,
    'vlas-3':2
}
rank = ranks[host]

# This can overwrite the file, don't use outside funs_tflow
def _save(logdir, target):
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
    df = df[['Type', 'Operation', '#Occurrences', 
             'Avg. self-time (us)',
             'Total self-time (us)',
             'Avg. time (us)', 
             'Total time (us)'
            ]]
    df = df.sort_values(by = ['Type', 'Operation']).reset_index(drop=True)
    
    return df

def distribute(strategy, Model, nodes):
    if(nodes < 2):
        raise NameError('More nodes needed')
        
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

    with strategy.scope():
        Model.create()
    
    return Model.model

tf_ops = ['conv1d', 'conv2d', 
          'average_pooling1d', 'average_pooling2d', 
          'max_pooling1d', 'max_pooling2d',
          'dense',
          'batch_normalization',
          'dropout',
          're_lu',
          'Tanh'
         ]

def check(keywords):
    def find(x):
        return any(
            [word in x for word in keywords]
        )
    return find

def total_on(df, words, column='Operation'):
    mask = df[column].apply(check(words))
    return df[mask]['Total self-time (us)'].sum()

def profile(model, dataset, batch, epochs):
    dataset = dataset.batch(batch)
    
    EPOCHS = epochs
    prof_file = 'out_tflow.csv'
    logdir = '/home/ubuntu/prof_run/logs'
    os.system('rm -rf {}'.format(logdir))

    with tf.profiler.experimental.Profile(logdir):
        model.fit(dataset, epochs = EPOCHS)
        pass

    _save(logdir, prof_file)

    df = get_ops(prof_file)

    return total_on(df, tf_ops)