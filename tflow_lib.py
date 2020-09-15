import tensorflow as tf
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

import numpy as np
import pandas as pd
import os
import wget

import funs

# This can overwrite the file, don't use outside funs_tflow
def _save(logdir, target):
    host = funs.host
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
        'task': {'type': 'worker', 'index': funs.rank}
    })

    with strategy.scope():
        Model.create()

tf_ops = ['Conv1D', 'Conv2D', 
          'AveragePooling1D', 'AveragePooling2D', 
          'MaxPooling1D', 'MaxPooling2D',
          'Dense',
          'BatchNormalization',
          'Dropout',
          'ReLU',
          'tanh'
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

def profile(model, x, y, batch, epochs):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch)
    
    EPOCHS = epochs
    if funs.rank == 0:
        prof_file = 'out_tflow.csv'
        logdir = '/home/ubuntu/prof_run/logs'
        os.system('rm -rf {}'.format(logdir))

        with tf.profiler.experimental.Profile(logdir):
            model.fit(dataset, epochs = EPOCHS)
            pass
        
        _save(logdir, prof_file)
        
        df = get_ops(prof_file)

        return total_on(tf_ops)
    
    else:
        model.fit(dataset, epochs = EPOCHS)
        
        return None