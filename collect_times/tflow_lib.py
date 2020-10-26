import tensorflow as tf
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

import numpy as np
import pandas as pd
import os
import wget

import sys
sys.path.append('/home/ubuntu/logs')

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

tf_typ_op = [
    (['AvgPool'],None),
    (['Conv2D', '_FusedConv2D', 'Conv2DBackpropFilter'],None),
    (['MatMul'],['dense']),
    (['RandomUniform'],None),
    (['_FusedMatMul', 'MatMul'],['FINAL_DENSE']),
    (['MaxPool'],None),
    (['SquaredDifference', 'Mean', 'FusedBatchNormV3', 'FusedBatchNormGradV3'],None),
    (['Relu'],None),
    (['Tanh'],None)
]

def check(keywords):
    def find(x):
        return any(
            [word in x for word in keywords]
        )
    return find

def check_just(keywords):
    def find(x):
        return any(
            [word == x for word in keywords]
        )
    return find

def total_on(_df, words_typ = None, words_op = None):  
    
    df = _df
    
    if words_typ != None:
        mask = df['Type'].apply(check_just(words_typ))
        df = df[mask]
    
    if words_op != None:
        mask = df['Operation'].apply(check(words_op))
        df = df[mask]
    
    return df['Total self-time (us)'].sum()


def profile(model, dataset, steps, epochs):    
    EPOCHS = epochs
    prof_file = 'out_tflow.csv'
    logdir = '/home/ubuntu/logs'
    os.system('rm -rf {}'.format(logdir))
    
    import math

    with tf.profiler.experimental.Profile(logdir):
        model.fit(dataset, epochs = EPOCHS, steps_per_epoch=math.floor(steps))
        pass

    _save(logdir, prof_file)

    df = get_ops(prof_file)

    return sum([total_on(df,typ,op) for typ,op in tf_typ_op])

