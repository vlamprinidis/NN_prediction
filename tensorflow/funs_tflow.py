import tensorflow as tf
import numpy as np
import pandas as pd
import os
import wget
import tf_data

import sys
sys.path.append('../')
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
    df = df[['Type', 'Operation', '#Occurrences', 
             'Avg. self-time (us)',
             'Total self-time (us)',
             'Avg. time (us)', 
             'Total time (us)'
            ]]
    df = df.sort_values(by = ['Type', 'Operation']).reset_index(drop=True)
    
    return df

def profile(model, x, y, batch, epochs):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(batch).prefetch(batch).repeat()
        return dataset
    
    import tempfile
    model_dir = tempfile.mkdtemp()
    keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir)
    
    if h.rank == 0:
        prof_file = 'out_tflow.csv'
        logdir = '/home/ubuntu/simple/tensorflow/logs'
        os.system('rm -rf {}'.format(logdir))

        with tf.profiler.experimental.Profile(logdir):
#             model.fit(tf_data, epochs = epochs)
            keras_estimator.train(input_fn=input_fn, steps=epochs*1024//batch)
            pass
        
        _save(logdir, prof_file)
        
        return prof_file
    
    else:
        model.fit(tf_data, epochs = epochs)
        
        return None