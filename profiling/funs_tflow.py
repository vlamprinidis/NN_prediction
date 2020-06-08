import numpy as np
import pandas as pd
import os
import socket
import wget

def get_df(logdir):
    host = socket.gethostname()
    dire = '{}/plugins/profile'.format(logdir)
    [entry] = os.listdir(dire)

    os.rename(os.path.join(dire,entry),(os.path.join(dire, 'my')))
    url = 'http://localhost:6006/data/plugin/profile/data?run=my&tag=tensorflow_stats&host={}&tqx=out:csv;'.format(host)

    out = './prof.csv'
    if os.path.exists(out):
        os.remove(out)
    wget.download(url,out)
    df = pd.read_csv(out, index_col=0)
    os.remove(out)
    
    return df

def get_tf_ops(logdir):
    df = get_df(logdir)
    df = df[['Type', 'Operation', '#Occurrences', 'Avg. self-time (us)']]
#     df = df[df['#Occurrences'] > 1]
    df = df.sort_values(by = ['Type', 'Operation']).reset_index(drop=True)
    
    return df
