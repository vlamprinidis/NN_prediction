import numpy as np
import pandas as pd

def give_df(file):  
    names = [
        'Epochs', 'Dataset size',
        'Number of features',
        'Channels',
        'Batch size',
        'Nodes',
        'Time (us)'
    ]
    
    n_all = [pd.read_csv('test/node_{}/{}'.format(node, file), sep=',', header=None,
                     names=names) for node in [1,2,3]]
    x = pd.concat(n_all)
    
    x['Time (s)'] = x['Time (us)']/1000/1000
    x = x.drop(columns=['Time (us)'])
    
    x = x.groupby(by=names[:-1]).mean()
    
    return x

