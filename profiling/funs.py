import pickle
import argparse
import socket
import os

host = socket.gethostname()
ranks = {
    'vlas-1':0,
    'vlas-2':1,
    'vlas-3':2
}
rank = ranks[host]

def my_key(model_str, numf, hp, batch, nodes):
    key = frozenset({
        ('model_str', model_str),
        ('numf', numf),
        ('hp', hp),
        ('batch', batch),
        ('nodes', nodes)
    })
    
    return key

def my_value(df, epochs):
    value = {
        'table':df,
        'epochs': epochs,
        'rank': rank
    }
    
    return value

# This can overwrite the file, don't use outside funs
def _save(data, fname):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)
    fp.close()

# Creates the file if it doesn't already exist
def load(fname):
    if( not os.path.exists(fname) ):
        print( 'No such file: {}'.format(fname) )
        print( 'Creating empty file: {}'.format(fname) )
        _save({}, fname)
        return {}

    with open(fname, 'rb') as fp:
        data = pickle.load(fp)
    fp.close()
        
    return data

def update(key, value, fname):
    data = load(fname)
    data[key] = value
    _save(data, fname)
    print('\nAdded: {}\n'.format(key))

def get_keys(fname):
    data = load(fname)
    return list(data.keys())
    
def get_value(model_str, numf, hp, batch, nodes, it, fname):
    data = load(fname)
    key = my_key(model_str, numf, hp, batch, nodes, it)
    
    value = data.get(key)
    if value == None:
        print('No such key')
        
    return value

numf_ls = [16, 32, 64, 128]
batch_ls = [32, 64, 128, 256, 512]
nodes_ls = [1,2,3]

hp_map = {
    'avg1d': [2,4],
    'avg2d': [2,4],
    'conv1d': [2,4,8],
    'conv2d': [2,4,8],
    'max1d': [2,4],
    'max2d': [2,4],
    'dense': [32,64,128],
    'norm1d':[0],
    'norm2d': [0]
}

def prof_parse():
    print('\n')
    print('This is ' + host)

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-m', '--model', type=str, required=True, 
                           choices = list(hp_map.keys()))
    my_parser.add_argument('-numf', '--num_features', type=int, required=True,
                          choices = numf_ls )
    
    my_parser.add_argument('-hp', '--hyper_param', type=int, required=True )
    
    my_parser.add_argument('-b', '--batch', type=int, required=True, 
                           choices = batch_ls )
    my_parser.add_argument('-n', '--nodes', type=int, required=True,
                          choices = nodes_ls )
    my_parser.add_argument('-e', '--epochs', type=int, default=5)
    
    args = my_parser.parse_args()
    
    print('\n')
    print(args)
    print('\n')
    
    return args

