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

def my_key(model_str, numf, batch, nodes, it):
    key = frozenset({
        ('model_str', model_str),
        ('numf', numf),
        ('batch', batch),
        ('nodes', nodes),
        ('it', it)
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
    
def load(fname):
    if( not os.path.exists(fname) ):
        print( 'No such file: {}'.format(fname) )
        return None

    with open(fname, 'rb') as fp:
        data = pickle.load(fp)
    fp.close()
        
    return data

# Creates the file if it doesn't already exist
def update(key, value, fname):
    if( not os.path.exists(fname) ):
        _save({}, fname)

    data = load(fname)
    data[key] = value
    _save(data, fname)

def get_keys(fname):
    data = load(fname)
    
    return list(data.keys())
    
def get_value(model_str, numf, batch, nodes, it, fname):
    data = load(fname)
    key = my_key(model_str, numf, batch, nodes, it)
    
    value = data.get(key)
    if value == None:
        print('No such key')
        
    return value

def parse(model_lst):
    print('\n')
    print('This is ' + host)

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-m', '--model', type=str, required=True, 
                           choices=model_lst)
    my_parser.add_argument('-numf', '--num_features', type=int, required=True,
                          choices=[16, 32, 64, 128, 256])
    my_parser.add_argument('-b', '--batch', type=int, required=True, 
                           choices=[32, 64, 128, 256, 512])
    my_parser.add_argument('-n', '--nodes', type=int, required=True,
                          choices=[1, 2, 3])
    my_parser.add_argument('-it', '--iteration', type=int, default=1)
    my_parser.add_argument('-e', '--epochs', type=int, default=5)
    my_parser.add_argument('-use_prof', '--use_profiler', type=bool, default=True)
    
    args = my_parser.parse_args()
    
    print('\n')
    print(args)
    print('\n')
    
    return args