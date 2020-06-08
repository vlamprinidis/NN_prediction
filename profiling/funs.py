import pickle
import argparse
import socket

host = socket.gethostname()
ranks = {
    'vlas-1':0,
    'vlas-2':1,
    'vlas-3':2
}
rank = ranks[host]

# dict_ = { (<layer>,<numf>,<batch>,<nodes>,<it>) : <dataframe> }

def save(data, fname):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)
    fp.close()
    
def load(fname):
    with open(fname, 'rb') as fp:
        data = pickle.load(fp)
    fp.close()
        
    return data

def update(key, df, fname):
    data = load(fname)
    data[key] = df
    save(data, fname)

def parse():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-m', '--model', type=str, required=True)
    my_parser.add_argument('-numf', '--num_features', type=int, required=True)
    my_parser.add_argument('-b', '--batch', type=int, required=True)
    my_parser.add_argument('-n', '--nodes', type=int, required=True)
    my_parser.add_argument('-it', '--iteration', type=int, default=1)
    my_parser.add_argument('-e', '--epochs', type=int, default=5)
    args = my_parser.parse_args()

    print(args)
    
    return args