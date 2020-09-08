import tensorflow as tf
from numpy.random import RandomState as R

seed = 42

def give1d(ds_size, numf, channels, out_size):
    x = R(seed).random((ds_size, numf, channels))
    x = x.reshape(x.shape[0], numf, channels)
    
    y = R(seed).randint(0,out_size,ds_size)
    y = tf.keras.utils.to_categorical(y, out_size)
    
    return tf.data.Dataset.from_tensor_slices((x, y))

def give2d(ds_size, numf, channels, out_size):
    x = R(seed).random((ds_size, numf, numf, channels))
    x = x.reshape(x.shape[0], numf, numf, channels)
    
    y = R(seed).randint(0,out_size,ds_size)
    y = tf.keras.utils.to_categorical(y, out_size)
    
    return tf.data.Dataset.from_tensor_slices((x, y))

def distribute(strategy, Model, nodes):
    import socket
    host = socket.gethostname()
    print(host)
    ranks = {
        'vlas-1':0,
        'vlas-2':1,
        'vlas-3':2
    }
    rank = ranks[host]
    
    import os
    import json    
    if(nodes < 2):
        raise NameError('More nodes needed')
        
    workers = []
    if nodes == 2:
        workers = ["10.0.1.121:8890", "10.0.1.104:8890"]
    else:
        workers = ["10.0.1.121:8890", "10.0.1.104:8890", "10.0.1.46:8890"]

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': workers
        },
        'task': {'type': 'worker', 'index': rank}
    })

    with strategy.scope():
        model = Model.create()
    
    return model
    
def train(model, dataset, batch, epochs):
    dataset = dataset.batch(batch)

    model.fit(dataset, epochs = 1)
    model.fit(dataset, epochs = epochs)
