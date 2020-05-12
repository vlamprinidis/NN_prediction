#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import os

BATCH = int(sys.argv[1])

RANK = int(sys.argv[2])
NODES = int(sys.argv[3])

print( 'Batch = {}, Rank = {}, Nodes = {}'.format(BATCH, RANK, NODES) )

# In[ ]:


import tensorflow as  tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from tensorflow import keras


# In[ ]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

rows, cols = 28, 28
 
x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)

input_shape = (rows, cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# one-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def build_cnn():
    
    CONV_KERNEL = 5
    CONV_STRIDE = 1

    AVG_KERNEL = 2
    AVG_STRIDE = 2
        
    model = Sequential()
    
    model.add(
        Conv2D(filters=6, 
               kernel_size=(CONV_KERNEL, CONV_KERNEL), 
               strides=(CONV_STRIDE, CONV_STRIDE),
               input_shape = input_shape,
               activation='tanh')
    )
    
    model.add(
        AveragePooling2D(pool_size=(2, 2), 
                         strides=(2, 2))
    )
    
    model.add(
        Conv2D(filters=16, 
               kernel_size=(CONV_KERNEL, CONV_KERNEL), 
               strides=(CONV_STRIDE, CONV_STRIDE), 
               activation='tanh')
    )
    
    model.add(
        AveragePooling2D(pool_size=(AVG_KERNEL, AVG_KERNEL), 
                         strides=(AVG_STRIDE, AVG_STRIDE))
    )
    
    model.add(Flatten())
    
    model.add(
        Dense(units = 120, activation='tanh')
    )
    
    model.add(Flatten())
    
    model.add(
        Dense(units = 84, activation='tanh')
    )
    
    model.add(
        Dense(units = 10, activation='softmax')
    )
    
    opt = tf.keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt,
             metrics=['accuracy'])
  
    return model


# In[ ]:


if NODES > 1:
    workers = []
    if NODES == 2:
        workers = ["10.0.1.121:8890", "10.0.1.104:8890"]
    else:
        workers = ["10.0.1.121:8890", "10.0.1.104:8890", "10.0.1.46:8890"]
    import json
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': workers
        },
        'task': {'type': 'worker', 'index': RANK}
    })
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        net = build_cnn()
else:
    net = build_cnn()


sys.path.append('../')
import funs.organise as org
import funs.graph as g
    
if RANK == 0:
    pred = g.tf(net, x_train, y_train, BATCH)
    
    data = org.load_dict('pred_data.pkl')
    
    diff = org.diff_tf(data['simple_tflow_cnn', BATCH, NODES], pred)
    data['diff_tflow_cnn', BATCH, NODES] = diff
    
    org.save_dict(data, 'pred_data.pkl')
else:
    g.just_tf(net, x_train, y_train, BATCH)