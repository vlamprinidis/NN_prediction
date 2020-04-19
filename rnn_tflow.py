#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import os

BATCH = int(sys.argv[1])
EPOCHS = int(sys.argv[2])

RANK = int(sys.argv[3])
NODES = int(sys.argv[4])

print( 'Batch = {}, Epochs = {}, Rank = {}, Nodes = {}'.format(BATCH, EPOCHS, RANK, NODES) )

# In[ ]:


import tensorflow as  tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# print(x_train[0].shape)

def build_rnn():
    model = Sequential()
    
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax')) # activation must be softmax for categorical cross entropy loss
    
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=0)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
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
        net = build_rnn()
else:
    net = build_rnn()


# In[ ]:


logdir = 'logs/tf_rnn_BATCH{}_RANK{}_NODES{}'.format(BATCH,RANK,NODES)
os.system('rm -rf {}'.format(logdir))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True,update_freq='batch',
                                                  profile_batch=(1,60000//BATCH))


# In[ ]:


net.fit(x_train, y_train, batch_size = BATCH,
            epochs=EPOCHS, validation_data=(x_test, y_test),
            callbacks=[tensorboard_callback])


# In[ ]:


# %tensorboard --logdir logs --port 6006 --bind_all


# In[ ]:


# model.save('Minst_Lstm.model')

