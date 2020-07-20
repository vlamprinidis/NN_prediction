import sys
pth = '/home/ubuntu/diploma'
if(pth not in sys.path):
    sys.path.insert(0, pth)
    
import tensorflow as  tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Dropout, LSTM
from tensorflow import keras

# Distribute tensorflow if needed
def tf_distribute(nodes, build):    
    if nodes > 1:
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
            'task': {'type': 'worker', 'index': RANK}
        })
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            net = build()
    else:
        net = build()
    
    return net

from funs.maps import maps
m = maps()

# tensorflow mappings
def give_tf_model(model, batch, rank, nodes):
    
    # check for error
    if(model not in m.known_models
       or batch not in m.batches
       or rank not in m.ranks
       or nodes not in m.nodes
      ):
        print('Unknown combo for model creation\n')
        exit()
            
    if model == 'cnn':
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

        net = tf_distribute(nodes, build_cnn)
        
        return net, x_train, y_train
    
    elif model == 'rnn':
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
        
        net = tf_distribute(nodes, build_rnn)
        
        return net, x_train, y_train


        