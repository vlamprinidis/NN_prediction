import sys
import os
import numpy as np

# Tensorflow

import tensorflow as  tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, Flatten, Conv1D, Conv2D, Dense, Dropout, ELU, Embedding, Flatten, GaussianDropout, GaussianNoise, MaxPool1D, MaxPool2D, ReLU, Softmax

# def pad(arr):
#     return np.pad(arr, ((0,0),(2,2),(2,2)))

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), _ = mnist.load_data()

# x = pad(x_train)

# rows, cols = 32, 32
 
# x = x.reshape(x.shape[0], rows, cols, 1)

# input_shape = (rows, cols, 1)

# x = x.astype('float32')

# # one-hot encode the labels
# y = tf.keras.utils.to_categorical(y_train, 10)

def ran2d(sz_in, sz_out, sz_dataset=60000):
    x = np.random.rand(sz_dataset, sz_in, sz_in)*256

    y = np.random.randint(0, sz_out, sz_dataset)

    rows, cols = sz_in, sz_in

    x = x.reshape(x.shape[0], rows, cols, 1)

    input_shape = (rows, cols, 1)

    x = x.astype('float32')

    # one-hot encode the labels
    y = tf.keras.utils.to_categorical(y, sz_out)
    
    return x,y,input_shape

def tf_conv2d(szfeat):
    x_train, y_train, input_shape = ran2d(32,szfeat)
    
    train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
        
#     model = Conv2D(filters=szfeat, 
#                kernel_size=(5, 5), 
#                input_shape = input_shape)
#     model.compile()
    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(filters=32, kernel_size=5, input_shape=input_shape)
        
        def call(self, x):
            x = self.conv1(x)
            
            return x
    model = MyModel()
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)
    
    EPOCHS = 1

    for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100)
             )

# PyTorch

import torch

from torch.nn import Conv1d, Conv2d, MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, ReLU, ELU, Softmax, Linear, Dropout


import torchvision
import torchvision.transforms as transforms
