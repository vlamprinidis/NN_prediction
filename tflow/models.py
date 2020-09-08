import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten

opt = tf.keras.optimizers.SGD(learning_rate=0.01)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 32x32 input
class LeNet5:        
    def create(self):
        model = Sequential()
        model.add( 
            layers.Conv2D(filters = 6, kernel_size = 5)
        )
        model.add(
            layers.Activation('tanh')        
        )
        model.add(
            layers.AveragePooling2D(pool_size = 2, strides = 2)
        )
        model.add( 
            layers.Conv2D(filters = 16, kernel_size = 5)
        )
        model.add(
            layers.Activation('tanh')        
        )
        model.add(
            layers.AveragePooling2D(pool_size = 2, strides = 2)
        )
        model.add( Flatten() )
        model.add( Dense(units = 120) )
        model.add(
            layers.Activation('tanh')        
        )
        
        model.add( Flatten() )
        model.add( Dense(units = 84) )
        model.add(
            layers.Activation('tanh')        
        )
        
        model.add( Flatten() )
        model.add( Dense(units = 10) )
        
        model.compile(loss = loss, optimizer = opt, metrics=['accuracy'])
        
        self.model = model
        
        return model
                