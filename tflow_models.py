import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten

opt = tf.keras.optimizers.SGD(learning_rate=0.01)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 28x28 input
class LeNet_1:        
    def create(self):
        model = Sequential()
        model.add( 
            layers.Conv2D(filters = 4, kernel_size = 5)
        )
        model.add(
            layers.ReLU()       
        )
        model.add(
            layers.MaxPooling2D(pool_size = 2, strides = 2)
        )
        model.add( 
            layers.Conv2D(filters = 12, kernel_size = 5)
        )
        model.add(
            layers.ReLU()        
        )
        model.add(
            layers.MaxPooling2D(pool_size = 2, strides = 2)
        )
        model.add( Flatten() )
        model.add( Dense(units = 10) )
        
        model.compile(loss = loss, optimizer = opt, metrics=['accuracy'])
        
        self.model = model
        
        return model
    
# 32x32 input
class LeNet_5:        
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
    
# 224x224x3
class VGG_11:
    def create(self):
        model = Sequential()
        #L1
        model.add(layers.Conv2D(64, kernel_size = 3, strides = 1))
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.4))
        model.add(layers.MaxPooling2D(pool_size = 2, strides = 2))
        
        #L2
        model.add(layers.Conv2D(128, kernel_size = 3, strides = 1))
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.4))
        model.add(layers.MaxPooling2D(pool_size = 2, strides = 2))
        
        #L3
        model.add(layers.Conv2D(256, kernel_size = 3, strides = 1))
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.4))
        
        #L4
        model.add(layers.Conv2D(256, kernel_size = 3, strides = 1))
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.4))
        model.add(layers.MaxPooling2D(pool_size = 2, strides = 2))

        #L5
        model.add(layers.Conv2D(512, kernel_size = 3, strides = 1))
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.4))
        
        #L6
        model.add(layers.Conv2D(512, kernel_size = 3, strides = 1))
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.4))
        model.add(layers.MaxPooling2D(pool_size = 2, strides = 2))

        #L7
        model.add(layers.Conv2D(512, kernel_size = 3, strides = 1))
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.4))
        
        #L8
        model.add(layers.Conv2D(512, kernel_size = 3, strides = 1))
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.4))
        model.add(layers.MaxPooling2D(pool_size = 2, strides = 2))        
        
        #L9
        model.add(Flatten())
        model.add(Dense(4096))
        
        #L10
        model.add(Dense(4096))
        
        #L11
        model.add(Dense(1000))

        model.compile(loss = loss, optimizer = opt, metrics=['accuracy'])
        
        self.model = model
        
        return model
        
# 227x227x3
class AlexNet:
    def create(self):
        model = Sequential([
            layers.Conv2D(filters=96, kernel_size=11, strides=4),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=3, strides=2),
            
            layers.Conv2D(filters=256, kernel_size=5, strides=1),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=3, strides=2),
            
            layers.Conv2D(filters=384, kernel_size=3, strides=1),
            layers.ReLU(),
            layers.BatchNormalization(),
            
            layers.Conv2D(filters=384, kernel_size=1, strides=1),
            layers.ReLU(),
            layers.BatchNormalization(),
            
            layers.Conv2D(filters=256, kernel_size=1, strides=1),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=3, strides=2),
            
            layers.Flatten(),
            layers.Dense(4096),
            layers.ReLU(),
            layers.Dropout(0.5),
            
            layers.Dense(4096),
            layers.ReLU(),
            layers.Dropout(0.5),
            
            layers.Dense(1000)
        ])
        
        model.compile(loss = loss, optimizer = opt, metrics=['accuracy'])        
        
        self.model = model
        
        return self.model

    