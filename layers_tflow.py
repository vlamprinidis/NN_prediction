import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten
from numpy.random import RandomState as R

seed = 46

def dummy(dim, n, channels):
    ds_size = 1024
    out_size = 10
    if dim == 1:
        x = R(seed).random((ds_size, n, channels))
        x = x.reshape(x.shape[0], n, channels)
    else:
        x = R(seed).random((ds_size, n, n, channels))
        x = x.reshape(x.shape[0], n, n, channels)
    
    y = R(seed).randint(0,out_size,ds_size)
#     y = tf.keras.utils.to_categorical(y, out_size)
    
    return tf.data.Dataset.from_tensor_slices((x, y))
    
def base(layer):    
    model = Sequential()
    model.add( layer )
    model.add( Flatten(name='FLATTEN') )
    model.add( Dense(units = 10, name='FINAL_DENSE') )
    
    return model

class Test:
    def __init__(self, dim, numf, channels, hp, 
                 opt=tf.keras.optimizers.SGD(learning_rate=0.01),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
        self.numf = numf
        self.channels = channels
        self.hp = hp
        self.tf_data = dummy(dim=dim, n=numf, channels=channels)
        
        self.opt = opt
        self.loss = loss
        
    def sett(self, model):
        model.compile(loss=self.loss, optimizer=self.opt,
                      metrics=['accuracy'])
        
        self.model = model

class conv1d(Test):
    def __init__(self, *a, **k):
        super().__init__(1, *a, **k)
    
    def create(self):
        print('\n\nThis is tflow-conv1d \n\n')
        super().sett( base(layers.Conv1D(filters = self.hp['filters'], kernel_size = self.hp['kernel'], 
                                         name = 'CONV1D', activation = 'relu')) )

class conv2d(Test):
    def __init__(self, *a, **k):
        super().__init__(2, *a, **k)
        
    def create(self):
        print('\n\nThis is tflow-conv2d \n\n')
        hp = self.hp
        super().sett( base(layers.Conv2D(filters = hp['filters'], kernel_size = hp['kernel'], 
                                         name = 'CONV2D', activation = 'relu')) )

class avg1d(Test):
    def __init__(self, *a, **k):
        super().__init__(1, *a, **k)
    
    def create(self):
        print('\n\nThis is tflow-avg1d \n\n')
        
        super().sett( base(layers.AveragePooling1D(pool_size = self.hp['pool'], stride = self.hp['stride'], 
                                                   name = 'AVG1D')) )
    
class avg2d(Test):
    def __init__(self, *a, **k):
        super().__init__(2, *a, **k)
        
    def create(self):
        print('\n\nThis is tflow-avg2d \n\n')
        
        super().sett( base(layers.AveragePooling2D(pool_size = self.hp['pool'], stride = self.hp['stride'], 
                                                   name='AVG2D')) )

class max1d(Test):
    def __init__(self, *a, **k):
        super().__init__(1, *a, **k)
        
    def create(self):
        print('\n\nThis is tflow-max1d \n\n')
        
        super().sett( base(layers.MaxPool1Dpool_size = self.hp['pool'], stride = self.hp['stride'], 
                           name = 'MAX1D')) )
    
class max2d(Test):
    def __init__(self, *a, **k):
        super().__init__(2, *a, **k)
        
    def create(self):
        print('\n\nThis is tflow-max2d \n\n')
        
        super().sett( base(layers.MaxPool2D(pool_size = self.hp['pool'], stride = self.hp['stride'], 
                                            name = 'MAX2D')) )

class dense1d(Test):
    def __init__(self, *a, **k):
        super().__init__(1, *a, **k)
    
    def create(self):
        print('\n\nThis is tflow-dense1d \n\n')
        super().sett( base(layers.Dense(units = hp['units'], name = 'DENSE', activation = 'relu')) )

class dense2d(Test):
    def __init__(self, *a, **k):
        super().__init__(2, *a, **k)
    
    def create(self):
        print('\n\nThis is tflow-dense2d \n\n')
        super().sett( base(layers.Dense(units = hp['units'], name = 'DENSE', activation = 'relu')) )

class norm1d(Test):
    def __init__(self, *a, **k):
        super().__init__(1, *a, **k)
    
    def create(self):
        print('\n\nThis is tflow-norm1d \n\n')
        super().sett( base(layers.BatchNormalization(name = 'NORM1D')) )

class norm2d(Test):
    def __init__(self, *a, **k):
        super().__init__(2, *a, **k)
    
    def create(self):
        print('\n\nThis is tflow-norm2d \n\n')
        super().sett( base(layers.BatchNormalization(name = 'NORM2D')) )

class drop1d(Test):
    def init(self, *a, **k):
        super().__init__(1, *a, **k)

    def create(self):
        print('\n\nThis is tflow-drop1d \n\n')
        
        super().sett( base(layers.Dropout(hp['pr'], name = 'DROP')) )

class drop2d(Test):
    def init(self, *a, **k):
        super().__init__(2, *a, **k)

    def create(self):
        print('\n\nThis is tflow-drop2d \n\n')
        
        super().sett( base(layers.Dropout(hp['pr'], name = 'DROP')) )

# class relu(Test):
    
mapp = {
    'avg1d': avg1d,
    'avg2d': avg2d,
    'conv1d': conv1d,
    'conv2d': conv2d,
    'max1d': max1d,
    'max2d': max2d,
    'dense1d': dense1d,
    'dense2d': dense2d,
    'norm1d':norm1d,
    'norm2d': norm2d,
    'drop1d': drop1d,
    'drop2d': drop2d
}