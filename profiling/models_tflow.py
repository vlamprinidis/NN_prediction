import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras.layers import Dense, Flatten
from numpy.random import RandomState as R

seed = 44

def dummy(dim, n):
    ds_size = 5000
    out_size = 10
    if dim == 1:
        x = R(seed).random((ds_size, n))
        x = x.reshape(x.shape[0], n, 1)
    else:
        x = R(seed).random((ds_size, n, n))
        x = x.reshape(x.shape[0], n, n, 1)
    
    y = R(seed).randint(0,out_size,ds_size)
    y = tf.keras.utils.to_categorical(y, out_size)
    
    return tf.data.Dataset.from_tensor_slices((x, y))
    
def base(layer):    
    model = Sequential()
    model.add( layer )
    model.add( Flatten(name='FLATTEN') )
    model.add( Dense(units = 10, name='FINAL_DENSE') )
    
    return model

class Test:
    def __init__(self, numf, hp, dim):
        self.numf = numf
        self.hp = hp
        self.tf_data = dummy(dim,numf)
        
    def sett(self, model):
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        loss = 'categorical_crossentropy'
        metric = 'accuracy'
        
        model.compile(loss=loss, optimizer=opt,
                     metrics=[metric])
        
        self.model = model
        
class Dim1(Test):
    def __init__(self, numf, hp):
        super().__init__(numf, hp = hp, dim = 1)

    def sett(self, model):
        super().sett(model)

class Dim2(Test):
    def __init__(self, numf, hp):
        super().__init__(numf, hp = hp, dim = 2)

    def sett(self, model):
        super().sett(model)

class conv1d(Dim1):
    def __init__(self, numf, hp):
        super().__init__(numf, hp)
    
    def create(self):
        print('\n\nThis is tflow-conv1d \n\n')
        super().sett( base(layers.Conv1D(filters = 1, kernel_size = self.hp, name = 'CONV1D', activation = 'relu')) )

class conv2d(Dim2):
    def __init__(self, numf, hp):
        super().__init__(numf, hp)
        
    def create(self):
        print('\n\nThis is tflow-conv2d \n\n')
        super().sett( base(layers.Conv2D(filters = 1, kernel_size = self.hp, name = 'CONV2D', activation = 'relu')) )

class avg1d(Dim1):
    def __init__(self, numf, hp):
        super().__init__(numf, hp)
    
    def create(self):
        print('\n\nThis is tflow-avg1d \n\n')
        super().sett( base(layers.AveragePooling1D(pool_size = self.hp, name = 'AVG1D')) )
    
class avg2d(Dim2):
    def __init__(self, numf, hp):
        super().__init__(numf, hp)
        
    def create(self):
        print('\n\nThis is tflow-avg2d \n\n')
        super().sett( base(layers.AveragePooling2D(pool_size = self.hp, name='AVG2D')) )

class max1d(Dim1):
    def __init__(self, numf, hp):
        super().__init__(numf, hp)
        
    def create(self):
        print('\n\nThis is tflow-max1d \n\n')
        super().sett( base(layers.MaxPool1D(pool_size = self.hp, name = 'MAX1D')) )
    
class max2d(Dim2):
    def __init__(self, numf, hp):
        super().__init__(numf, hp)
        
    def create(self):
        print('\n\nThis is tflow-max2d \n\n')
        super().sett( base(layers.MaxPool2D(pool_size = self.hp, name = 'MAX2D')) )

class dense(Dim2):
    def __init__(self, numf, hp):
        super().__init__(numf, hp)
    
    def create(self):
        print('\n\nThis is tflow-dense \n\n')
        super().sett( base(layers.Dense(units = self.hp, name = 'DENSE', activation = 'relu')) )

class norm1d(Dim1):
    def __init__(self, numf, hp):
        super().__init__(numf, hp)
    
    def create(self):
        print('\n\nThis is tflow-norm1d \n\n')
        super().sett( base(layers.BatchNormalization(name = 'NORM1D')) )

class norm2d(Dim2):
    def __init__(self, numf, hp):
        super().__init__(numf, hp)
    
    def create(self):
        print('\n\nThis is tflow-norm2d \n\n')
        super().sett( base(layers.BatchNormalization(name = 'NORM2D')) )

mapp = {
    'avg1d': avg1d,
    'avg2d': avg2d,
    'conv1d': conv1d,
    'conv2d': conv2d,
    'max1d': max1d,
    'max2d': max2d,
    'dense': dense,
    'norm1d':norm1d,
    'norm2d': norm2d
}