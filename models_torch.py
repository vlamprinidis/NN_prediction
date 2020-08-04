import torch 
import torch.nn as nn
from numpy.random import RandomState as R

seed = 46

def dummy(dim, n, channels):
    ds_size = 1024
    out_size = 10
    if dim == 1:
        x = R(seed).random((ds_size, channels, n))
        x = x.reshape(x.shape[0], channels, n)
    else:
        x = R(seed).random((ds_size, channels, n, n))
        x = x.reshape(x.shape[0], channels, n, n)
    
    y = R(seed).randint(0,out_size,ds_size)
    
    x = torch.from_numpy(x).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)

    train_data = torch.utils.data.TensorDataset(x, y)
    
    return train_data
   
def conv_size_out(size_in, kern, stride):
    pad = 0
    dilation = 1
    return (size_in + 2*pad - dilation*(kern - 1) - 1) // stride + 1

def avg_size_out(size_in, kern, stride):
    pad = 0
    return (size_in + 2*pad - kern) // stride + 1

def max_size_out(size_in, kern, stride):
    pad = 0
    dilation = 1
    return (size_in + 2*pad - dilation*(kern - 1) - 1) // stride + 1
   
def base_relu(layer, lin_in):
    model = nn.Sequential(
        layer,
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(
            in_features = lin_in,
            out_features = 10
        )
    )
    
    return model

def base(layer, lin_in):
    model = nn.Sequential(
        layer,
        nn.Flatten(),
        nn.Linear(
            in_features = lin_in,
            out_features = 10
        )
    )
    
    return model
    
class Test:
    def __init__(self, numf, hp, dim, channels):
        self.train_dataset = dummy(dim, numf, channels)
        self.numf = numf
        self.hp = hp
        self.channels = channels
        
    def sett(self, model):
        learning_rate = 0.01
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        self.model = model
        
class Dim1(Test):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, dim = 1, channels = channels)
        
    def sett(self, model):
        super().sett(model)
        
class Dim2(Test):
    def __init__(self, numf, hp):
        super().__init__(numf, hp, dim = 2, channels = channels)
        
    def sett(self, model):
        super().sett(model)

class conv1d(Dim1):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, channels)
    
    def create(self):
        print('\n\nThis is torch-conv1d \n\n')
        numf = self.numf
        [out_channels, kern, stride] = self.hp
        in_channels = self.channels
        
        lin_in = out_channels * ( conv_size_out(numf, kern, stride) )
        
        model = base_relu( nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kern, stride = stride), lin_in )
        super().sett(model)

class conv2d(Dim2):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, channels)

    def create(self):
        print('This is torch-conv2d \n')
        numf = self.numf
        [out_channels, kern, stride] = self.hp
        in_channels = self.channels
        
        lin_in = out_channels * ( conv_size_out(numf, kern, stride) ** 2 )    

        model = base_relu( nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kern, stride = stride), lin_in)     
        super().sett(model)
        
class avg1d(Dim1):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, channels)

    def create(self):
        print('This is torch-avg1d \n')
        numf = self.numf
        [kern, stride, _] = self.hp
        lin_in = avg_size_out(numf, kern, stride)
        
        model = base(nn.AvgPool1d(kernel_size = kern, stride = stride), lin_in)
        super().sett(model)

class avg2d(Dim2):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, channels)

    def create(self):
        print('This is torch-avg2d \n')
        numf = self.numf
        [kern, stride, _] = self.hp
        lin_in = avg_size_out(numf, kern, stride) ** 2
        
        model = base( nn.AvgPool2d( kernel_size = kern, stride = stride ), lin_in)
        super().sett(model)
        
class dense(Dim2):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, channels)

    def create(self):
        print('This is torch-linear \n')
        numf = self.numf
        
        outf = self.hp[0]
        channels = self.channels
        
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = channels*(numf ** 2), out_features = outf),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features = outf,
                out_features = 10
            )
        )
        
        super().sett(model)

class max1d(Dim1):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, channels)
    
    def create(self):
        print('This is torch-max1d \n')
        numf = self.numf
        [kern, stride, _] = self.hp
        lin_in = max_size_out(numf, kern, stride)
        
        model = base( nn.MaxPool1d( kernel_size = kern, stride = stride ), lin_in)        
        super().sett(model) 

class max2d(Dim2):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, channels)

    def create(self):
        print('This is torch-max2d \n')
        numf = self.numf
        [kern, stride, _] = self.hp
        lin_in = max_size_out(numf, kern, stride) ** 2
        
        model = base( nn.MaxPool2d( kernel_size = kern, stride = stride ), lin_in)
        super().sett(model)         
    
class norm1d(Dim1):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, channels)

    def create(self):
        print('This is torch-norm1d \n')
        numf = self.numf
        channels = self.channels
        
        model = base( nn.BatchNorm1d(channels), numf )
        super().sett(model)
        
class norm2d(Dim2):
    def __init__(self, numf, hp, channels):
        super().__init__(numf, hp, channels)

    def create(self):
        print('This is torch-norm2d \n')
        numf = self.numf
        channels = self.channels
        
        model = base( nn.BatchNorm2d(channels), numf**2 )
        super().sett(model)
    
mapp = {
    'avg1d': avg1d,
    'avg2d': avg2d,
    'conv1d': conv1d,
    'conv2d': conv2d,
    'max1d': max1d,
    'max2d': max2d,
    'dense': dense,
    'norm1d': norm1d,
    'norm2d': norm2d
}