import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from numpy.random import RandomState as R

criterion = nn.CrossEntropyLoss
optimizer = torch.optim.SGD
learning_rate = 0.01

def dummy(dim, n):
    ds_size = 5000
    out_size = 10
    if dim == 1:
        x = R(42).random((ds_size, n))
        x = x.reshape(x.shape[0], 1, n)
    else:
        x = R(42).random((ds_size, n, n))
        x = x.reshape(x.shape[0], 1, n, n)
    
    y = R(42).randint(0,out_size,ds_size)
    
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
    
class Test:
    def sett(self, model):
        learning_rate = 0.01
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        self.model = model
        
class Dim1(Test):
    def __init__(self, numf):
        self.train_dataset = dummy(1,numf)
        self.numf = numf
        
    def sett(self, model):
        super().sett(model)
        
class Dim2(Test):
    def __init__(self, numf):
        self.train_dataset = dummy(2,numf)
        self.numf = numf
        
    def sett(self, model):
        super().sett(model)

class conv1d(Dim1):
    def __init__(self, numf):
        super().__init__(numf)
    
    def create(self):
        print('\n\nThis is torch-conv1d \n\n')
        numf = self.numf
        
        out_channels = 6
        kern = 5
        stride = 1

        lin_in = out_channels * ( conv_size_out(numf, kern, stride) )    

        model = nn.Sequential(
              nn.Conv1d(
                  in_channels = 1, out_channels = out_channels,
                  kernel_size = kern, stride = stride
              ),
              nn.Flatten(),
              nn.Linear(
                in_features = lin_in,
                out_features = 10
            )
        )
        
        super().sett(model)

class conv2d(Dim2):
    def __init__(self, numf):
        super().__init__(numf)

    def create(self):
        print('This is torch-conv2d \n')
        numf = self.numf
        
        out_channels = 6
        kern = 5
        stride = 1

        lin_in = out_channels * ( conv_size_out(numf, kern, stride) ** 2 )    

        model = nn.Sequential(
              nn.Conv2d(
                  in_channels = 1, out_channels = out_channels,
                  kernel_size = kern, stride = stride
              ),
              nn.Flatten(),
              nn.Linear(
                in_features = lin_in,
                out_features = 10
            )
        )
        
        super().sett(model)
        
class avg1d(Dim1):
    def __init__(self, numf):
        super().__init__(numf)

    def create(self):
        print('This is torch-avg1d \n')
        numf = self.numf
        kern = 5
        stride = 1
        
        lin_in = avg_size_out(numf, kern, stride)
        
        model = nn.Sequential(
            nn.AvgPool1d( kernel_size = kern, stride = stride ),
            nn.Flatten(),
            nn.Linear(
                in_features = lin_in,
                out_features = 10
            )
        )
        
        super().sett(model)

class avg2d(Dim2):
    def __init__(self, numf):
        super().__init__(numf)

    def create(self):
        print('This is torch-avg2d \n')
        numf = self.numf
        kern = 5
        stride = 1
        
        lin_in = avg_size_out(numf, kern, stride) ** 2
        
        model = nn.Sequential(
            nn.AvgPool2d( kernel_size = kern, stride = stride ),
            nn.Flatten(),
            nn.Linear(
                in_features = lin_in,
                out_features = 10
            )
        )
        
        super().sett(model)
        
class dense(Dim2):
    def __init__(self, numf):
        super().__init__(numf)

    def create(self):
        print('This is torch-linear \n')
        numf = self.numf
        
        lin_in = numf ** 2
        
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features = lin_in,
                out_features = 10
            )
        )
        
        super().sett(model)

class max1d(Dim1):
    def __init__(self, numf):
        super().__init__(numf)
    
    def create(self):
        print('This is torch-max1d \n')
        numf = self.numf
        kern = 5
        stride = 1

        lin_in = max_size_out(numf, kern, stride)
        
        model = nn.Sequential(
            nn.MaxPool1d( kernel_size = kern, stride = stride ),
            nn.Flatten(),
            nn.Linear(
                in_features = lin_in,
                out_features = 10
            )
        )
        
        super().sett(model) 

class max2d(Dim2):
    def __init__(self, numf):
        super().__init__(numf)

    def create(self):
        print('This is torch-max2d \n')
        numf = self.numf
        kern = 5
        stride = 1
        
        lin_in = max_size_out(numf, kern, stride) ** 2
        
        model = nn.Sequential(
            nn.MaxPool2d( kernel_size = kern, stride = stride ),
            nn.Flatten(),
            nn.Linear(
                in_features = lin_in,
                out_features = 10
            )
        )
        
        super().sett(model)         
        
mapp = {
    'avg1d': avg1d,
    'avg2d': avg2d,
    'conv1d': conv1d,
    'conv2d': conv2d,
    'max1d': max1d,
    'max2d': max2d,
    'dense': dense
}