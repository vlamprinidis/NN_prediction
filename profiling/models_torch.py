import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from numpy.random import RandomState as R

criterion = nn.CrossEntropyLoss
optimizer = torch.optim.SGD
learning_rate = 0.01

# def dataset(dim):
#     if(dim == 2):
#         trans = torchvision.transforms.Compose([
#             transforms.Resize(32),
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     else:# dim == 1
#         trans = torchvision.transforms.Compose([
#             transforms.Resize(32),
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,)),
#             transforms.Lambda(lambda x: x.view(1,-1))
#         ])
        
#     train_dataset = torchvision.datasets.MNIST(
#         root='./mnist_torch',
#         train=True, 
#         transform=trans, 
#         download=True
#     )
#     return train_dataset

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
    size_out = (size_in + 2*pad - (kern - 1) - 1)/stride +1
    return size_out

def avg_size_out(size_in, kern, stride):
    pad = 0
    size_out = (size_in + 2*pad - kern)/stride +1
    return size_out

class conv1d:
    def __init__(self, numf):
        self.train_dataset = dummy(1,numf)
        self.numf = numf
    
    def create(self):
        print('\n\nThis is torch-conv1d \n\n')
        numf = self.numf
        
        out_channels = 6
        kern = 5
        stride = 1

        conv_out = conv_size_out(numf, kern, stride)
        lin_in = out_channels * ( int(conv_out) )    

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
        self.criterion = criterion()
        self.optimizer = optimizer(model.parameters(), lr = learning_rate)
        self.model = model

class conv2d:
    def __init__(self, numf):
        self.train_dataset = dummy(2,numf)
        self.numf = numf

    def create(self):
        print('This is torch-conv2d \n')
        numf = self.numf
        
        out_channels = 6
        kern = 5
        stride = 1

        conv_out = conv_size_out(numf, kern, stride)
        lin_in = out_channels * ( int(conv_out) ** 2 )    

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
        self.criterion = criterion()
        self.optimizer = optimizer(model.parameters(), lr = learning_rate)        
        self.model = model
        
mapp = {
#     'avg1d': avg1d,
#     'avg2d': avg2d,
    'conv1d': conv1d,
    'conv2d': conv2d,
#     'max1d': max1d,
#     'max2d': max2d,
#     'dense': dense
}