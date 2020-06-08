import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

def dataset2d():
    trans = torchvision.transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='../../torch_data/',
        train=True, 
        transform=trans, 
        download=True
    )
    return train_dataset

def dataset1d():
    trans = torchvision.transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(1,-1))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='../../torch_data/',
        train=True, 
        transform=trans, 
        download=True
    )
    return train_dataset

def conv_size_out(size_in, kern, stride):
    pad = 0
    size_out = (size_in + 2*pad - (kern - 1) - 1)/stride +1
    return size_out

def avg_size_out(size_in, kern, stride):
    pad = 0
    size_out = (size_in + 2*pad - kern)/stride +1
    return size_out

def conv1d(numf):
    train_dataset = dataset1d()
    
    conv_out = conv_size_out(32*32, 5, 1)
    lin_in = numf * ( int(conv_out) )    
    
    model = nn.Sequential(
          nn.Conv1d(
              in_channels = 1, out_channels = numf,
              kernel_size = 5, stride = 1
          ),
          nn.Flatten(),
          nn.Linear(
            in_features = lin_in,
            out_features = 10
        )
    )
    
    return model, train_dataset

def conv2d(numf):
    train_dataset = dataset()
    
    conv_out = conv_size_out(32, 5, 1)
    lin_in = numf * ( int(conv_out) ** 2 )    
    
    model = nn.Sequential(
          nn.Conv2d(
              in_channels = 1, out_channels = numf,
              kernel_size = 5, stride = 1
          ),
          nn.Flatten(),
          nn.Linear(
            in_features = lin_in,
            out_features = 10
        )
    )
    
    return model, train_dataset

            
            
            