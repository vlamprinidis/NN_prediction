import torch 
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms

from torch_models import LeNet5
import torch_lib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-nodes', type = int, required = True)
parser.add_argument('-batch', type = int, required = True)
parser.add_argument('-epochs', type = int, required = True)
args = parser.parse_args()

model = LeNet5().model
# transforms = transforms.Compose([transforms.Resize((32, 32)),
#                                  transforms.ToTensor()])
# train_dataset = datasets.MNIST(root='mnist_data', 
#                                train=True, 
#                                transform=transforms,
#                                download=True)

from numpy.random import RandomState as R

seed = 42

def give(dim, n, channels):
    ds_size = 60000
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

train_dataset = give(2,32,1)

if args.nodes > 1:
    model, train_loader = torch_lib.distribute(model, train_dataset, args.nodes, args.batch)
else:
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.batch,
        shuffle = True
    )

the_time = torch_lib.profile(model, train_loader, args.epochs)

import socket
host = socket.gethostname()

print()
print('Host: ', host,', Time : ', the_time/1000/1000, 's')
print()
