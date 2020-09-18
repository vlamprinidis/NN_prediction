import torch 
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms

from torch_models import LeNet5
import torch_lib

import tor_data

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

ds_size = 60000
train_dataset = tor_data.give(2,32,1, out_size=10, ds_size=ds_size)

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
