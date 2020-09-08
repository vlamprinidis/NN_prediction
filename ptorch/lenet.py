import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import LeNet5
import torch_lib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-nodes', type = int, required = True)
parser.add_argument('-batch', type = int, required = True)
parser.add_argument('-epochs', type = int, required = True)
args = parser.parse_args()

model = LeNet5()
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

if args.nodes > 1:
    model, train_loader = torch_lib.distribute(model, train_dataset, args.nodes, args.batch)
else:
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.batch,
        shuffle = True
    )

torch_lib.train(model, train_loader, args.epochs)
