import torch 
import torch.nn as nn
import argparse

from tor_data import give
import lib_torch

import sys
sys.path.append('/home/ubuntu/profile')
import lib

parser = argparse.ArgumentParser()
parser = lib.arg_all(parser)
parser = lib.arg_conv(parser)
args = parser.parse_args()

DIM = args.dim
conv = nn.Conv1d if DIM==1 else nn.Conv2d    

layer = conv(in_channels = args.channels, out_channels = args.filters, kernel_size = args.kernel, stride = args.stride)

model = nn.Sequential(
    layer,
    nn.Flatten(),
    nn.Linear(
        in_features = args.filters * lib_torch.conv_size_out(args.numf, args.kernel, args.stride) ** DIM,
        out_features = 10
    )
)

train_dataset = give(DIM, args.numf, args.channels)

if args.nodes > 1:
    model, train_loader = lib_torch.distribute(model, train_dataset, args.nodes, args.batch)
else:
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.batch,
        shuffle = True
    )

time = lib_torch.profile(['conv{}d'.format(DIM), 
                          'MkldnnConvolutionBackward'], 
                         model, train_loader, args.epochs)

import numpy as np

data = np.array([[
    args.epochs, 9*512, # dataset size
    args.numf,
    args.channels,
    args.batch,
    args.nodes,
    args.kernel,
    args.stride,
    args.filters,
    time
]])
with open('conv{}d.ptorch'.format(DIM),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
    