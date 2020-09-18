import torch 
import torch.nn as nn
import argparse

from tor_data import give
import tor_data

import lib_torch

import sys
sys.path.append('/home/ubuntu/profile')
import lib

parser = argparse.ArgumentParser()
parser = lib.arg_all(parser)
parser = lib.arg_pool(parser)
args = parser.parse_args()

DIM = args.dim
max_pool = nn.MaxPool1d if DIM==1 else nn.MaxPool2d    

layer = max_pool(kernel_size = args.pool, stride = args.stride)

model = nn.Sequential(
    layer,
    nn.Flatten(),
    nn.Linear(
        in_features = args.channels * lib_torch.max_size_out(args.numf, args.pool, args.stride) ** DIM,
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

time = lib_torch.profile(['max_pool{}d'.format(DIM)], 
                         model, train_loader, args.epochs)

import numpy as np

data = np.array([[
    args.epochs, tor_data.ds_size, # dataset size
    args.numf,
    args.channels,
    args.batch,
    args.nodes,
    args.pool,
    args.stride,
    time
]])
with open('max{}d.ptorch'.format(DIM),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
