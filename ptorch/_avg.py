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
parser = lib.arg_pool(parser)
args = parser.parse_args()

DIM = args.dim
avg_pool = nn.AvgPool1d if DIM==1 else nn.AvgPool2d    

layer = avg_pool(kernel_size = args.pool, stride = args.stride)

model = nn.Sequential(
    layer,
    nn.Flatten(),
    nn.Linear(
        in_features = args.channels * lib_torch.avg_size_out(args.numf, args.pool, args.stride) ** DIM,
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

time = lib_torch.profile(['avg_pool{}d'.format(DIM)], 
                         model, train_loader, args.epochs)

import numpy as np

data = np.array([[
    args.numf,
    args.channels,
    args.batch,
    args.nodes,
    args.pool,
    args.stride,
    time
]])
with open('avg{}d.ptorch'.format(DIM),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
