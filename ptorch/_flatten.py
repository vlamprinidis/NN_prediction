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
args = parser.parse_args()

DIM = args.dim

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(
        in_features = args.channels * args.numf ** DIM,
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

time = lib_torch.profile(['flatten'], 
                         model, train_loader, args.epochs)

import numpy as np

data = np.array([[
    args.epochs, tor_data.ds_size, # dataset size
    args.numf,
    args.channels,
    args.batch,
    args.nodes,
    time
]])
with open('flatten{}d.ptorch'.format(DIM),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
