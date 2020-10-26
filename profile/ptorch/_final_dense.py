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

parser.add_argument('-numf', type = int, required = True)
parser.add_argument('-batch', type = int, required = True)
parser.add_argument('-nodes', type = int, required = True)
parser.add_argument('-epochs', type = int, required = True)

parser.add_argument('-units', type = int, required = True)
args = parser.parse_args()

layer = nn.Linear(
    in_features = args.numf,
    out_features = args.units
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(
        in_features = args.numf,
        out_features = args.units
    )
)

train_dataset = give(1, args.numf, 1, out_size = args.units)

if args.nodes > 1:
    model, train_loader = lib_torch.distribute(model, train_dataset, args.nodes, args.batch)
else:
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.batch,
        shuffle = True
    )

time = lib_torch.profile(['addmm', 'AddmmBackward'], 
                         model, train_loader, args.epochs)

import numpy as np

data = np.array([[
    args.epochs, tor_data.ds_size, # dataset size
    args.numf,
    args.batch,
    args.nodes,
    args.units,
    time
]])
with open('final_dense.ptorch','a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
