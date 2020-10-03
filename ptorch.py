import torch 
import torch.nn as nn
import torch.nn.functional as F

import ptorch_models
import ptorch_lib

import pt_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-model', type = str, required = True)
parser.add_argument('-ds', type = int, required = True)
parser.add_argument('-numf', type = int, required = True)
parser.add_argument('-ch', type = int, required = True)
parser.add_argument('-out', type = int, required = True)

parser.add_argument('-nodes', type = int, required = True)
parser.add_argument('-batch', type = int, required = True)
parser.add_argument('-epochs', type = int, required = True)
args = parser.parse_args()

print('model:',args.model,'dataset size:',args.ds,'numf:',args.numf,'channels:',args.ch,'out:',args.out)
        
Model = getattr(ptorch_models, args.model)()

model = Model.create()

ds_size = args.ds
train_dataset = pt_data.give(2,args.numf, args.ch, out_size=args.out, ds_size=ds_size)

if args.nodes > 1:
    model, train_loader = ptorch_lib.distribute(model, train_dataset, args.nodes, args.batch)
else:
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.batch,
        shuffle = True
    )

the_time = ptorch_lib.profile(model, train_loader, args.epochs)

import socket
host = socket.gethostname()

import numpy as np

data = np.array([[
    args.epochs, ds_size, # dataset size
    args.numf,
    args.ch,
    args.batch,
    args.nodes,
    the_time
]])
with open('{}.ptorch'.format(args.model),'a') as file:
    np.savetxt(file, data, delimiter=",", fmt="%s")
