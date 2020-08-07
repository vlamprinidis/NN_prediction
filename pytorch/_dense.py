import torch 
import torch.nn as nn
import argparse

from tor_data import give
import funs_torch
import funs

parser = argparse.ArgumentParser()
parser = funs.arg_all(parser)
parser.add_argument('-units', type = int, required = True)
args = parser.parse_args()

DIM = args.dim
RESULT = '__dense{}d.torch'.format(DIM)

layer = nn.Linear(
    in_features = args.channels * args.numf ** DIM,
    out_features = args.units
)

model = nn.Sequential(
    nn.Flatten(),
    layer,
    nn.Flatten(),
    nn.Linear(
        in_features = args.units,
        out_features = 10
    )
)

train_dataset = give(DIM, args.numf, args.channels)

if args.nodes > 1:
    model, train_loader = funs_torch.distribute(model, train_dataset, args.nodes, args.batch)
else:
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.batch,
        shuffle = True
    )

prof = funs_torch.profile(model, train_loader, args.epochs)

if prof != None:
    key = funs.my_key({
        'numf':args.numf,
        'batch':args.batch,
        'nodes':args.nodes,
        'channels':args.channels,
        'units':args.units,
    })
    value = funs_torch.get_ops(prof)
    
    funs.update(key, value, RESULT)
