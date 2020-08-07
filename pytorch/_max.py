import torch 
import torch.nn as nn
import argparse

from tor_data import give
import funs_torch
import funs

parser = argparse.ArgumentParser()
parser = funs.arg_all(parser)
parser = funs.arg_pool(parser)
args = parser.parse_args()

DIM = args.dim
RESULT = '__max{}d.torch'.format(DIM)
max_pool = nn.MaxPool1d if DIM==1 else nn.MaxPool2d    

layer = max_pool(kernel_size = args.pool, stride = args.stride)

model = nn.Sequential(
    layer,
    nn.Flatten(),
    nn.Linear(
        in_features = args.channels * funs_torch.max_size_out(args.numf, args.pool, args.stride) ** DIM,
        out_features = 10
    )
)

train_dataset = give(DIM, args.numf, args.channels)

if args.nodes > 1:
    model, train_loader = funs_torch.distribute(model, train_dataset, args.nodes, batch)
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
        'pool':args.pool,
        'stride':args.stride
    })
    value = funs_torch.get_ops(prof)
    
    funs.update(key, value, RESULT)
