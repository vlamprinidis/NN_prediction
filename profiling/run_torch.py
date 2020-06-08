import funs

args = funs.parse()

model_str = args.model
numf = args.num_features
batch = args.batch
rank = args.rank
nodes = args.nodes
it = args.iteration
epochs = args.epochs

import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms

import models_torch as m
from models_torch import dataset as ds

import funs_torch as f

# dict_ = { (<layer>,<numf>,<batch>,<nodes>,<it>) : <dataframe> }
mapp = {
#     'avg1d':mod.avg1d,
#     'avg2d':mod.avg2d,
    'conv1d':( m.conv1d, ds(dim=1) ),
    'conv2d':( m.conv2d, ds(dim=2) )
}

build_func, train_dataset = mapp[model_str]

model, train_loader = f.prepare(
    build_func = build_func,
    train_dataset = train_dataset,
    numf = numf,
    batch = batch,
    rank = rank,
    nodes = nodes
)

f.profile(
    model = model, 
    train_loader = train_loader, 
    epochs = epochs, 
    rank = rank, 
    criterion = m.criterion(), 
    optimizer = m.optimizer(model.parameters(), lr = m.learning_rate)
)