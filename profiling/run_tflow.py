import funs

args = funs.parse()

model_str = args.model
numf = args.num_features
batch = args.batch
rank = args.rank
nodes = args.nodes
it = args.iteration
epochs = args.epochs

# before running this script run:
# tensorboard --logdir /home/ubuntu/logs_tflow --bind_all
import tensorflow as tf
import numpy as np
import os
import sys

import models_tflow as m
import funs_tflow as f

# dict_ = { (<layer>,<numf>,<batch>,<nodes>,<it>) : <dataframe> }

mapp = {
    'avg1d':( m.avg1d, m.dataset(dim=1) ),
    'avg2d':(m.avg2d, m.dataset(dim=2)),
    'conv1d':(m.conv1d, m.dataset(dim=1)),
    'conv2d':(m.conv2d, m.dataset(dim=2))
}

build_func, (x,y) = mapp[model_str]

model = f.prepare(
    build_func = build_func,
    x = x, y = y,
    numf = numf, 
    rank = rank, 
    nodes = nodes
)

f.profile(
    model = model, 
    x = x, y = y,
    batch = batch, 
    epochs = epochs, 
    rank = rank, 
    nodes = nodes
)