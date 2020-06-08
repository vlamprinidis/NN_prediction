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

import funs as h

args = h.parse( list( mapp.keys() ) )

model_str = args.model
numf = args.num_features
batch = args.batch
nodes = args.nodes
it = args.iteration
epochs = args.epochs

build_func, (x,y) = mapp[model_str]

model = f.prepare(build_func, x, y, numf, nodes)

f.profile(model, x, y, batch, epochs, nodes, use_prof = h.rank == 0)