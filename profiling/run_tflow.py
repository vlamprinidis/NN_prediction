# before running this script, run:
# tensorboard --logdir /home/ubuntu/logs_tflow --bind_all
import tensorflow as tf
import numpy as np
import os
import sys

import models_tflow as m
import funs_tflow as f
import funs as h

args = h.parse( list( m.mapp.keys() ) )

model_str = args.model
numf = args.num_features
batch = args.batch
nodes = args.nodes
it = args.iteration
epochs = args.epochs
use_prof=args.use_profiler

model_class = m.mapp[model_str](numf)

f.distribute(model_class, nodes)

prof = f.profile(model_class, batch, epochs, 
                 use_prof = h.rank == 0 and use_prof)

if prof != None:
    df = f.get_ops(prof)
    
    key = h.my_key(model_str, numf, batch, nodes, it)
    value = h.my_value(df, epochs)
    
    target = './tflow.pkl'
    h.update(key, value, target)

print('\n\n')
