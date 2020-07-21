# before running this script, run:
# tensorboard --logdir /home/ubuntu/logs_tflow --bind_all
import tensorflow as tf
import numpy as np
import os
import sys

import models_tflow as m
import funs_tflow as f
import funs as h

args = h.prof_parse()

model_str = args.model
numf = args.num_features
hp = args.hyper_param
batch = args.batch
nodes = args.nodes
it = args.iteration
epochs = args.epochs

model_class = m.mapp[model_str](numf, hp)

f.prepare(model_class, nodes)

prof = f.profile(model_class, batch, epochs)

if prof != None:
    df = f.get_ops(prof)
    
    key = h.my_key(model_str, numf, hp, batch, nodes, it)
    value = h.my_value(df, epochs)
    
    target = 'results/tflow.pkl'
    h.update(key, value, target)

print('\n\n')
