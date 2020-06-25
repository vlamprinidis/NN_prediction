import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms

import models_torch as m
import funs_torch as f
import funs as h

args = h.parse( list( m.mapp.keys() ) )

model_str = args.model
numf = args.num_features
batch = args.batch
nodes = args.nodes
it = args.iteration
epochs = args.epochs
use_prof = args.use_profiler

model_class = m.mapp[model_str](numf)

f.prepare(model_class, batch, nodes)

prof = f.profile(model_class, epochs, use_prof = use_prof)

if prof != None:
    df = f.get_ops(prof)
    
    key = h.my_key(model_str, numf, batch, nodes, it)
    value = h.my_value(df, epochs)
    
    target = './torch.pkl'
    h.update(key, value, target)

print('\n\n')