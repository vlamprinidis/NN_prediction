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

import funs as h

args = h.parse( list( mapp.keys() ) )

model_str = args.model
numf = args.num_features
batch = args.batch
nodes = args.nodes
it = args.iteration
epochs = args.epochs
use_prof=args.use_profiler

build_func, train_dataset = mapp[model_str]

model, train_loader = f.prepare(build_func, train_dataset, numf, batch, nodes)

prof = f.profile(model, train_loader, epochs, 
                 m.criterion(), m.optimizer(model.parameters(), lr = 0.01), 
                 use_prof = use_prof)

if prof != None:
    df = f.get_ops(prof)
    
    key = h.my_key(model_str, numf, batch, nodes, it)
    value = h.my_value(df, epochs)
    
    target = './torch.pkl'
    h.update(key, value, target)

print('\n\n')