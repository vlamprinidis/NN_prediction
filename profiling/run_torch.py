import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms

import models_torch as m
import funs_torch as f
import funs as h

import argparse

parser = argparse.ArgumentParser()
args = h.insert_prof_args(parser).parse_args()

model_str = args.model
numf = args.num_features
hp = args.hyper_param
batch = args.batch
nodes = args.nodes
epochs = args.epochs

model_class = m.mapp[model_str](numf, hp)

f.prepare(model_class, batch, nodes)

prof = f.profile(model_class, epochs)

if prof != None:
    df = f.get_ops(prof)
    
    key = h.my_key(model_str, numf, hp, batch, nodes)
    value = df
    
    h.update(key, value, 'data.torch')

print('\n\n')
