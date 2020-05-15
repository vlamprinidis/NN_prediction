#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
pth = '/home/ubuntu/diploma'
if(pth not in sys.path):
    sys.path.insert(0, pth)

    
from funs.parse import parse

info = parse(sys.argv)

fw = info['FW']
model = info['MODEL']
batch = info['BATCH']
rank = info['RANK']
nodes = info['NODES']

from funs.models import give_tf_model as tflow

net, x_train, y_train = tflow(model, batch, rank, nodes)

import funs.graph as g

if rank == 0:
    import funs.data as fdt
    
    pred = g.tf(net, x_train, y_train, batch)
    
    fdt.insert_df(file = 'pred_data.pkl', name = 'pred_tflow_{model}'.format(model=model),
               batch = batch, nodes = nodes, df = pred)
    
else:
    g.just_tf(net, x_train, y_train, batch)