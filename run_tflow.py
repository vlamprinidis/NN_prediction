# before running this script, run: <./run_tf.sh>
import tensorflow as tf
import numpy as np
import os
import sys
import layers_tflow
import funs_tflow
import funs
import argparse

parser = argparse.ArgumentParser()
args = h.insert_prof_args(parser).parse_args()

hp = {
    'filters': args.filters
    'kernel': args.kernel
    'stride': args.stride
    'drop': args.drop
}

Layer = layers_tflow.mapp[args.layer](numf=args.numf, channels=args.channels, hp=hp)

funs_tflow.prepare(Layer, args.nodes)

prof = funs_tflow.profile(Layer, args.batch, args.epochs)

if prof != None:
    df = funs_tflow.get_ops(prof)
    
    key = funs.my_key()
    value = df
    
    funs.update(key, value, 'data.tflow')

print('\n\n')
