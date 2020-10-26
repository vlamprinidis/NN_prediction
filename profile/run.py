tf_files = ['_avg.py', '_conv.py', '_dense.py', '_drop.py', '_max.py', '_norm.py', '_relu.py', '_tanh.py']
pt_files = tf_files + ['_flatten.py']

import argparse
from lib import clean_go

parser = argparse.ArgumentParser()
parser.add_argument('-f', type = str, choices=['tflow', 'ptorch'], required = True)
args = parser.parse_args()

frames = {
    'tflow':'/home/ubuntu/.env/bin/python3 /home/ubuntu/profile/tflow/{file} {p1} {p2}'.format,
    'ptorch':'/home/ubuntu/.env/bin/python3 /home/ubuntu/profile/ptorch/{file} {p1} {p2}'.format
}
FRAME = frames[args.f]

opt = {
    'all': '-numf {numf} -batch {b} -nodes {nodes} -epochs {e} -channels {ch} -dim {dim}'.format,
    'conv': '-kern {kernel} -filters {filters} -stride {stride}'.format,
    'pool': '-pool {pool} -stride {stride}'.format,
    'dense': '-numf {numf} -batch {b} -nodes {nodes} -epochs {e} -units {units}'.format,
    'drop': '-drop {drop}'.format
}

dim = 2
epochs = 5

batch_ls = [16,32,64,128,256,512]
numf_ls = [16,32,64]
channels_ls = [1,2,4,8,16]

for nodes in [3,2,1]:
    for numf in numf_ls:
        for batch in batch_ls:
            for channels in channels_ls:
                opt_all = opt['all'](numf=numf, b=batch, nodes=nodes, e=epochs, ch=channels, dim=dim)

                # Conv 
                for kernel in [2,4,8,16]:
                    for filters in [1,2,4,8,16]:
                        for stride in [1,2,4]:
                            cmd = FRAME(file = '_conv.py', 
                                 p1 = opt_all,
                                 p2 = opt['conv'](kernel=kernel, filters=filters, stride=stride))
                            clean_go(cmd, nodes)


for nodes in [3,2,1]:
    for numf in numf_ls:
        for batch in batch_ls:
            for channels in channels_ls:
                opt_all = opt['all'](numf=numf, b=batch, nodes=nodes, e=epochs, ch=channels, dim=dim)

                # Pool 
                for file in ['_avg.py', '_max.py']:
                    for pool in [2,4,8]:
                        for stride in [1,2,4]:
                            cmd = FRAME(file = file, 
                                 p1 = opt_all,
                                 p2 = opt['pool'](pool=pool, stride=stride))
                            clean_go(cmd, nodes)


for nodes in [3,2,1]:
    for numf in numf_ls:
        for batch in batch_ls:
            for channels in channels_ls:
                opt_all = opt['all'](numf=numf, b=batch, nodes=nodes, e=epochs, ch=channels, dim=dim)

                # Dropout 
                for drop in [0.2, 0.4, 0.8]:
                    cmd = FRAME(file = '_drop.py',
                                p1 = opt_all,
                                p2 = opt['drop'](drop = drop))
                    clean_go(cmd, nodes)


myfiles = ['_norm.py', '_relu.py', '_tanh.py']
if args.f == 'ptorch':
    myfiles += ['_flatten.py']
    
for nodes in [3,2,1]:
    for numf in numf_ls:
        for batch in batch_ls:
            for channels in channels_ls:
                opt_all = opt['all'](numf=numf, b=batch, nodes=nodes, e=epochs, ch=channels, dim=dim)
                
                # Batch Normalization, Relu, Tanh
                for file in myfiles:
                    cmd = FRAME(file = file,
                                p1 = opt_all,
                                p2 = '')
                    clean_go(cmd, nodes)

dense_numf = [16,32,64,128,256,512,1024,4096]
dense_units = [8,16,32,64,128,256,512,1024]

for nodes in [3,2,1]:
    for numf in dense_numf:
        for batch in batch_ls:
            # Dense
            for units in dense_units:
                cmd = FRAME(file = '_dense.py',
                            p1 = opt['dense'](numf=numf, b=batch, nodes=nodes, e=epochs, units = units),
                            p2 = '')
                clean_go(cmd, nodes)

for nodes in [3,2,1]:
    for numf in dense_numf:
        for batch in batch_ls:
            # Dense
            for units in dense_units:
                cmd = FRAME(file = '_final_dense.py',
                            p1 = opt['dense'](numf=numf, b=batch, nodes=nodes, e=epochs, units = units),
                            p2 = '')
                clean_go(cmd, nodes)

