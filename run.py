from funs import load, numf_ls, batch_ls, clean_go

# frame_to_load = {'tflow': load('data.tflow'), 'torch': load('data.torch')}

TF = '/home/ubuntu/.env/bin/python3 /home/ubuntu/vms/tensorflow/{file} {p1} {p2}'.format
PT = '/home/ubuntu/.env/bin/python3 /home/ubuntu/vms/pytorch/{file} {p1} {p2}'.format

opt = {
    'all': '-numf {numf} -batch {b} -nodes {nodes} -epochs {e} -channels {ch} -dim {dim}'.format,
    'conv': '-kern {kernel} -filters {filters} -stride {stride}'.format,
    'pool': '-pool {pool} -stride {stride}'.format,
    'dense': '-units {units}'.format,
    'drop': '-drop {drop}'.format
}

files = ['_alone.py', '_avg.py', '_conv.py', '_dense.py', '_drop.py', '_max.py', '_norm.py', '_relu.py', '_tanh.py']

epochs = 5
for nodes in [3,2,1]:
    for numf in numf_ls:
        for batch in batch_ls:
            for channels in [1,3]:
                for dim in [1,2]:
                    for FRAME in [TF, PT]:
                        
                        opt_all = opt['all'](numf=numf, b=batch, nodes=nodes, e=epochs, ch=channels, dim=dim)
                        
                        # Conv
                        for kernel in [2,4,8]:
                            for filters in [1,2,4,8,16]:
                                for stride in [1,2,4]:
                                    cmd = FRAME(file = '_conv.py', 
                                         p1 = opt_all,
                                         p2 = opt['conv'](kernel=kernel, filters=filters, stride=stride))
                                    clean_go(cmd, nodes)
                        
                        # Pool
                        for file in ['_avg.py', '_max.py']:
                            for pool in [2,4,8]:
                                for stride in [1,2,3]:
                                    cmd = FRAME(file = file, 
                                         p1 = opt_all,
                                         p2 = opt['pool'](pool=pool, stride=stride))
                                    clean_go(cmd, nodes)
                                    
                        # Dense
                        for units in [16, 32, 64, 128]:
                            cmd = FRAME(file = '_dense.py',
                                        p1 = opt_all,
                                        p2 = opt['dense'](units = units))
                            clean_go(cmd, nodes)
                            
                        # Dropout
                        for drop in [0.2, 0.4, 0.8]:
                            cmd = FRAME(file = '_drop.py',
                                        p1 = opt_all,
                                        p2 = opt['drop'](drop = drop))
                            clean_go(cmd, nodes)
                            
                        # Batch Normalization, Relu, Tanh, Alone
                        for file in ['_norm.py', '_relu.py', '_tanh.py', '_alone.py']:
                            cmd = FRAME(file = file,
                                        p1 = opt_all,
                                        p2 = '')
                            clean_go(cmd, nodes)
