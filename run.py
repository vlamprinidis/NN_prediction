from funs import load, numf_ls, batch_ls, clean_go

# def big_loop(framework, timeout): #times = list of seconds for timeout
#     frame_to_load = {'tflow': load('data.tflow'), 'torch': load('data.torch')}
#     for nodes in [3,2,1]:
#         for model in list(hp_map):
#             for hp in hp_map[model]:
#                 for numf in numf_ls:
#                     for batch in batch_ls:
                        
#                         print('Success') if success else print('Failure')

TF = '/home/vlassis/.env/bin/python3 /home/vlassis/prof_cloud/tensorflow/{file} {p1} {p2}'.format
PT = '/home/vlassis/.env/bin/python3 /home/vlassis/prof_cloud/pytorch/{file} {p1} {p2}'.format

opt = {
    'all': '-numf {numf} -batch {b} -nodes {nodes} -epochs {e} -channels {ch} -dim {dim}'.format,
    'conv': '-kern {kernel} -filters {filters} -stride {stride}'.format,
    'pool': '-pool {pool} -stride {stride}'.format,
    'dense': '-units {units}'.format,
    'drop': '-drop {drop}'.format
}

files = ['_alone.py', '_avg.py', '_conv.py', '_dense.py', '_drop.py', '_max.py', '_norm.py', '_relu.py']

# NODES = 3
# #TF
# cmd = TF(file = '_conv.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = opt['conv'](kernel=4, filters=1, stride=1))

# clean_go(cmd,NODES)

# #PT
# cmd = PT(file = '_conv.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = opt['conv'](kernel=4, filters=1, stride=1))

# clean_go(cmd,NODES)

# NODES = 3
# #TF
# cmd = TF(file = '_max.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = opt['pool'](pool=4, stride=1))

# clean_go(cmd,NODES)

# NODES = 3
# #TF
# cmd = TF(file = '_alone.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = '')

# clean_go(cmd,NODES)
         
# NODES = 3
# #PT
# cmd = PT(file = '_alone.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = '')

# clean_go(cmd,NODES)

# NODES = 3
# #TF
# cmd = TF(file = '_norm.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = '')

# clean_go(cmd,NODES)
         
# NODES = 3
# #PT
# cmd = PT(file = '_norm.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = '')

# clean_go(cmd,NODES)

# NODES = 3
# #TF
# cmd = TF(file = '_relu.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = '')

# clean_go(cmd,NODES)
         
# NODES = 3
# #PT
# cmd = PT(file = '_relu.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = '')

# clean_go(cmd,NODES)

# NODES = 3
# #TF
# cmd = TF(file = '_drop.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = opt['drop'](drop=0.2))

# clean_go(cmd,NODES)

# NODES = 3
# #PT
# cmd = PT(file = '_drop.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
#          p2 = opt['drop'](drop=0.2))

# clean_go(cmd,NODES)

# NODES = 3
# #TF
# cmd = TF(file = '_dense.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=2),
#          p2 = opt['dense'](units=10))

# clean_go(cmd,NODES)

# NODES = 3
# #PT
# cmd = PT(file = '_dense.py', 
#          p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=2),
#          p2 = opt['dense'](units=10))

# clean_go(cmd,NODES)

NODES = 3
#TF
cmd = TF(file = '_tanh.py', 
         p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
         p2 = '')

clean_go(cmd,NODES)
         
NODES = 3
#PT
cmd = PT(file = '_tanh.py', 
         p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
         p2 = '')

clean_go(cmd,NODES)