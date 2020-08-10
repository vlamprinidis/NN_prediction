from funs import load, numf_ls, batch_ls, clean_go

# def big_loop(framework, timeout): #times = list of seconds for timeout
#     frame_to_load = {'tflow': load('data.tflow'), 'torch': load('data.torch')}
#     for nodes in [3,2,1]:
#         for model in list(hp_map):
#             for hp in hp_map[model]:
#                 for numf in numf_ls:
#                     for batch in batch_ls:
                        
#                         print('Success') if success else print('Failure')

TF = '/home/vlassis/.env/bin/python3 /home/vlassis/simple/tensorflow/{file} {p1} {p2}'.format
PT = '/home/vlassis/.env/bin/python3 /home/vlassis/simple/pytorch/{file} {p1} {p2}'.format

opt = {
    'all': '-numf {numf} -batch {b} -nodes {nodes} -epochs {e} -channels {ch} -dim {dim}'.format,
    'conv': '-kern {kernel} -filters {filters} -stride {stride}'.format,
    'pool': '-pool {pool} -stride {s}'.format,
    'dense': '-units {units}'.format,
    'drop': '-drop {drop}'.format
}

files = ['_alone.py', '_avg.py', '_conv.py', '_dense.py', '_drop.py', '_max.py', '_norm.py', '_relu.py']

NODES = 1
#TF
cmd = TF(file = '_conv.py', 
         p1 = opt['all'](numf=32, b=32, nodes=NODES, e=5, ch=1, dim=1),
         p2 = opt['conv'](kernel=4, filters=1, stride=1))

clean_go(cmd,NODES)