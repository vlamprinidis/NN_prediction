from funs import load, numf_ls, batch_ls, clean_go

# def big_loop(framework, timeout): #times = list of seconds for timeout
#     frame_to_load = {'tflow': load('data.tflow'), 'torch': load('data.torch')}
#     for nodes in [3,2,1]:
#         for model in list(hp_map):
#             for hp in hp_map[model]:
#                 for numf in numf_ls:
#                     for batch in batch_ls:
                        
#                         print('Success') if success else print('Failure')

TF = '/home/ubuntu/.env/bin/python3 /home/ubuntu/simple/tensorflow/{file} {p1} {p2}'.format
PT = '/home/ubuntu/.env/bin/python3 /home/ubuntu/simple/pytorch/{file} {p1} {p2}'.format

opt = {
    'all': '-numf {nf} -batch {b} -nodes {nd} -epochs {e} -channels {ch} -dim {dm}'.format,
    'conv': '-kern {k} -filters {f} -stride {s}'.format,
    'pool': '-pool {p} -stride {s}'.format,
    'dense': '-units {u}'.format,
    'drop': '-drop {dr}'.format
}

files = ['_alone.py', '_avg.py', '_conv.py', '_dense.py', '_drop.py', '_max.py', '_norm.py', '_relu.py']

#TF
cmd = TF(file = '_conv.py', 
         p1 = opt['all'](nf=32, b=32, nd=1, e=5, ch=1, dm=1),
         p2 = opt['conv'](k=4, f=1, s=1))

clean_go(cmd,1)