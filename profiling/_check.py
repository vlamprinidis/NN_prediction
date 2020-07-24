import subprocess as sub
from funs import numf_ls, batch_ls, nodes_ls, get_keys, hp_map

hps = sum([ len(hp_map[key]) for key in hp_map ])

curr_tflow = len(get_keys('tflow.pkl'))
total_tflow = len(numf_ls)*len(batch_ls)*len(nodes_ls)*hps
print( 'Tensorflow: {} %'.format( 100*curr_tflow/total_tflow ) )

curr_torch = len(get_keys('torch.pkl'))
total_torch = len(numf_ls)*len(batch_ls)*len(nodes_ls)*hps
print( 'PyTorch: {} %'.format( 100*curr_torch/total_torch ) )
