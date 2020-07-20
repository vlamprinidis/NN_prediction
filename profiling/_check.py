import subprocess as sub
from funs import numf_ls, batch_ls, nodes_ls, get_keys

iters=4
hps=5*2+3*2+3*2+5+1+1

curr_tflow = len(get_keys('results/tflow.pkl'))
total_tflow = len(numf_ls)*len(batch_ls)*len(nodes_ls)*iters*hps
print( 'Tensorflow: {} %'.format( 100*curr_tflow/total_tflow ) )

curr_torch = len(get_keys('results/torch.pkl'))
total_torch = len(numf_ls)*len(batch_ls)*len(nodes_ls)*iters*hps
print( 'PyTorch: {} %'.format( 100*curr_torch/total_torch ) )
