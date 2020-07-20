import subprocess as sub
from funs import numf_ls, batch_ls, nodes_ls, get_keys
from models_tflow import model_ls as tflow_models
from models_torch import model_ls as torch_models

curr_tflow = len(get_keys('./tflow.pkl'))
total_tflow = len(numf_ls)*len(batch_ls)*len(nodes_ls)*len(tflow_models)
print( 'Tensorflow: {} %'.format( 100*curr_tflow/total_tflow ) )

curr_torch = len(get_keys('./torch.pkl'))
total_torch = len(numf_ls)*len(batch_ls)*len(nodes_ls)*len(torch_models)
print( 'PyTorch: {} %'.format( 100*curr_torch/total_torch ) )
