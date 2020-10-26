import torch
from numpy.random import RandomState as R

seed = 42
ds_size = 9*512*2

def give(dim, n, channels, out_size = 10):
    if dim == 1:
        x = R(seed).random((ds_size, channels, n))
        x = x.reshape(x.shape[0], channels, n)
    else:
        x = R(seed).random((ds_size, channels, n, n))
        x = x.reshape(x.shape[0], channels, n, n)
    
    y = R(seed).randint(0,out_size,ds_size)
    
    x = torch.from_numpy(x).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)

    train_data = torch.utils.data.TensorDataset(x, y)
    
    return train_data
