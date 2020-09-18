import lib
from lib import detuple

import torch
from torch import nn
from collections import OrderedDict

def extract(model, the_dim, 
                      the_batch, the_numf, the_channels):
    def MyName(name):
        name_map = {
            'Conv1d': 'conv1d', 'Conv2d':'conv2d',
            'AvgPool1d': 'avg1d', 'AvgPool2d': 'avg2d',
            'MaxPool1d': 'max1d', 'MaxPool2d': 'max2d',
            'BatchNorm1d': 'norm1d', 'BatchNorm2d': 'norm2d',
            'Dropout': 'drop1d', 'Dropout2d': 'drop2d',
            'ReLU': [None, 'relu1d', 'relu2d'],
            'Tanh': [None, 'tanh1d', 'tanh2d'],
            'Flatten': [None, 'flatten1d', 'flatten2d'],
            'Linear': 'dense'
        }
        if name not in name_map.keys():
            return name
        
        if name in ['ReLU', 'Tanh', 'Flatten']:
            return name_map[name][the_dim]
        
        return name_map[name]
    
    def MyAttr(attr, class_name):
        _MyAttr = {
            'stride':'stride',
            'p':'drop',
            'in_channels':'channels',
            'out_channels':'filters',
            'out_features':'units'
        }
        if class_name in ['AvgPool1d', 'AvgPool2d', 'MaxPool1d', 'MaxPool2d']:
                _MyAttr['kernel_size'] = 'pool'
        else:
            _MyAttr['kernel_size'] = 'kernel'
        
        return _MyAttr[attr]

    Search = {
        'Conv1d':['out_channels', 'kernel_size', 'stride'],
        'Conv2d':['out_channels', 'kernel_size', 'stride'],
        'AvgPool1d':['kernel_size', 'stride'],
        'AvgPool2d':['kernel_size', 'stride'],
        'MaxPool1d':['kernel_size', 'stride'],
        'MaxPool2d':['kernel_size', 'stride'],
        'Dropout':['p'],
        'Dropout2d':['p'],
        'Linear':['out_features']
    }
    
    def give_size_summary_dct(model, dim, batch, channels, numf):
        if dim==1:
            x = torch.rand(batch, channels, numf)

        else:
            x = torch.rand(batch, channels, numf, numf)

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % ('Layer', module_idx + 1)
                summary[m_key] = {'name': class_name, 'input_shape':list(input[0].size())}

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
            ):
                hooks.append(module.register_forward_hook(hook))

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(x)

        return summary
    
    summ_dct = give_size_summary_dct(model, the_dim, the_batch, the_channels, the_numf)

    def give_info(layer, inp_info):
        Info = {}
        name = layer.__class__.__name__
        Info['name'] = MyName(name)
        
        assert(name == inp_info['name'])
        
        Info['input_shape'] = inp_info['input_shape']
        Info['batch'] = inp_info['input_shape'][0]
        
        inp_size = len(inp_info['input_shape'])
        if inp_size == 2:
            Info['dim'] = 1
            Info['channels'] = 1
            Info['numf'] = inp_info['input_shape'][1]
        elif inp_size == 3:
            Info['dim'] = 1
            Info['channels'] = inp_info['input_shape'][1]
            Info['numf'] = inp_info['input_shape'][2]
        else:
            Info['dim'] = 2
            Info['channels'] = inp_info['input_shape'][1]
            Info['numf'] = inp_info['input_shape'][2]
        
        if name in Search.keys():
            search = Search[name]
            for attr in search:
                Info[MyAttr(attr, name)] = detuple(layer.__dict__[attr])
        
        return Info

    return [give_info(layer, inp_info) for layer, inp_info in zip(model, summ_dct.values())]

    