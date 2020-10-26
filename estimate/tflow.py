from lib import detuple

def extract(model, the_dim, 
                      the_batch, the_numf, the_channels):
    def MyName(name):
        name_map = {
            'Conv1D': 'conv1d', 'Conv2D':'conv2d',
            'AveragePooling1D': 'avg1d', 'AveragePooling2D': 'avg2d',
            'MaxPooling1D': 'max1d', 'MaxPooling2D': 'max2d',
            'BatchNormalization': [None, 'norm1d', 'norm2d'],
            'Dropout': [None, 'drop1d', 'drop2d'],
            'ReLU': [None, 'relu1d', 'relu2d'],
            'tanh': [None, 'tanh1d', 'tanh2d'],
            'Dense': 'dense'
        }
        if name not in name_map.keys():
            return name
        
        if name in ['BatchNormalization', 'Dropout', 'ReLU', 'tanh']:
            return name_map[name][the_dim]
        
        return name_map[name]
    
    MyAttr = {
        'strides':'stride',
        'pool_size':'pool',
        'filters':'filters',
        'kernel_size':'kernel',
        'units':'units',
        'rate':'drop'
    }

    Search = {
        'Conv1D':['filters', 'kernel_size', 'strides'],
        'Conv2D':['filters', 'kernel_size', 'strides'],
        'AveragePooling1D':['pool_size', 'strides'],
        'AveragePooling2D':['pool_size', 'strides'],
        'MaxPooling1D':['pool_size', 'strides'],
        'MaxPooling2D':['pool_size', 'strides'],
        'Dropout':['rate'],
        'Dense':['units']
    }

    if the_dim==1:
        model.build(input_shape = (the_batch, the_numf, 
                                   the_channels))
    else:
        model.build(input_shape = (the_batch, the_numf, 
                                   the_numf, the_channels))
    
    def give_info(layer):
        Info = {}
        name = layer.__class__.__name__
        
        if name == 'Activation':
            Info['name'] = MyName(layer.__dict__['activation'].__name__)
        else:
            Info['name'] = MyName(name)
        
        if name == 'Add':
            return Info
        
        inp = layer.input.shape
        
        Info['batch'] = inp[0]
        Info['numf'] = inp[1]
        
        if len(inp) == 2:
            Info['dim'] = 1
            Info['channels'] = 1
        elif len(inp) == 3:
            Info['dim'] = 1
            Info['channels'] = inp[2]
        else:
            Info['dim'] = 2
            Info['channels'] = inp[3]
            
        Info['input_shape'] = layer.input.shape
        Info['output_shape'] = layer.output.shape
        
        if name in Search.keys():
            search = Search[name]
            for attr in search:                
                Info[MyAttr[attr]] = detuple(layer.__dict__[attr])
        
        return Info
    
    layers = [give_info(layer) for layer in model.layers]
    
    assert layers[-1]['name'] == 'dense'
    layers[-1]['name'] = 'final_dense'
    
    return layers