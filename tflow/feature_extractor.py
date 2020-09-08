def extract(model, the_dim, 
                      the_batch, the_numf, the_channels):    
    MyAttr = {
        'strides':'stride',
        'pool_size':'pool',
        'filters':'filters',
        'kernel_size':'kernel',
        'units':'units',
        'rate':'drop'
    }

    Search = {
        'Conv1D':['filters', 'kernel_size'],
        'Conv2D':['filters', 'kernel_size'],
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
            Info['name'] = layer.__dict__['activation'].__name__
        else:
            Info['name'] = name
        
        inp = layer.input.shape
        
        Info['batch'] = inp[0]
        Info['numf'] = inp[1]
        
        if len(inp) == 2:
            Info['dim'] = 0
            #Info['channels'] = 1
        elif len(inp) == 3:
            Info['dim'] = 1
            Info['channels'] = inp[2]
        else:
            Info['dim'] = 2
            Info['channels'] = inp[3]
            
        Info['input_shape'] = layer.input.shape
        
        if name in Search.keys():
            search = Search[name]
            for attr in search:                
                Info[MyAttr[attr]] = layer.__dict__[attr]
        
        return Info
    
    return [give_info(layer) for layer in model.layers]