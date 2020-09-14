import data_parser as dp
import numpy as np

from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def in_order(key, words):
    return [key[word] for word in words]

def open_layer(dct, params):    
    x = []
    y = []
    for key,value in dct.items():
        x.append(in_order(dp.from_key(key),params))
        y.append(value)
    
    return np.array(x), np.array(y)

def tflow_open_layer(name, params):    
    dct = dp.load('database/tflow/{}.tflow_db'.format(name))
    
    return open_layer(dct, params)

def torch_open_layer(name, params):
    dct = dp.load('database/ptorch/{}.torch_db'.format(name))
    
    return open_layer(dct, params)

def the_train(x, y):
    model = RandomForestRegressor()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model.fit(x_train, y_train)
    print('Model: {}'.format(model))
    
    r2 = model.score(x_test, y_test)
    print('R^2 Score: {}'.format(r2))
    
    y_true = y
    y_pred = model.predict(x)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print('Root Mean Squared Error: {} ms'.format(rmse/1000))
    
    print()
    
    return model

param_map = {
    'conv1d':['numf', 'channels', 'batch', 'nodes', 'kernel', 'stride', 'filters'] , 
    'conv2d':['numf', 'channels', 'batch', 'nodes', 'kernel', 'stride', 'filters'] , 
    
    'avg1d':['numf', 'channels', 'batch', 'nodes', 'pool', 'stride'], 
    'avg2d':['numf', 'channels', 'batch', 'nodes', 'pool', 'stride'], 
    
    'max1d':['numf', 'channels', 'batch', 'nodes', 'pool', 'stride'], 
    'max2d':['numf', 'channels', 'batch', 'nodes', 'pool', 'stride'], 
    
    'dense':['numf', 'channels', 'batch', 'nodes', 'units'], 
    
    'norm1d':['numf', 'channels', 'batch', 'nodes'], 
    'norm2d':['numf', 'channels', 'batch', 'nodes'], 
    
    'relu1d':['numf', 'channels', 'batch', 'nodes'], 
    'relu2d':['numf', 'channels', 'batch', 'nodes'],
        
    'tanh1d':['numf', 'channels', 'batch', 'nodes'], 
    'tanh2d':['numf', 'channels', 'batch', 'nodes'],
    
    'drop1d':['numf', 'channels', 'batch', 'nodes', 'drop'], 
    'drop2d':['numf', 'channels', 'batch', 'nodes', 'drop']
}
    
def time_per_step(name_map, pred_map, features, nodes):
    total = 0
    for layer in features:
        layer_name = layer['name']
        
        if layer_name not in name_map.keys():
            print('could not find: ', layer['name'])
            continue
            
        dim = layer['dim']
        name = name_map[layer_name]        
        if isinstance(name, list):
            name = name[dim]
        
        if name in ['conv1d', 'conv2d']:
            elem = np.array([
                layer['numf'],
                layer['channels'],
                layer['batch'],
                nodes,
                layer['kernel'][0],
                layer['stride'][0],
                layer['filters']
            ]).reshape(1,-1)
            
        elif name in ['avg1d', 'avg2d', 'max1d', 'max2d']:            
            elem = np.array([
                layer['numf'],
                layer['channels'],
                layer['batch'],
                nodes,
                layer['pool'][0],
                layer['stride'][0]
            ]).reshape(1,-1)
        
        elif name in ['norm1d', 'norm2d', 'tanh1d', 'tanh2d', 'relu1d', 'relu2d']:
            elem = np.array([
                layer['numf'],
                layer['channels'],
                layer['batch'],
                nodes,
            ]).reshape(1,-1)
        
        elif name in ['drop1d', 'drop2d']:
            elem = np.array([
                layer['numf'],
                layer['channels'],
                layer['batch'],
                nodes,
                layer['drop']
            ]).reshape(1,-1)
        
        elif name == 'dense':
            elem = np.array([
                layer['numf'],
                layer['channels'],
                layer['batch'],
                nodes,
                layer['units']
            ]).reshape(1,-1)
        
        [current] = pred_map[name].predict(elem)
        total += current

    name = 'dataset1d' if features[0]['dim'] == 1 else 'dataset2d'
    elem = np.array([
        layer['numf'],
        layer['channels'],
        layer['batch'],
        nodes,
    ]).reshape(1,-1)
    
    [current] = pred_map[name].predict(elem)
    total += current
    
    return total

# Tensorflow
print('TensorFlow')
tflow_pred_map = {}
for name,params in param_map.items():
    print(name)
    tflow_pred_map[name] = the_train(*tflow_open_layer(name,params))
    
# tflow_pred_map = {name: the_train(*tflow_open_layer(name,params)) for name,params in param_map.items()}

tflow_pred_map['dataset1d'] = the_train(*tflow_open_layer('dataset1d',['numf', 'channels', 'batch', 'nodes']))
tflow_pred_map['dataset2d'] = the_train(*tflow_open_layer('dataset2d',['numf', 'channels', 'batch', 'nodes']))

tflow_name_map = {
    'Conv1D': 'conv1d', 'Conv2D':'conv2d',
    'AveragePooling1D': 'avg1d', 'AveragePooling2D': 'avg2d',
    'MaxPooling1D': 'max1d', 'MaxPooling2D': 'max2d',
    'BatchNormalization': [None, 'norm1d', 'norm2d'],
    'Dropout': [None, 'drop1d', 'drop2d'],
    'ReLU': [None, 'relu1d', 'relu2d'],
    'tanh': [None, 'tanh1d', 'tanh2d'],
    'Dense': 'dense'
}

def tflow_time_per_step(features, nodes):
    return time_per_step(tflow_name_map, tflow_pred_map, features, nodes)


# Pytorch
print('PyTorch')
torch_pred_map = {}
for name,params in param_map.items():
    print(name)
    torch_pred_map[name] = the_train(*torch_open_layer(name,params))
    
# torch_pred_map = {name: the_train(*torch_open_layer(name,params)) for name,params in param_map.items()}

torch_pred_map['dataset1d'] = the_train(*torch_open_layer('dataset1d',['numf', 'channels', 'batch', 'nodes']))
torch_pred_map['dataset2d'] = the_train(*torch_open_layer('dataset2d',['numf', 'channels', 'batch', 'nodes']))

torch_name_map = {
    'Conv1d': 'conv1d', 'Conv2d':'conv2d',
    'AvgPool1d': 'avg1d', 'AvgPool2d': 'avg2d',
    'MaxPool1d': 'max1d', 'MaxPool2d': 'max2d',
    'BatchNorm1d': 'norm1d', 'BatchNorm2d': 'norm2d',
    'Dropout': 'drop1d', 'Dropout2d': 'drop2d',
    'ReLU': [None, 'relu1d', 'relu2d'],
    'Tanh': [None, 'tanh1d', 'tanh2d'],
    'Linear': 'dense'
}

def torch_time_per_step(features, nodes):
    return time_per_step(torch_name_map, torch_pred_map, features, nodes)
    