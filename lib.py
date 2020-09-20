import numpy as np
import os

from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def detuple(elem):
    return elem[0] if isinstance(elem, tuple) else elem

def proc(_arr, isdense = 0):
    def mymax(x):
        return max(1,x)

    arr = _arr.copy()
    
    epochs = arr[:,0]
    ds = arr[:,1]
    batch = arr[:,4 - isdense]
    nodes = arr[:,5 - isdense]
    steps = epochs*np.vectorize(mymax)(ds/nodes/batch)
    
    arr[:,-1] = arr[:,-1]/steps
    
    return arr[:,2:-1], arr[:,-1]
    
def the_train(x, y):
    model = RandomForestRegressor()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
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

class Reg:
    def __init__(self, fw):
        assert fw in ['tflow','ptorch']
        
        self.reg_map = {}
        names = ['conv1d', 'conv2d', 
            'avg1d', 'avg2d', 
            'max1d', 'max2d',
            'dense', 'final_dense',
            'drop1d', 'drop2d',
            'norm1d', 'norm2d',
            'relu1d', 'relu2d',
            'tanh1d', 'tanh2d',
            'flatten1d', 'flatten2d']
        
        for name in names:
            file = name + '.' + fw
            try:
                arr = np.genfromtxt(os.path.join('stats', file), delimiter=',')
            except:
                print(file, 'missing')
                print()
                continue
            
            isdense = 1 if name in ['dense', 'final_dense'] else 0
            
            x,y = proc(arr, isdense=isdense)
            print(name)
            self.reg_map[name] = the_train(x,y)

def total_time(reg_map, features, nodes):
    total = 0
    for layer in features:
        name = layer['name']
        
        if name not in reg_map.keys():
            continue
        
        if name in ['conv1d', 'conv2d']:
            elem = np.array([
                layer['numf'],
                layer['channels'],
                layer['batch'],
                nodes,
                detuple(layer['kernel']),
                detuple(layer['stride']),
                layer['filters']
            ]).reshape(1,-1)
            
        elif name in ['avg1d', 'avg2d', 'max1d', 'max2d']:            
            elem = np.array([
                layer['numf'],
                layer['channels'],
                layer['batch'],
                nodes,
                detuple(layer['pool']),
                detuple(layer['stride'])
            ]).reshape(1,-1)
        
        elif name in ['norm1d', 'norm2d', 'tanh1d', 'tanh2d', 'relu1d', 'relu2d', 'flatten1d', 'flatten2d']:
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
        
        elif name in ['dense','final_dense']:
            elem = np.array([
                layer['numf'],
                layer['batch'],
                nodes,
                layer['units']
            ]).reshape(1,-1)
        
        [current] = reg_map[name].predict(elem)
        total += current
    
    return total

def predict(reg_map, features, epochs, ds, nodes, batch):
    steps = epochs*max(1,ds/nodes/batch)
    
    return steps*total_time(reg_map, features, nodes)/1000/1000


