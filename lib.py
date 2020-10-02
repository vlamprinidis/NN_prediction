import numpy as np
import os

from sklearn import model_selection

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def detuple(elem):
    return elem[0] if isinstance(elem, tuple) else elem

def normalize(arr):
    mean = np.mean(arr)
    var = np.var(arr)
    
    arr = (arr - mean)/var
    
    return arr, mean, var

def proc(_arr, nodes, isdense = 0):
    def mymax(x):
        return max(1,x)
    
    nodes_col = 5 - isdense

    arr = _arr[_arr[:,nodes_col] == nodes].copy()
    
    epochs = arr[:,0]
    ds = arr[:,1]
    batch = arr[:,4 - isdense]
    nodes = arr[:,nodes_col]
    steps = epochs*np.vectorize(mymax)(ds/batch/nodes)
    
    arr[:,-1] = arr[:,-1]/steps
    
    proc_arr = np.delete(arr, nodes_col, 1) # remove nodes column
    
    return proc_arr[:,2:-1], proc_arr[:,-1]

def the_name(name):
    dct = {
        'conv2d':'Convolutional 2D',
        'max2d':'Max Pooling 2D',
        'avg2d':'Average Pooling 2D',
        'dense':'Fully Connected 2D', 'final_dense':'Fully Connected 2D',
        'drop2d':'Dropout 2D',
        'norm2d':'Batch Normalization',
        'relu2d':'ReLU 2D',
        'tanh2d':'Tanh 2D',
        'flatten2d':'Flatten 2D'
    }
    
    if name in dct.keys():
        return dct[name]
    else:
        return name


class Reg_N:
    def __init__(self, fw, nodes):
        assert fw in ['tflow','ptorch']
        assert nodes in [1,2,3]
        
        self.nodes = nodes
        self.reg_map = {}
        self.fw = fw
        
        names = ['conv1d', 'conv2d', 
            'avg1d', 'avg2d', 
            'max1d', 'max2d',
            'dense', 'final_dense',
            'drop1d', 'drop2d',
            'norm1d', 'norm2d',
            'relu1d', 'relu2d',
            'tanh1d', 'tanh2d',
            'flatten1d', 'flatten2d']
        
#         data = np.array([['Layer', 'Regressor', 'R^2', 'RMSE']])
#         with open('scores.{}.n{}'.format(self.fw,self.nodes),'w') as file:
#             np.savetxt(file, data, delimiter=",", fmt="%s")
            
        for name in names:
            file = name + '.' + fw
            try:
                arrs = [np.genfromtxt(os.path.join('stats', 'node_{}'.format(node), file), delimiter=',') for node in [1,2,3]]
                arr = np.concatenate(arrs, axis=0)
            except:
                print(file, 'missing')
                continue
            
            isdense = 1 if name in ['dense', 'final_dense'] else 0

            x,y = proc(arr, nodes, isdense=isdense)
            self.x = x
            self.y = y
            
            y,mean,var = normalize(y)
            self.y_norm = y
            self.mean = mean
            self.var = var
            
            #print(the_name(name))
            
#             self.reg_map[name] = self.the_score_train(x,y,name)
            self.reg_map[name] = self.the_train(x,y)
            
    def the_train(self, x_train, y_train):
        model = RandomForestRegressor()
        model.fit(x_train, y_train)

        return model 
    
    def the_score_train(self, x, y, name):
        for model in [LinearRegression(), Ridge(), Lasso(), ElasticNet(), 
                      KNeighborsRegressor(), DecisionTreeRegressor(), SVR(), RandomForestRegressor()]:

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            model.fit(x_train, y_train)

            r2 = model.score(x_test, y_test)

            y_true = y
            y_pred = model.predict(x)
            rmse = sqrt(mean_squared_error(y_true, y_pred))
            
            data = np.array([[the_name(name), model.__class__.__name__, r2, rmse]])

            with open('scores.{}.n{}'.format(self.fw,self.nodes),'a') as file:
                np.savetxt(file, data, delimiter=",", fmt="%s")

        return model

    def predict(self, features, epochs, ds, batch):
        steps = epochs*max(1,ds/batch/self.nodes)
        time = self.mean + self.var*total_time(self.reg_map, features)

        return steps*time/1000/1000

    def total_time(self, features):
        reg_map = self.reg_map
        
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
                    #nodes,
                    detuple(layer['kernel']),
                    detuple(layer['stride']),
                    layer['filters']
                ]).reshape(1,-1)

            elif name in ['avg1d', 'avg2d', 'max1d', 'max2d']:            
                elem = np.array([
                    layer['numf'],
                    layer['channels'],
                    layer['batch'],
                    #nodes,
                    detuple(layer['pool']),
                    detuple(layer['stride'])
                ]).reshape(1,-1)

            elif name in ['norm1d', 'norm2d', 'tanh1d', 'tanh2d', 'relu1d', 'relu2d', 'flatten1d', 'flatten2d']:
                elem = np.array([
                    layer['numf'],
                    layer['channels'],
                    layer['batch'],
                    #nodes,
                ]).reshape(1,-1)

            elif name in ['drop1d', 'drop2d']:
                elem = np.array([
                    layer['numf'],
                    layer['channels'],
                    layer['batch'],
                    #nodes,
                    layer['drop']
                ]).reshape(1,-1)

            elif name in ['dense','final_dense']:
                elem = np.array([
                    layer['numf'],
                    layer['batch'],
                    #nodes,
                    layer['units']
                ]).reshape(1,-1)

            [current] = reg_map[name].predict(elem)
            total += current

        return total


