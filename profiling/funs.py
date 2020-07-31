import pickle
import socket
import os
import subprocess as sub
import signal
import argparse

host = socket.gethostname()
ranks = {
    'vlas-1':0,
    'vlas-2':1,
    'vlas-3':2,
    'vlassis_ld_pc':-1
}
rank = ranks[host]

def my_key(model_str, numf, hp, batch, nodes):
    key = frozenset({
        ('model_str', model_str),
        ('numf', numf),
        ('hp', hp),
        ('batch', batch),
        ('nodes', nodes)
    })
    
    return key

# This can overwrite the file
def _save(data, fname):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)
    fp.close()

# Creates the file if it doesn't already exist
def load(fname):
    if( not os.path.exists(fname) ):
        print( 'No such file: {}'.format(fname) )
        print( 'Creating empty file: {}'.format(fname) )
        _save({}, fname)
        return {}

    with open(fname, 'rb') as fp:
        data = pickle.load(fp)
    fp.close()
        
    return data

def update(key, value, fname):
    data = load(fname)
    data[key] = value
    _save(data, fname)
    print('\nAdded: {}\n'.format(key))

def get_keys(fname):
    data = load(fname)
    return list(data.keys())
    
def get_value(data, model_str, numf, hp, batch, nodes):
    key = my_key(model_str, numf, hp, batch, nodes)
    value = data.get(key)
    
    if value == None:
        print('No such key')
        
    return value

numf_ls = [16, 32, 64, 128]
batch_ls = [32, 64, 128, 256, 512]
nodes_ls = [1,2,3]

hp_map = {
    'avg1d': [2,4],
    'avg2d': [2,4],
    'conv1d': [2,4,8],
    'conv2d': [2,4,8],
    'max1d': [2,4],
    'max2d': [2,4],
    'dense': [32,64,128],
    'norm1d':[0],
    'norm2d': [0]
}

def insert_prof_args(my_parser):
    print('\n')
    print('This is ' + host)

    my_parser.add_argument('-m', '--model', type = str, required = True, 
                           choices = list(hp_map.keys()))
    my_parser.add_argument('-numf', '--num_features', type = int, required = True,
                          choices = numf_ls )
    
    my_parser.add_argument('-hp', '--hyper_param', type = int, required = True )
    
    my_parser.add_argument('-b', '--batch', type = int, required = True, 
                           choices = batch_ls )
    my_parser.add_argument('-n', '--nodes', type = int, required = True,
                          choices = nodes_ls )
    my_parser.add_argument('-e', '--epochs', type = int, default = 5)
    
    return my_parser

def go(cmd, nodes, timeout):
    print('RUNNING CMD:')
    print(cmd)
    
    p1 = sub.Popen(cmd, shell=True)
    print('RAN ON FIRST NODE')
    
    if nodes >= 2:
        p2 = sub.Popen('ssh vm2 "{}"'.format(cmd), shell=True)
        print('RAN ON SECOND NODE')
        
    if nodes == 3:
        p3 = sub.Popen('ssh vm3 "{}"'.format(cmd), shell=True)
        print('RAN ON THIRD NODE')
    
    def kill(p):
        try:
            os.killpg(p.pid, signal.SIGINT) # send signal to the process group
        except:
            print('Kill unsuccessful')
        
    def comm(p):
        try:
            out = p.communicate(timeout=timeout)[0]
            if p.returncode != 0: 
                print('Command failed')
                return False
            
            return True
        
        except sub.TimeoutExpired:
            print('Timeout')
            kill(p)
            return False
        
        except:
            print('Communication failed')
            return False
            
    success = comm(p1)
    
    print('FIRST NODE END')
    
    if nodes >= 2:
        if not success:
            print('killing p2')
            kill(p2)
            
        else:
            success = comm(p2)
            print('SECOND NODE END')
            
    if nodes == 3:
        if not success:
            print('killing p3')
            kill(p3)
            
        else:
            success = comm(p3)            
            print('THIRD NODE END')
    
    return success

def clean_go(cmd, nodes, timeout):
    # run commands
    success = go(cmd, nodes, timeout)

    if not success:
        # kill 8890 ports
        go('fuser -k 8890/tcp', nodes, timeout)
        print('Failure')
    
    return success

def execute_prof(framework, model, numf, hp, batch, nodes, timeout, frame_to_load = None):    
    if framework not in ['tflow', 'torch']:
        raise NameError('Enter tflow or torch')
    
    data = load('data.{}'.format(framework)) if frame_to_load == None else frame_to_load[framework]
    
    CMD = '/home/ubuntu/.night/bin/python3 /home/ubuntu/diploma/profiling/{file} -m {model} -numf {numf} -hp {hp} -b {batch} -n {nodes} -e 10 >> prof_all.out 2>> prof_all.err'
    
    if get_value(data=data, model_str=model, numf=numf, hp=hp, batch=batch, nodes=nodes) == None:
        
        print('Combination missing: {} numf{} hp{} batch{} nodes{} framework_{}'.format(
            model,numf,hp,batch,nodes, framework
        ))
        
        cmd = CMD.format(
            file = 'run_{}.py'.format(framework),
            model = model,
            numf = numf,
            hp = hp,
            batch = batch,
            nodes = nodes
        )
        
        # run commands
        return clean_go(cmd, nodes, timeout)

    else:
        print('Combination exists: {} numf{} hp{} batch{} nodes{} framework_{}'.format(
            model,numf,hp,batch,nodes,framework
        ))
        return False
