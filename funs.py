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
    'vlas-3':2
}
rank = ranks[host]

def my_key(dct):
    key = frozenset({
        (key,value) for key,value in dct.items()
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

numf_ls = [16, 32, 64, 128]
batch_ls = [32, 64, 128, 256, 512]
nodes_ls = [1,2,3]

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

def clean_go(cmd, nodes, timeout=20*60):
    # run commands
    success = go(cmd + ' >> prof.out 2>> prof.err', nodes, timeout)

    if not success:
        # kill 8890 ports
        go('fuser -k 8890/tcp', nodes, timeout)
        print('Failure')
    
    return success
