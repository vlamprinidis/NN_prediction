import pickle
import os

def my_key(dct_key):
    return frozenset({
        (x,y) for x,y in dct_key.items()
    })

def from_key(fz_key): # inverse of my_key
    return {x:y for x,y in fz_key}
    
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
    

def check(keywords):
    def find(x):
        return any(
            [word in x for word in keywords]
        )
    return find

def check_just(keywords):
    def find(x):
        return any(
            [word == x for word in keywords]
        )
    return find

def from_set(fz):
    return {name:value for (name,value) in fz}