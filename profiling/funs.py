import pickle

# dict_ = { (<layer>,<numf>,<batch>,<nodes>,<it>) : <dataframe> }

def save(data, fname):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)
    fp.close()
    
def load(fname):
    with open(fname, 'rb') as fp:
        data = pickle.load(fp)
    fp.close()
        
    return data

def update(key, df, fname):
    data = load(fname)
    data[key] = df
    save(data, fname)

def parse(argv):
    message = 'Arguments (in the order given): Layer Num_features Batch Rank Nodes Iteration\n\n'
    if(len(argv) < 7 ):
        print(message)
        exit()

    [_, layer, numf, batch, rank, nodes, it] = argv

    numf, batch, rank, nodes, it = int(numf), int(batch), int(rank), int(nodes), int(it)

    print( 'Layer = {}, Num_features = {}, Batch = {}, Rank = {}, Nodes = {}, Iteration = {}\n\n'.format(layer, numf, batch, rank, nodes, it))
    
    return layer, numf, batch, rank, nodes, it
