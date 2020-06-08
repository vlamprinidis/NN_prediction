import sys
pth = '/home/ubuntu/diploma'
if(pth not in sys.path):
    sys.path.insert(0, pth)

from funs.maps import maps
maps = maps()

def parse(argv):

    message = 'Arguments (in the order given): \n<Framework>:= [tflow, ptorch] \n<Model>:= {models} \
    \n<Batch>:= [64, 128, 512] \n<Rank>:= [0, 1, 2] \n<Nodes>:= [1, 2, 3]'\
    .format( models = str(maps.known_models).translate({39: None}) )

    if(len(argv) < 6 ):
        print(message)
        exit()

    [_, FW, MODEL, batch, rank, nodes] = argv

    if(FW not in maps.fws 
       or MODEL not in maps.known_models
       or batch not in maps.batches_str
       or rank not in maps.ranks_str
       or nodes not in maps.nodes_str
      ):
        print('Incorrent argument format\n')
        print(message)
        exit()

    BATCH, RANK, NODES = int(batch), int(rank), int(nodes)

    print( 'Framework = {}, Model = {}, Batch = {}, Rank = {}, Nodes = {}'.format(FW, MODEL, BATCH, RANK, NODES) )
    
    return {'FW': FW, 'MODEL': MODEL, 'BATCH': BATCH, 'RANK': RANK, 'NODES': NODES}
