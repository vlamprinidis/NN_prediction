from funs import numf_ls, batch_ls, hp_map, get_value

_tflow = load('./results/tflow.pkl')
_torch = load('./results/torch.pkl')

for nodes in [3,2,1]:
    for model in list(hp_map):
        for hp in hp_map[model]:
            for numf in numf_ls:
                for batch in batch_ls:
                    def check(data):
                        if get_value(data = data, model_str=model, numf=numf, hp=hp,
                                     batch=batch, nodes=nodes) == None:
                            print('Combination missing: {} numf{} hp{} batch{} nodes{}'.format(
                                model,numf,hp,batch,nodes
                            ))
                            
                    print('Tflow:')        
                    check(_tflow)
                    
                    print('Torch:')
                    ckeck(_torch)