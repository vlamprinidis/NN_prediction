from funs import load, numf_ls, batch_ls, hp_map, get_value, clean_go, insert_prof_args
import argparse

CMD = '/home/ubuntu/.night/bin/python3 /home/ubuntu/diploma/profiling/{file} -m {model} -numf {numf} -hp {hp} -b {batch} -n {nodes} -e 10 -target {target} > out.out'

parser = argparse.ArgumentParser()
parser.add_argument('-run_tflow', action='store_true') #implies default=False and vice versa
parser.add_argument('-run_torch', action='store_true')
parser.add_argument('-run_once', action='store_true')
parser.add_argument('-tf', type = str, default = './results/tflow.pkl', help = 'File to store TensorFlow results')
parser.add_argument('-pt', type = str, default = './results/torch.pkl', help = 'File to store PyTorch results')
parser.add_argument('-t','--times', type = int, nargs='+', default = [20*60, 30*60], help='Timeouts in seconds')
args = parser.parse_args()

_tflow = load(args.tf)
_torch = load(args.pt)

def execute(framework, model, numf, hp, batch, nodes, timeout):
    if framework == 'tflow':
        data = _tflow
        target = args.tf
        
    elif framework == 'torch':
        data = _torch
        target = args.pt
        
    else:
        raise NameError('Enter tflow or torch')
        
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
            nodes = nodes,
            target = target
        )
        
        # run commands
        return clean_go(cmd, nodes, timeout)

    else:
        print('Combination exists: {} numf{} hp{} batch{} nodes{} framework_{}'.format(
            model,numf,hp,batch,nodes,framework
        ))
        return False

def big_loop(framework, times, once=False): #times = list of seconds for timeout
    for timeout in times:
        for nodes in [3,2,1]:
            for model in list(hp_map):
                for hp in hp_map[model]:
                    for numf in numf_ls:
                        for batch in batch_ls:
                            success = execute(
                                framework=framework,
                                model=model, 
                                numf=numf, 
                                hp=hp,
                                batch=batch, 
                                nodes=nodes, 
                                timeout=timeout
                            )
                            
                            if once:
                                return True
        if(success):
            return True
    
    return success

if args.run_tflow:
    big_loop('tflow', args.times, args.run_once)

if args.run_torch:
    big_loop('torch', args.times, args.run_once)

