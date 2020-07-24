from funs import load, numf_ls, batch_ls, hp_map, execute_prof
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tflow', action='store_true') #implies default=False and vice versa
parser.add_argument('-torch', action='store_true')
parser.add_argument('-once', action='store_true')
parser.add_argument('-t','--times', type = int, nargs='+', default = [20*60, 30*60], help='Timeouts in seconds')
args = parser.parse_args()

def big_loop(framework, times, once=False): #times = list of seconds for timeout
    for timeout in times:
        for nodes in [3,2,1]:
            for model in list(hp_map):
                for hp in hp_map[model]:
                    for numf in numf_ls:
                        for batch in batch_ls:
                            success = execute_prof(
                                framework=framework,
                                model=model, 
                                numf=numf, 
                                hp=hp,
                                batch=batch, 
                                nodes=nodes, 
                                timeout=timeout
                            )
                            print('Success') if success else print('Failure')
                            if once:
                                return True
        if(success):
            return True
    
    return success

if args.tflow:
    big_loop('tflow', args.times, args.once)

if args.torch:
    big_loop('torch', args.times, args.once)

