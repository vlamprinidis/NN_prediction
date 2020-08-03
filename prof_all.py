from funs import load, numf_ls, batch_ls, hp_map, execute_prof, load
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tflow', action='store_true') #implies default=False and vice versa
parser.add_argument('-torch', action='store_true')
parser.add_argument('-t','--timeout', type = int, default = 30*60, help='Timeout in seconds')
args = parser.parse_args()

def big_loop(framework, timeout): #times = list of seconds for timeout
    frame_to_load = {'tflow': load('data.tflow'), 'torch': load('data.torch')}
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
                            timeout=timeout,
                            frame_to_load=frame_to_load
                        )
                        print('Success') if success else print('Failure')
                        #exit()

if args.tflow:
    big_loop('tflow', args.timeout)

if args.torch:
    big_loop('torch', args.timeout)

