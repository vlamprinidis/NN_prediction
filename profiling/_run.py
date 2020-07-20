import subprocess as sub
from funs import numf_ls, batch_ls, nodes_ls, get_value
from models_tflow import model_ls as tflow_models
from models_torch import model_ls as torch_models

CMD = '/home/ubuntu/.night/bin/python3 /home/ubuntu/diploma/profiling/{file} -m {model} -numf {numf} -b {batch} -n {nodes} -e 5'

def go(cmd, nodes):
    print('RUNNING CMD:')
    print(cmd)
    
    p1 = sub.Popen(cmd, shell=True, stdout=sub.PIPE)
    print('RAN ON FIRST NODE')
    
    if nodes >= 2:
        p2 = sub.Popen('ssh vm2 "{}"'.format(cmd), shell=True, stdout=sub.PIPE)
        print('RAN ON SECOND NODE')
        
    if nodes == 3:
        p3 = sub.Popen('ssh vm3 "{}"'.format(cmd), shell=True, stdout=sub.PIPE)
        print('RAN ON THIRD NODE')

    res1 = p1.communicate()[0]
    print(res1.decode())
    print('FIRST NODE END')
    
    if nodes >= 2:
        res2 = p2.communicate()[0]
        print(res2.decode())
        print('SECOND NODE END')
    
    if nodes == 3:
        res3 = p3.communicate()[0]
        print(res3.decode())
        print('THIRD NODE END')

def execute(framework):
    # First Iteration
    for model in tflow_models:
        for numf in numf_ls:
            for batch in batch_ls:
                for nodes in [1,2,3]:
                    if get_value(model_str=model, numf=numf, 
                                 batch=batch, nodes=nodes, it=1, fname='./{}.pkl'.format(framework)) == None:
                        cmd = CMD.format(
                            file = 'run_{}.py'.format(framework),
                            model = model,
                            numf = numf,
                            batch = batch,
                            nodes = nodes
                        )
                        # run commands
                        go(cmd, nodes)

                        # kill 8890 ports just in case
                        go('fuser -k 8890/tcp', nodes)

execute('torch')
execute('tflow')