import subprocess as sub
from funs import numf_ls, batch_ls, nodes_ls
from models_tflow import model_ls as tflow_models
from models_torch import model_ls as torch_models

CMD = '/home/ubuntu/.night/bin/python3 /home/ubuntu/diploma/profiling/{file} -m {model} -numf {numf} -b {batch} -n {nodes} -e 5'

def go(cmd, nodes):
    print('\n*************RUNNING*************')
    print(cmd)
    print('************END OF CMD***********\n')
    print('FIRST NODE\n')
    
    p1 = sub.Popen(cmd, shell=True, stdout=sub.PIPE)
    
    if nodes >= 2:
        print('SECOND NODE\n')
        p2 = sub.Popen('ssh vm2 "{}"'.format(cmd + ' -use_prof False'), shell=True, stdout=sub.PIPE)
        
    if nodes == 3:
        print('THIRD NODE\n')
        p3 = sub.Popen('ssh vm3 "{}"'.format(cmd + ' -use_prof False'), shell=True, stdout=sub.PIPE)

    res1 = p1.communicate()[0]
    print(res1.decode())
    
    if nodes >= 2:
        res2 = p2.communicate()[0]
        print(res2.decode())
    
    if nodes == 3:
        res3 = p3.communicate()[0]
        print(res3.decode())

for model in tflow_models:
    for numf in numf_ls:
        for batch in batch_ls:
            for nodes in nodes_ls:
#                 # Tensorflow
#                 cmd = CMD.format(
#                     file = 'run_tflow.py',
#                     model = model,
#                     numf = numf,
#                     batch = batch,
#                     nodes = nodes
#                 )

#                 go(cmd, nodes)
                
                # PyTorch
                cmd = CMD.format(
                    file = 'run_torch.py',
                    model = model,
                    numf = numf,
                    batch = batch,
                    nodes = nodes
                )

                go(cmd, nodes)
                
                if(nodes==3):
                    exit()

# Afterwards, check that all ran correctly

# p2 = sub.Popen('ssh vm2 "{}"'.format(cmd+' -use_prof False'), shell=True, stdout=sub.PIPE)
# p3 = sub.Popen('ssh vm3 "{}"'.format(cmd+' -use_prof False'), shell=True, stdout=sub.PIPE)