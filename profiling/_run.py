import subprocess as sub
from funs import numf_ls, batch_ls, get_value, hp_map

CMD = '/home/ubuntu/.night/bin/python3 /home/ubuntu/diploma/profiling/{file} -m {model} -numf {numf} -hp {hp} -b {batch} -n {nodes} -it {it} -e 5'

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
                       
                        
def execute(framework, model, hp, nodes, it):
    for numf in numf_ls:
        for batch in batch_ls:
            if get_value(model_str=model, numf=numf, hp=hp,
                         batch=batch, nodes=nodes, it=it, fname='results/{}.pkl'.format(framework)) == None:
                cmd = CMD.format(
                    file = 'run_{}.py'.format(framework),
                    model = model,
                    numf = numf,
                    hp = hp,
                    batch = batch,
                    nodes = nodes,
                    it = it
                )
                # run commands
                go(cmd, nodes)

#                 if nodes > 1:
#                     # kill 8890 ports just in case
#                     go('fuser -k 8890/tcp', nodes)

for it in [1]:
    for framework in ['tflow', 'torch']:
        for nodes in [1,2,3]:
            for key in hp_map:
                for hp in hp_map[key]:
                    execute(framework, model, hp, nodes, it)
