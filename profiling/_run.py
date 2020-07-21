import subprocess as sub
import os
import signal
from funs import numf_ls, batch_ls, get_value, hp_map

CMD = '/home/ubuntu/.night/bin/python3 /home/ubuntu/diploma/profiling/{file} -m {model} -numf {numf} -hp {hp} -b {batch} -n {nodes} -it {it} -e 5'

def go(cmd, nodes):
    print('RUNNING CMD:')
    print(cmd)
    
    p1 = sub.Popen(cmd, shell=True)
    print('RAN ON FIRST NODE')
    
    if nodes >= 2:
        p2 = sub.Popen('ssh vm2 "{}"'.format(cmd), shell=True)
        print('RAN ON SECOND NODE')
        
    if nodes == 3:
        p3 = sub.Popen('ssh vm3 "{}"'.format(cmd), shell=True)
        print('RAN ON THIRD NODE')

    fail = False
    
    def kill(p):
        os.killpg(p.pid, signal.SIGINT) # send signal to the process group
        out = p.communicate()[0]     
        
    def comm(p):
        try:
            out = p.communicate(timeout=1)[0]
        except sub.TimeoutExpired:
            fail = True
            kill(p)
            print('Timeout')
        
    def tofail(p):
        if (not fail) and p.returncode != 0: 
            fail = True
            print('Command failed')
            
    comm(p1)
    tofail(p1)
    
    print('FIRST NODE END')
    
    if nodes >= 2:
        if fail:
            kill(p2)
            print('killed p2')
            
        else:
            comm(p2)
            tofail(p2)
            
            print('SECOND NODE END')
            
    if nodes == 3:
        if fail:
            kill(p3)
            print('killed p3')
            
        else:
            comm(p3)
            tofail(p3)
            
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
            else:
                print('Combination exists: {} numf{} hp{} batch{} nodes{} it{} fw{}'.format(
                    model,numf,hp,batch,nodes,it,framework
                ))

for nodes in [3,2,1]:
    for framework in ['tflow']:
        for model in list(hp_map):
            for hp in hp_map[model]:
                execute(framework, model, hp, nodes, it)
