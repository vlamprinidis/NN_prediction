import subprocess as sub
import os
import signal
from funs import numf_ls, batch_ls, get_value, hp_map

CMD = '/home/ubuntu/.night/bin/python3 /home/ubuntu/diploma/profiling/{file} -m {model} -numf {numf} -hp {hp} -b {batch} -n {nodes} -e 5'

def go(cmd, nodes, timeout):
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
    
    def kill(p):
        try:
#             p.kill()
            os.killpg(p.pid, signal.SIGINT) # send signal to the process group
#             out = p.communicate()[0]    
        except:
            print('Kill unsuccessful')
        
    def comm(p):
        try:
            out = p.communicate(timeout=timeout)[0]
            if p.returncode != 0: 
                print('Command failed')
                return False
            
            return True
        
        except sub.TimeoutExpired:
            print('Timeout')
            kill(p)
            return False
        
        except:
            print('Communication failed')
            return False
            
    success = comm(p1)
    
    print('FIRST NODE END')
    
    if nodes >= 2:
        if not success:
            print('killing p2')
            kill(p2)
            
        else:
            success = comm(p2)
            print('SECOND NODE END')
            
    if nodes == 3:
        if not success:
            print('killing p3')
            kill(p3)
            
        else:
            comm(p3)            
            print('THIRD NODE END')
    
    return success

def execute(framework, model, hp, nodes, timeout):
    for numf in numf_ls:
        for batch in batch_ls:
            if get_value(model_str=model, numf=numf, hp=hp,
                         batch=batch, nodes=nodes, fname='results/{}.pkl'.format(framework)) == None:
                cmd = CMD.format(
                    file = 'run_{}.py'.format(framework),
                    model = model,
                    numf = numf,
                    hp = hp,
                    batch = batch,
                    nodes = nodes
                )
                # run commands
                success = go(cmd, nodes, timeout)

                if not success:
                    # kill 8890 ports
                    go('fuser -k 8890/tcp', nodes, timeout)
                    
            else:
                print('Combination exists: {} numf{} hp{} batch{} nodes{} fw{}'.format(
                    model,numf,hp,batch,nodes,framework
                ))

for nodes in [3,2,1]:
    for framework in ['tflow']:
        for model in list(hp_map):
            for hp in hp_map[model]:
                execute(framework, model, hp, nodes)