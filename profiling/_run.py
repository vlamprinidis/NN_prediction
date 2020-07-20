import subprocess as sub
from funs import numf_ls, batch_ls, get_value

CMD = '/home/ubuntu/.night/bin/python3 /home/ubuntu/diploma/profiling/{file} -m {model} -numf {numf} -hp {hp} -b {batch} -n {nodes} -e 5'

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
                    nodes = nodes
                )
                # run commands
                go(cmd, nodes)

                if nodes > 1:
                    # kill 8890 ports just in case
                    go('fuser -k 8890/tcp', nodes)

model_ls = [('avg1d',[1,2,3]),
            ('avg2d',[1,2,3]),
            ('conv1d',[1,3,5,7,11]),
            ('conv2d',[1,3,5,7,11]),
            ('max1d',[1,2,3]),
            ('max2d',[1,2,3]),
            ('dense',[10,84,120,1000,4096]),
            ('norm1d',[0]),
            ('norm2d',[0])]

# for it in [1,2,3,4]:
#     for framework in ['tflow', 'torch']:
#         for nodes in [1,2,3]:
#             for model,hps in model_ls:
#                 for hp in hps:
#                     execute(framework, model, hp, nodes, it)

for fr in ['tflow', 'torch']:
    for model,hp in [
#             ('avg2d',2),
#             ('conv1d',3),
#             ('conv2d',5),
#             ('max1d',2),
#             ('max2d',3),
#             ('dense',10),
#             ('norm1d',0),
            ('norm2d',0)]:
        cmd = CMD.format(   file = 'run_{}.py'.format(fr),
                            model = model,
                            numf = 32,
                            hp = hp,
                            batch = 32,
                            nodes = 3
                        )
                # run commands
        go(cmd, 3)