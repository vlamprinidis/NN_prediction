import os
import subprocess as sub
import signal

def go(cmd, nodes, timeout=20*60):
    print('RUNNING CMD:')
    print(cmd)
    
    p1 = sub.Popen('ssh vm1 "{}"'.format(cmd), shell=True)
    print('RAN ON FIRST NODE')
    
    if nodes >= 2:
        p2 = sub.Popen('ssh vm2 "{}"'.format(cmd), shell=True)
        print('RAN ON SECOND NODE')
        
    if nodes == 3:
        p3 = sub.Popen('ssh vm3 "{}"'.format(cmd), shell=True)
        print('RAN ON THIRD NODE')
    
    def kill(p):
        try:
            os.killpg(p.pid, signal.SIGINT) # send signal to the process group
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
            success = comm(p3)            
            print('THIRD NODE END')
    
    return success

def clean_go(cmd, nodes, timeout=20*60):
    # run commands
    success = go(cmd + ' >> pred.out 2>> pred.err', nodes, timeout)

    if not success:
        # kill 8890 ports
        go('fuser -k 8890/tcp', nodes, timeout)
        print('Failure')
    
    return success

