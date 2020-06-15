import subprocess as sub

cmd = '/home/ubuntu/.night/bin/python3 /home/ubuntu/diploma/profiling/run_torch.py -m conv1d -numf 32 -b 32 -n 3 -e 1'

p1 = sub.Popen(cmd, shell=True, stdout=sub.PIPE)
p2 = sub.Popen('ssh vm2 "{}"'.format(cmd), shell=True, stdout=sub.PIPE)
p3 = sub.Popen('ssh vm3 "{}"'.format(cmd), shell=True, stdout=sub.PIPE)

res1 = p1.communicate()[0]
print(res1.decode())

res2 = p2.communicate()[0]
print(res2.decode())

res3 = p3.communicate()[0]
print(res3.decode())