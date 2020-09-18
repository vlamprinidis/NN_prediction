from lib import clean_go
nodes = 1

cmd = '/home/ubuntu/.env/bin/python3 /home/ubuntu/profile/tflow/_conv.py -numf 32 -batch 64 -nodes {} -epochs 5 -channels 3 -dim 2 -filters 3 -kernel 4 -stride 2'.format(nodes)

clean_go(cmd, nodes)