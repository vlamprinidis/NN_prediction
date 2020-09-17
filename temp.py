from lib import clean_go

cmd = '/home/ubuntu/.env/bin/python3 /home/ubuntu/profile/tflow/_conv_tanh.py -numf 32 -batch 64 -nodes 3 -epochs 5 -channels 3 -dim 2 -filters 3 -kernel 4 -stride 2'

clean_go(cmd, 3)