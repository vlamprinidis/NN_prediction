from funs import load, numf_ls, batch_ls, clean_go

cmd = '/home/ubuntu/.env/bin/python3 /home/ubuntu/vms/tensorflow/_conv.py -numf 16 -batch 64 -nodes 1 -epochs 5 -channels 1 -dim 2 -filters 1 -kernel 4 -stride 2'

clean_go(cmd, 1)