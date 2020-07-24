from funs import insert_prof_args, execute_prof
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t','--times', type = int, nargs='+', default = [20*60, 30*60], help='Timeouts in seconds')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-tflow', action='store_true') #implies default=False and vice versa
group.add_argument('-torch', action='store_true')

args = insert_prof_args(parser).parse_args()

success = execute_prof(
            framework = 'tflow' if args.tflow else 'torch',
            model = args.model, 
            numf = args.num_features, 
            hp = args.hyper_param,
            batch = args.batch, 
            nodes = args.nodes, 
            timeout = args.times[0]
        )

print('Success') if success else print('Failure')
