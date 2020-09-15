import torch
import csv
import pandas as pd

import socket
host = socket.gethostname()
print(host)
ranks = {
    'vlas-1':0,
    'vlas-2':1,
    'vlas-3':2
}
rank = ranks[host]

def conv_size_out(size_in, kern, stride):
    pad = 0
    dilation = 1
    return (size_in + 2*pad - dilation*(kern - 1) - 1) // stride + 1

def avg_size_out(size_in, kern, stride):
    pad = 0
    return (size_in + 2*pad - kern) // stride + 1

def max_size_out(size_in, kern, stride):
    pad = 0
    dilation = 1
    return (size_in + 2*pad - dilation*(kern - 1) - 1) // stride + 1

def give_evt_lst(evt):
    return [evt.key, str(evt.cpu_time), 
            str(evt.cpu_time_total), str(evt.self_cpu_time_total), str(evt.count)]

# This can overwrite the file, don't use outside funs_torch
def _save(function_events, target):
    headers = [
        'Name',
        'CPU time avg (us)',
        'CPU total (us)',
        'Self CPU time total (us)',
        'Number of Calls'
    ]
    with open(target, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        #way to write to csv file
        writer.writerow(headers)
        for evt in function_events:
            writer.writerow( give_evt_lst(evt) )

def get_ops(source):
    df = pd.read_csv(source, index_col=0)
    df = df.sort_values(by = ['Name'])
    
    return df

def distribute(model, train_dataset, nodes, batch):
    if(nodes < 2):
        raise NameError('More nodes needed')
        
    import os
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = '10.0.1.121'
        os.environ['MASTER_PORT'] = '8890'
        os.environ['GLOO_SOCKET_IFNAME'] = 'ens3'

        # initialize the process group
        dist.init_process_group(backend='gloo', 
                                init_method='env://', rank=rank, world_size=world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(42)

    def cleanup():
        dist.destroy_process_group()

    setup(rank = rank, world_size = nodes)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas = nodes,
                rank = rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch,
        sampler = train_sampler
    )

    model = DDP( model )
    
    return model, train_loader

pt_ops = [
    'addmm', 'mm',
    'AddmmBackward',

    'conv1d', 
    'MkldnnConvolutionBackward',

    'conv2d', 
    'MkldnnConvolutionBackward',

    'avg_pool1d',

    'avg_pool2d',

    'max_pool1d',

    'max_pool2d',

    'batch_norm',
    'NativeBatchNormBackward',

    'dropout',

    'feature_dropout',

    'tanh',

    'relu',

    'flatten'
]

def check_just(keywords):
    def find(x):
        return any(
            [word == x for word in keywords]
        )
    return find

def total_on_just(df, words):
    column = 'CPU total (us)'
    
    mask = df['Name'].apply(check_just(words))
    return df[mask][column].sum()

def profile(model, train_loader, given_epochs):    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    def train(epochs):
        total_step = len(train_loader)
        print(total_step)

        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(images)            
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {}' 
                           .format(epoch+1, epochs, i+1, total_step, loss))
    
    EPOCHS = given_epochs
    if rank == 0:
        prof_file = 'out_torch.csv'
        with torch.autograd.profiler.profile() as prof:
            train(EPOCHS)

        # save results
        _save(prof.key_averages(), prof_file)
        
        df = get_ops(prof_file)
        

        return total_on_just(df.reset_index(), pt_ops)
    
    else:
        train(EPOCHS)
        return None
