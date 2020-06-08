import torch
import csv

def give_evt_lst(evt):
    return [evt.key, str(evt.cpu_time), 
            str(evt.cpu_time_total), str(evt.self_cpu_time_total), str(evt.count)]

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
    
    return False

def prepare(build_func, train_dataset, numf, batch, rank, nodes):
    if nodes > 1:
        import os
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallelCPU as DDP

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

        model = DDP( build_func(numf) )

    else:
        train_loader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = batch,
            shuffle = True
        )
        
        model = build_func(numf)

    return model, train_loader

def profile(model, train_loader, epochs, rank, criterion, optimizer):
    def train():
        total_step = len(train_loader)
        print(total_step)

        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                labels = labels

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
    
    if(rank == 0):
        with torch.autograd.profiler.profile() as prof:
            train()
            
        # save results
        _save(prof.key_averages(), 'out_torch.csv')
        
    else:
        train()
