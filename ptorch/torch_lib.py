import torch

def distribute(model, train_dataset, nodes, batch):
    import socket
    host = socket.gethostname()
    print(host)
    ranks = {
        'vlas-1':0,
        'vlas-2':1,
        'vlas-3':2
    }
    rank = ranks[host]
    
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

def get_accuracy(model, data_loader):
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def train(model, train_loader, given_epochs):    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    def _train(epochs):
        total_step = len(train_loader)
        print(total_step)

        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                outputs,_ = model(images)            
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {}' 
                           .format(epoch+1, epochs, i+1, total_step, loss))
                    
        print('Accuracy: {}'.format(
                        get_accuracy(model, train_loader)
        ))
    
    _train(1)
    _train(given_epochs)
    