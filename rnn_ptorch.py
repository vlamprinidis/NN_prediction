#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys

BATCH = int(sys.argv[1])
EPOCHS = int(sys.argv[2])

RANK = int(sys.argv[3])
NODES = int(sys.argv[4])

print( 'Batch = {}, Epochs = {}, Rank = {}, Nodes = {}'.format(BATCH, EPOCHS, RANK, NODES) )

# In[ ]:


import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict


# In[ ]:


import torch.nn.functional as F

sequence_length = 28
input_size = 28

# Hyper-parameters
learning_rate = 0.01

num_classes = 10
num_cells = 128
dense_size = 32
drop_pr = 0.2

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size, num_cells, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, dense_size)
        self.fc2 = nn.Linear(dense_size, num_classes)
        self.dropout = nn.Dropout(drop_pr)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(1, x.size(0), num_cells)
        c0 = torch.zeros(1, x.size(0), num_cells)
        
        # Forward propagate LSTM
        out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, num_cells)
        
        out = self.dropout(out[:, -1, :])
        
        out = F.relu(self.fc1(out))
        
        out = self.dropout(out)
        
        out = self.fc2(out) # no softmax needed - nn.CrossEntropy does it
        
        return out


# In[ ]:


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH, 
                                          shuffle=False)


# In[ ]:


if NODES > 1:
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

    setup(rank = RANK, world_size = NODES)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas = NODES,
                rank = RANK
            )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH,
                                               sampler = train_sampler)
    
    model = DDP(RNN())
    
else:
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH,
                                               shuffle=True)
    model = RNN()


# In[ ]:


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    # Train the model
    total_step = len(train_loader)
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size)
            labels = labels

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, EPOCHS, loss.item()))


def test():
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size)
            labels = labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 


# In[ ]:


with torch.autograd.profiler.profile() as prof:
    train()


# In[ ]:


import helper
name = 'ptorch_RNN_{}batch_node{}of{}_4CPUs'.format(BATCH,RANK+1,NODES)
full_name = 'ptorch_csv/{}.csv'.format(name)

helper.save_to_csv(prof.key_averages(),full_name)


# In[ ]:




