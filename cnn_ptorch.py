#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F


# In[ ]:
import sys

BATCH = int(sys.argv[1])
EPOCHS = int(sys.argv[2])

RANK = int(sys.argv[3])
NODES = int(sys.argv[4])

print( 'Batch = {}, Epochs = {}, Rank = {}, Nodes = {}'.format(BATCH, EPOCHS, RANK, NODES) )

sequence_length = 28
input_size = 28

# In[ ]:


def conv_size_out(size_in, kern, stride):
    pad = 0
    size_out = (size_in + 2*pad - (kern - 1) - 1)/stride +1
    return size_out

def avg_size_out(size_in, kern, stride):
    pad = 0
    size_out = (size_in + 2*pad - kern)/stride +1
    return size_out
    
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        
        CONV_KERNEL = 5
        CONV_STRIDE = 1

        AVG_KERNEL = 2
        AVG_STRIDE = 2
        
        L1 = avg_size_out(conv_size_out(28, CONV_KERNEL, CONV_STRIDE), 
                     AVG_KERNEL, AVG_STRIDE)
        
        L2 = avg_size_out(conv_size_out(L1, CONV_KERNEL, CONV_STRIDE), 
                     AVG_KERNEL, AVG_STRIDE)
        
        LINEAR_IN = 16*(int(L2) ** 2)
        
        self.conv1 = nn.Conv2d(
            in_channels = 1, out_channels = 6,
            kernel_size = CONV_KERNEL, stride = CONV_STRIDE
        )
        
        self.conv2 = nn.Conv2d(
            in_channels = 6, out_channels = 16,
            kernel_size = CONV_KERNEL, stride = CONV_STRIDE
        )
        
        self.pool = nn.AvgPool2d(
            kernel_size = AVG_KERNEL, stride = AVG_STRIDE
        )
        
        self.flat = nn.Flatten()
        
        self.fc1 = nn.Linear(
            in_features = LINEAR_IN,
            out_features = 120
        )
        
        self.fc2 = nn.Linear(
            in_features = 120,
            out_features = 84
        )
        
        self.fc3 = nn.Linear(
            in_features = 84,
            out_features = 10
        )

    def forward(self, img):
        out = torch.tanh(self.conv1(img))
        out = self.pool(out)
        
        out = torch.tanh(self.conv2(out))
        out = self.pool(out)
        
        out = self.flat(out)
        
        out = torch.tanh(self.fc1(out))
        out = self.flat(out)
        
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)
        
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
    
    model = DDP(Cnn())
    
else:
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH,
                                           shuffle=True)
    model = Cnn()


# In[ ]:


# %load_ext tensorboard

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('logs/fit/torch_cnn_64_n1of1_cpu4')

# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# writer.add_graph(model, images)
# writer.close()


# In[ ]:


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def train():
    # Train the model
    total_step = len(train_loader)
    print(total_step)
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            labels = labels

            # Forward pass
            outputs = model(images)            
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#             if (i+1) % 100 == 0:
#                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                        .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, EPOCHS, loss.item()))

def test():
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            labels = labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 


# In[ ]:


import helper


# In[ ]:


with torch.autograd.profiler.profile() as prof:
    train()


# In[ ]:


name = 'ptorch_cnn_{}batch_node{}of{}_4CPUs'.format(BATCH,RANK+1,NODES)
full_name = 'ptorch_csv/{}.csv'.format(name)

helper.save_to_csv(prof.key_averages(),full_name)


# In[ ]:


# prof.key_averages()[0].cpu_time
# prof.key_averages()[0].cpu_time_str


# In[ ]:


# from csvsort import csvsort
# csvsort('ptorch_csv/{}.csv'.format(full_name), [0])


# In[ ]:




