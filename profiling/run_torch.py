import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import models_torch as mod
import funs_torch as f

numf = 32
model, train_dataset = mod.conv1d(numf)

batch = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch,
                                           shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 1
with torch.autograd.profiler.profile() as prof:
    f.train(model, train_loader, epochs, criterion, optimizer)