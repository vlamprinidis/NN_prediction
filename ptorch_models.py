import torch 
import torch.nn as nn
import torch.nn.functional as F

def convs(size_in, kern, stride):
    pad = 0
    dilation = 1
    return (size_in + 2*pad - dilation*(kern - 1) - 1) // stride + 1

def avgs(size_in, kern, stride):
    pad = 0
    return (size_in + 2*pad - kern) // stride + 1

def maxs(size_in, kern, stride):
    pad = 0
    dilation = 1
    return (size_in + 2*pad - dilation*(kern - 1) - 1) // stride + 1

# 28x28 input
class LeNet_1:        
    def create(self):
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels = 4, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels=4, out_channels = 12, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Flatten(),
            nn.Linear(in_features=12*4*4, out_features=10)
        )

        return self.model

# 32x32 input
class LeNet_5():
    def create(self):
        self.model = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
            
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10)
        )
        
        return self.model
    
# 224x224x3
class VGG_11:
    def create(self):
        def c(n):
            return convs(n,3,1)
        
        def m(n):
            return maxs(n,2,2)
        
        def mc(n):
            return m(c(n))
        
        l1=mc(224)
        l2=mc(l1)
        l3=c(l2)
        l4=mc(l3)
        l5=c(l4)
        l6=mc(l5)
        l7=c(l6)
        l8=mc(l7)

        self.model = nn.Sequential(
            #L1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #L2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #L3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            
            #L4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #L5
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            
            #L6
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #L7
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            
            #L8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #L9
            nn.Flatten(),
            nn.Linear(in_features=(l8**2)*512, out_features=4096),
            
            #L10
            nn.Linear(in_features=4096, out_features=4096),
            
            #L11
            nn.Linear(in_features=4096, out_features=1000)
        )
        
        return self.model

# 227x227x3
class AlexNet:
    def create(self):
        c = convs
        def m(n):
            return maxs(n,3,2)
        
        l1=m(c(227,11,4))
        l2=m(c(l1,5,1))
        l3=c(l2,3,1)
        l4=c(l3,1,1)
        l5=m(c(l4,1,1))
        
        self.model = nn.Sequential(
            #L1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #L2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #L3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(384),
            
            #L4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(384),
            
            #L5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #L6
            nn.Flatten(),
            nn.Linear(in_features=(l5**2)*256, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            
            #L7
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            
            #L8
            nn.Linear(in_features=4096, out_features=1000)
        )
        
        return self.model


    