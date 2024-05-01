#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:40:26 2024

@author: sirishant
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device initialized to: " + str(device))

# input_size = 784
# hidden_size = 500
num_class = 10
num_epochs = 5
batch_size = 100

"""
1000 images --> 10 batches
"""
lr = 0.001

# Download MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root = 'data',
                                           train = 'true',
                                           transform = transforms.ToTensor(),
                                           download = 'True')
test_dataset = torchvision.datasets.MNIST(root = 'data',
                                          train = 'False',
                                          transform = transforms.ToTensor(),
                                          download = 'True')

# Data loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = False)


# Network

######################### Your Code starts here #######################
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#################### Your code ends here##############################

total_step = len(train_loader)
lossval = []
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        #images = images.reshape(-1,28*28)
        images = images.to(device)
        labels = labels.to(device)
        ###################Your code starts here####################
        out = net(images)
        loss = criterion(out, labels)
        
        #Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100==0:
            print('Epoch = ', epoch, ' Itr = ',i,' Loss = ',loss.item())
            lossval.append(loss.item())
        ##################### Your code ends here ###################
        
correct = 0
total = 0
for images,labels in test_loader:
    #images = images.reshape(-1,28*28)
    images = images.to(device)
    labels = labels.to(device)
    out = net(images)
    _, pred = torch.max(out.data,1)
    total+=labels.size(0)
    correct += (pred==labels).sum().item()

Acc = correct/total*100
print("Accuracy of the system is ",Acc,"%") 