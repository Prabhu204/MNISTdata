"""
author: Prabhu
"""

#creating a neural network in pytorch

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets


def create_nn(batch_size = 100, learning_rate = 0.1, epochs= 20, log_interval = 10):

    train_loader = torch.utils.data.DataLoader(datasets.MNNIST())
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28*28, 200)
            self.fc2 = nn.Linear(200,200)
            self.fc3 = nn.Linear(200,10)

        # we have to define how data flows through out network

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.fc3(x)
            return F.log_softmax(x)

    net = Net()
    print(net)


    # training the network
    # Next we have to setup an optimizer and a loss criterion:
    # crate a stochastic gradient decent optimizer
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum= 0.9)
    # crate a loss function
    criterion = nn.NLLLoss()
    # run the main training loop

    for epoch in range( epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target) # converting data into Pytorch variables
            # resize data from (batch_size,1,28,28) to (batch_size, 28*28)
            data = data.view(-1,28*28)  # applying reshape to the function for single dimension
            optimizer.zero_grad()   # to reset all gradients in the model
            # net_out = net(data)  #
            # loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
        if batch_idx % log_interval ==0:
            print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data),len(train_loader.dataset),
            100. *batch_idx/len(train_loader), loss.data[0]))






