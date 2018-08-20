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
from torchvision import datasets


def create_nn(batch_size = 200, learning_rate = 0.1, epochs= 5, log_interval = 10):
    # loading the data using datasets
    # transforms.Compose clubs all the transforms provided to it and are applied to the input one by one.
    # transforms.ToTensor converts image into pytorch tensor.
    # transforms.Normalize is just input data scaling(mean and std) and these values are predefined and changing these values not advised.
    train_loader = DataLoader(datasets.MNIST('../Data',
                                             train= True,
                                             download= True,
                                             transform = transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.1307,),(0.3081,))])),
                                             batch_size = batch_size,
                                             shuffle = True
                             )
    test_loader = DataLoader(datasets.MNIST('../Data',
                                            train= False,
                                            download= True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,),(0.13081,))])),
                                            batch_size = batch_size,
                                            shuffle = True
                            )

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28*28, 200) # input 28*28 output 200
            self.fc2 = nn.Linear(200,200) # input 200 output 200
            self.fc3 = nn.Linear(200,10)  # input 200 output 10

        # we have to define how data flows through out network

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)

    net = Net()
    print(net)

    # training the network
    # Next we have to setup an optimizer and a loss criterion:
    # crate a stochastic gradient decent optimizer and with parameters() supplies all values to networks
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum= 0.9)
    # crate a loss function
    criterion = nn.NLLLoss()
    # run the main training loop

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = Variable(data), Variable(target) # converting data into Pytorch variables
            # resize data from (batch_size,1,28,28) to (batch_size, 28*28)
            data = data.view(-1,28*28)  # applying reshape to the function for single dimension
            optimizer.zero_grad()       # to reset all gradients in the model
            net_out = net(data)         # supplying data into the model, it calls forward method() in the class
                                        # and then net_out holds softmax output for the given data batch
            loss = criterion(net_out, target)
            loss.backward()    # applying back propagation on the networks. Scalar variables do not require arguments
            # telling pytorch to to execute a gradient decent step based on the gradients
            # calculated during back propagation
            optimizer.step()
            if batch_idx % log_interval ==0:
                print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data),len(train_loader.dataset),
                100. *batch_idx/len(train_loader), loss.data[0]))


if __name__ == '__main__':
    create_nn()



