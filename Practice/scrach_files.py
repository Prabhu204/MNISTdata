"""
author: Prabhu
email: prabhu.appalapuri@gmail.com
"""
"""
Linear Regression:
    y = Ax+B
"""
# For example, we have car company. If the car price is low, we sell more car. If the car price is
# high, we sell less car. This is the fact that we know and we have data set about this fact.
# The question is that what will be number of car sell if the car price is 100.

import numpy as np
import torchvision
import torch
from torch.autograd import Variable


# car_prices_array = [3,4,5,6,7,8,9]
# car_price_np = np.array(car_prices_array,dtype=np.float32)
# print(car_price_np)
# print(np.shape(car_price_np))
# car_price_np = car_price_np.reshape(-1,1)
# print(car_price_np)
# number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
# number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
# number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
#
# car_prices_tensor = variable(torch.from_numpy(car_price_np))
# number_of_car_sell_tensor = variable(torch.from_numpy(number_of_car_sell_np))
#
# # lets visualize our data
#
# import matplotlib.pyplot as plt
# plt.scatter(car_prices_array,number_of_car_sell_array)
# plt.xlabel("Car Price $")
# plt.ylabel("number of sellings")
# plt.title("car vs sell")
# plt.show()

# import torch.nn as nn
# import torch
# x = torch.Tensor(2,3)

# autograd in pycharm
# from torch.autograd import Variable
# from torch import autograd
#
# x = Variable(torch.ones(2,2)*2, requires_grad= True)
# print(x)
#
# z = 2*(x*x)+ 5*x
# print(z)
#
# z.backward(torch.ones(2,2))
# print(x.grad)

#creating a neural network in pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets




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
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum= 0.9)
# crate a loss function
criterion = nn.NLLLoss()
# run the main training loop

for epoch in range( epochs):





