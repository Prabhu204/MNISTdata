"""
author: Prabhu
"""
'''
 *******A convolutional neuralnet**********
 

convolution:
    Convolution is the first filter applied as part of the feature engineering step.
    An application of a filter to a image. We pass over a mini image, usually called a kernel and outputting 
    the filtered subset of our image.

involved parameters:
    Kernel Size : The size of the filter
    Kernel Type: The values of the actual filter. Some examples include identity, edge detection and sharpen. 
    Stride : The rate at which the kernel passes over the edges of the image.
    Padding : We can add layers of 0's to the outside of the image in order to make sure that kernel kernel properly pass 
    over the edges of the image.
    Output layers : How many different layers are applied to the image.

The output of the convolution process is called the "convolved feature" or "feature map."

ReLU:
    It converts all negative pixel values to 0. The function itself is output = Max(0, input). Nonlinear functions: tanh or
    softmax. Default: ReLu

Max Pooling:
    We passover sections of our image and pool them into the highest value in the section. Depending on the size of the 
    pool, this can greatly reduce the size of the feature set that we pass into the neural net work. 

Pooling parameters :  Sum pooling or Average pooling. Default Max pooling

Fully connected Layer: It is similar to standard neural network

'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import  torchvision.transforms as transforms
from src.Evaluations import get_metrics
import csv

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/prabhu/MNISTdata/Data', train = True, download = True,
                                           transform = transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Normalize((0.1307,),(0.3081,))
                                                                           ])), batch_size = 100, shuffle= True
                                         )

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/prabhu/MNISTdata/Data', train= False,
                                                         transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))
                                                                                       ])),batch_size= 100,shuffle = True
                                          )




seed = 50 # A standard random seed for reproducible result.
np.random.seed(seed)
torch.manual_seed(seed)

# the compose function allows you for multiple transforms
# transforms.ToTensor() converts our PILImage to a tensor of shape(C x H x W) in the range [0,1]
# transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R,G,B)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # input chanel is based on image color i.e gray scale = 1 or color RGB = 3, kernel_size is your preference.
        self.conv1 = nn.Conv2d(1, 10,kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=4)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=2)
        # self.conv3_drop = nn.Dropout2d() drop out usally applies after fully connected layer
        self.fc1 = nn.Linear(80, 40)
        self.fc2 = nn.Linear(40,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(-1, 80)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


'''
Kernel size: 
    Remember, deeper networks is always better, at the cost of more data and increased complexity of learning.
    Minibatch size is usually set of few hundreds.
    You should initially use fewer filters and gradually increase and monitor the error rate to see how it is varying.
    Very small filter sizes will capture very fine details of the image. On the other hand having a bigger filter size will 
    leave out minute details in the image. 
    However conventional kernel size's are 3x3, 5x5 and 7x7.
    
Dropout layer:
    Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex 
    co-adaptations on training data
'''
# for initiating network using forward method
# net = Net()
# why do wqe need momentum? which helps accelerate gradients vectors in the right directions, thus leading
# to faster converging. https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
# what is SGD?

model = Net()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum= 0.9)
criterion = nn.CrossEntropyLoss()

def train(train_loader, model, optimizer,  epoch, log_interval = 10):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # to compute weight values
        optimizer.step() # to update weight values
        if batch_idx % log_interval  == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset),
                                                                           100. * batch_idx/len(train_loader), loss.item()))




def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            print(output)
            test_loss += criterion(output, target) # to sum up batch loss
            pred = output.max(1, keepdim = True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. *correct/ len(test_loader.dataset)))


    # names = ['True Label', 'Predicted Label', 'Content']
    # with open('src/Result', 'w') as csv_file:
    #     data_writer = csv.DictWriter(csv_file, fieldnames=names, quoting=csv.QUOTE_NONNUMERIC)
    #     for i, j, k in zip(target, pred, data):
    #         data_writer.writerow({
    #             'True Label': i+1, 'Predicted Label': j+1, 'Content': k})
    # test_metrics = get_metrics(target, pred, list_metrics=['Accuracy', 'Loss','Confusion_matrics'])
    # print("Prediction:\n  Accuracy: {} Loss: {} \nConfusion matrix: \n{}".format(
    #     test_metrics['Accuracy'], test_metrics['Loss'], test_metrics['Confusion_matrix']))

for epoch in range(1, 10):
    print(train(train_loader= train_loader,model = model, optimizer= optimizer,  epoch= epoch, log_interval = 10))



print(test(model= model, test_loader= test_loader))




