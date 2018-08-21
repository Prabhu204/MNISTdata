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
import torchvision
import  torchvision.transforms as transforms


seed = 50                      # A standard random seed for reproducible result.
np.random.seed(seed)
torch.manual_seed(seed)

# the compose function allows you for mutliple transforms
# transforms.ToTensor() converts our PILImage to a tensor of shape(C x H x W) in the range [0,1]
# transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R,G,B)

class Net(nn.Module):
    def ___init__(self):
        super(Net,self).__init__()
        # input chanel is based on image color i.e gray scale = 1 or color RGB = 3, kernel_size is your preference.
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,100)
        self.fc2 = nn.Linear(100,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool1d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training= self.training)
        x = self.fc2(2)
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
