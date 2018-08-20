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





'''




