import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# DCGAN generator class
# --------------------------------------------------------------------------


class DCGen(nn.Module):
    """
    The model here is basically ripped-off from
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    def __init__(self, input_dimensions, ft_multiplier=4):

        super(DCGen, self).__init__()



        self.input_dimension = input_dimensions

        self.main_net = nn.Sequential(

            # square image dimension: 1
            nn.ConvTranspose2d(input_dimensions, ft_multiplier * 4,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ft_multiplier * 4),

            # square image dimension: 4
            nn.ConvTranspose2d(ft_multiplier * 4, ft_multiplier * 2,
                               kernel_size=4, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ft_multiplier * 2),

            # square image dimension: 10
            nn.ConvTranspose2d(ft_multiplier * 2, ft_multiplier,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ft_multiplier),

            # square image dimension: 13
            nn.ConvTranspose2d(ft_multiplier, 1,
                               kernel_size=4, stride=2, bias=False),
            nn.Tanh()
            # square image dimension: 28

        )

    def forward(self, input_vector):

        # The input vector can have dimension (batch size, input dimension).
        # This line of code is to set up the input vector for convolution.
        input_vector = input_vector.view(-1, self.input_dimension, 1, 1)

        return self.main_net(input_vector)


# --------------------------------------------------------------------------
# DCGAN discriminator with mini-batch discrimination
# --------------------------------------------------------------------------

class DiscriminatorMBD(nn.Module):

    def __init__(self,  mbd_dim_b, mbd_dim_c): # TODO: b, c annoying

        super(DiscriminatorMBD, self).__init__()

        self.mbd_dim_b = mbd_dim_b
        self.mbd_dim_c = mbd_dim_c

        self.convolutional_layers = nn.Sequential(

            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.feature_scan_tensor = nn.Linear(40 * 4 * 4,
                                             mbd_dim_b * mbd_dim_c,
                                             bias=False)

        self.processing_layers = nn.Sequential(

            nn.Linear(40 * 4 * 4 + mbd_dim_b, 100),
            nn.ReLU(),
            nn.Dropout(p=.2),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, input_batch):

        features = self.convolutional_layers(input_batch)

        features_flat = features.view(-1, num_flat_features(features))

        batch_similarity = self.batch_discrimination(features_flat)

        out = torch.cat((features_flat, batch_similarity), dim=1)

        out = self.processing_layers(out)

        return out

    def batch_discrimination(self, features):

        batch_size = features.size()[0]

        layer = self.feature_scan_tensor(features)
        layer = layer.view(-1, self.mbd_dim_b, self.mbd_dim_c)

        distances = torch.zeros(batch_size, batch_size,
                                self.mbd_dim_b, self.mbd_dim_c)

        # TODO: is there a faster way to do this?
        for i in range(batch_size):
            for j in range(batch_size):

                if j > i:
                    distances[i][j] = (layer[i] - layer[j]).abs()

                # The distance tensor is symmetric in its first two indices
                elif j < i:
                    distances[i][j] = distances[j][i]

                # The remaining case of i = j is handled by
                # the zero initialization

        distances = distances.sum(dim=3)  # Sum the "rows"
        closeness = (-distances).exp()
        closeness = closeness.sum(dim=1)

        return closeness


#
# --------------------------------------------------------------------------
# Ancillary functions
# --------------------------------------------------------------------------
#


def num_flat_features(state):

    size = state.size()[1:]

    flat_features = 1

    for k in size:
        flat_features *= k

    return flat_features

# -------------------------------------------------------------------------- #
# OLD THINGS DEPRECATED MORE OR LESS
# -------------------------------------------------------------------------- #

#
# class GeneratorType1(nn.Module):
#     """ Network to generate MNIST-looking digits
#
#     This is an artificial neural network designed to reproduce 28 x 28
#     images that look like MNIST digits after weights and biases are
#     well-chosen.
#
#     The input is a k_0-dimensional vector.  It is desired that
#     different regions of the input space map to a wide and "unbiased" variety
#     of handwritten digits.
#
#     ARCHITECTURE:
#     """
#
#     def __init__(self, input_dimensions, error_scale=.03):
#         super(GeneratorType1, self).__init__()
#
#         self.linear_layers = nn.Sequential(
#
#             nn.Linear(input_dimensions,100),
#             nn.ReLU(),
#             nn.Dropout(p=.1),
#             nn.Linear(100,24*24*20),
#             nn.ReLU(),
#             nn.Dropout(p=.1),
#
#
#             #nn.Linear(80, 28*28)
#         )
#
#         self.conv_layer = nn.Sequential(
#
#             nn.ConvTranspose2d(20,1,5),
#             nn.Tanh()
#         )
#
#         self.input_dimension = input_dimensions
#         self.error_scale = error_scale
#
#
#     def forward(self, input_vector):
#
#         out = self.linear_layers(input_vector)
#
#         out = out + self.error_scale * out * torch.randn(out.size())
#
#         out = out.view(-1, 20, 24, 24)
#
#         out = self.conv_layer(out)
#
#         return out
#
#
# class GeneratorType2(nn.Module):
#     """ Network to generate MNIST-looking digits
#
#     This is an artificial neural network designed to reproduce 28 x 28
#     images that look like MNIST digits after weights and biases are
#     well-chosen.
#
#     The input is a k_0-dimensional vector.  It is desired that
#     different regions of the input space map to a wide and "unbiased" variety
#     of handwritten digits.
#
#     ARCHITECTURE:
#     """
#
#     def __init__(self, input_dimensions, sharpen=True):
#         super(GeneratorType2, self).__init__()
#
#         self.planning_layers = nn.Sequential(
#
#             #nn.Linear(input_dimensions, 100),
#             #nn.ReLU(),
#             nn.Linear(input_dimensions, 40 * 4 * 4),
#             nn.ReLU()
#         )
#
#         self.transpose_conv_1 = nn.Sequential(
#
#             nn.ConvTranspose2d(40, 20, 5),
#             nn.ReLU()
#         )
#
#         self.transpose_conv_2 = nn.Sequential(
#
#             nn.ConvTranspose2d(20, 1, 5),
#             nn.Tanh()
#         )
#
#         direct_connections = nn.Linear(28*28, 28*28)
#         direct_connections.weight = torch.nn.Parameter(torch.eye(28**2))
#         direct_connections.bias = torch.nn.Parameter(torch.zeros(28**2))
#
#         self.sharpening_layer = nn.Sequential(
#             direct_connections,
#             nn.Tanh()
#         )
#
#         self.input_dimension = input_dimensions
#         self.sharpen = sharpen
#
#     def forward(self, input_vector):
#
#         out = self.planning_layers(input_vector)
#
#         # We organize the output from the planning layers into 20 different
#         # 4x4 planes.
#         out = out.view(-1, 40, 4, 4)
#
#         out = nn.functional.interpolate(out, 8, mode="bilinear")
#
#         out = self.transpose_conv_1(out)
#
#         out = nn.functional.interpolate(out, 24, mode="bilinear")
#
#         out = self.transpose_conv_2(out)
#
#         if self.sharpen:
#             out = out.view(-1,1,28*28)
#             out = self.sharpening_layer(out)
#             out = out.view(-1,1,28,28)
#         # out = self.sharpening_layer(out)
#
#         return out
#
#
# class GeneratorType3(nn.Module):
#     """ Network to generate MNIST-looking digits
#
#     This is an artificial neural network designed to reproduce 28 x 28
#     images that look like MNIST digits after weights and biases are
#     well-chosen.
#
#     The input is a k_0-dimensional vector.  It is desired that
#     different regions of the input space map to a wide and "unbiased" variety
#     of handwritten digits.
#
#     ARCHITECTURE:
#     """
#
#     def __init__(self, input_dimensions):
#         super(GeneratorType3, self).__init__()
#
#         self.planning_layers = nn.Sequential(
#
#             #nn.Linear(input_dimensions, 100),
#             #nn.ReLU(),
#             nn.Linear(input_dimensions, 40 * 4 * 4),
#             nn.ReLU()
#         )
#
#         self.transpose_conv_1 = nn.Sequential(
#
#             nn.ConvTranspose2d(40, 20, 5),
#             nn.ReLU()
#         )
#
#         self.transpose_conv_2 = nn.Sequential(
#
#             nn.ConvTranspose2d(20, 1, 5),
#             nn.Tanh()
#         )
#
#         self.input_dimension = input_dimensions
#
#     def forward(self, input_vector):
#
#         out = self.planning_layers(input_vector)
#
#         # We organize the output from the planning layers into 20 different
#         # 4x4 planes.
#         out = out.view(-1, 40, 4, 4)
#
#         out = nn.functional.interpolate(out, 8, mode="bilinear")
#
#         out = self.transpose_conv_1(out)
#
#         # TODO: change second interpolation to a fully connected layer
#
#         # (batch, 20, 12, 12) -> (batch, 20, 24, 24)
#
#         out = nn.functional.interpolate(out, 24, mode="bilinear")
#
#         out = self.transpose_conv_2(out)
#
#         return out
#
#
#
# #                                                                            #
# # -------------------------------------------------------------------------- #
# #                                                                            #
#
#
# class Discriminator(nn.Module):
#
#     def __init__(self):
#
#         super(Discriminator, self).__init__()
#
#         self.convolutional_layers = nn.Sequential(
#
#             nn.Conv2d(1, 20, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(20, 40, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.processing_layers = nn.Sequential(
#
#             nn.Linear(40 * 4 * 4, 100),
#             nn.ReLU(),
#             nn.Dropout(p=.2),
#             nn.Linear(100, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input_image):
#
#         out = self.convolutional_layers(input_image)
#
#         out = out.view(-1, num_flat_features(out))
#
#         out = self.processing_layers(out)
#
#         return out
#
#
# #                                                                            #
# # -------------------------------------------------------------------------- #
# #                                                                            #
#
# class DiscriminatorMulti(nn.Module):
#
#     def __init__(self):
#
#         super(DiscriminatorMulti, self).__init__()
#
#         self.convolutional_layers = nn.Sequential(
#
#             nn.Conv2d(1, 20, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(20, 40, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.processing_layers = nn.Sequential(
#
#             nn.Linear(40 * 4 * 4 * 2, 100),
#             nn.ReLU(),
#             nn.Dropout(p=.2),
#             nn.Linear(100, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, image1, image2):
#
#         out1 = self.convolutional_layers(image1)
#         out2 = self.convolutional_layers(image2)
#
#         out1 = out1.view(-1, num_flat_features(out1))
#         out2 = out2.view(-1, num_flat_features(out2))
#
#         out = torch.cat((out1,out2), dim=1)
#
#         out = self.processing_layers(out)
#
#         return out
#
#
# #                                                                            #
# # -------------------------------------------------------------------------- #
# #                                                                            #
#
#
# class Identifier(nn.Module):
#
#     def __init__(self):
#
#         super(Identifier, self).__init__()
#
#         self.convolutional_layers = nn.Sequential(
#
#             nn.Conv2d(1, 20, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(20,40, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.fc_layers = nn.Sequential(
#
#             nn.Linear(40 * 4 * 4, 100),
#             nn.ReLU(),
#             nn.Linear(100, 10),
#             nn.LogSoftmax(1)
#         )
#
#     def forward(self, input_image):
#
#         out = self.convolutional_layers(input_image)
#         out = out.view(-1, num_flat_features(out))
#         out = self.fc_layers(out)
#
#         return out
#
#
#
#
#
#
#
#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                       download=True, transform=transform)
#
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=4,
#                                           shuffle=True,
#                                           num_workers=2)
#
#
# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=True, transform=transform)
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)
#
#
#
#
#
#
#
# #
#
# # ---------------------------------------------------------------- #
#
# #
#
# net = Identifier()
#
# criterion = nn.NLLLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
#
# for epoch in range(2):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs
#         inputs, labels = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#
# print('Finished Training')
#
#
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
#
#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# # functions to show an image
#
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
#
#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
#
#
# # make up some bs data
# img_batch = torch.rand(20,1,28,28)
# target = torch.empty(20,dtype=torch.long).random_(10)
#
# # Loss function
# lossfunc = nn.NLLLoss()
#
# # create network model
# net = Identifier()
#
# # create optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)
#
# # in your training loop:
# for i in range(5000):
#
#     optimizer.zero_grad()   # zero the gradient buffers
#
#     outputs = net(img_batch)
#
#     loss = lossfunc(outputs, target)
#
#     if i % 100 == 0:
#         print("loss:")
#         print(loss)
#         print("success rate:")
#         print(torch.eq(target, net(img_batch).argmax(1)).sum().item() / len(target))
#         print('\n\n\n')
#
#     loss.backward()
#     optimizer.step()    # Does the update
#
# print("success rate:")
#
# print(torch.eq(target, net(img_batch).argmax(1)).sum().item() / len(target))
#
#
# class Generator(nn.Module):
#
#     def __init__(
#             self,
#             layer_sizes,
#             intermediate_activation=nn.ReLU(),
#             final_activation=nn.Sigmoid()
#     ):
#
#         super(Generator, self).__init__()
#
#         self.layer_sizes = layer_sizes
#         self.num_layers = len(layer_sizes)
#
#         self.intermediate_activation = intermediate_activation
#         self.final_activation = final_activation
#
#         self.input_steps = []
#         for i in range(self.num_layers - 1):
#
#             num_last_layer = self.layer_sizes[i]
#             num_next_layer = self.layer_sizes[i+1]
#
#             self.input_steps.append(
#                 nn.Linear(num_last_layer,num_next_layer)
#             )
#
#     def forward(self, network_input):
#
#         vector = network_input
#
#         # We push the signal through each layer except the last,
#         # applying the function self.intermediate_activation after each step
#         for i in range(self.num_layers - 2):
#             vector = self.input_steps[i](vector)
#             vector = self.intermediate_activation(vector)
#
#         # Input to the final layer
#         vector = self.input_steps[self.num_layers - 2](vector)
#
#         # Activation from the final layer
#         return self.final_activation(vector)
#
#
# gen = Generator([2,3,3,3])
# print(gen)
#
#
# out = gen.forward(torch.tensor([.4,.2]))
#
# out[0].backward()



