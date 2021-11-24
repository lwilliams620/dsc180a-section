###### import packags
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


# define quantization function
def binary_quantization(x):
    # x_back is the actual tensor for gradient computation
    # while sign(x) is the forwarded tensor
    # detach() can eliminate the gradient
    x_back = torch.clamp(x, -1, 1)
    return (torch.sign(x)*0.01 - x_back).detach() + x_back


# define binary convolutional layer
class BinConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BinConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                        bias)

    def forward(self, x):
        weight = binary_quantization(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# define binary fc layer
class BinLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BinLinear, self).__init__(in_features, out_features, bias=True)

    def forward(self, x):
        weight = binary_quantization(self.weight)
        return F.linear(x, weight, self.bias)


class ShiftBatchNorm(nn.BatchNorm1d):
    def __init__(self, channel, *args):
        super(ShiftBatchNorm, self).__init__(channel, *args)

    def round_pass(self, x):
        # to make rounded tensor still have gradient
        return (x.round() - x) + x

    def LogAP2(self, x, eps=0.0):
        # implement Most Significant Bit according to the paper
        return torch.sign(x) * 2 ** (torch.round(torch.log2(x.abs()+eps)))

    def get_var(self, x, mean, channel_dim):
        # implement variance with LogAP2()
        centerd_mean = x - mean.reshape(1, channel_dim)
        variance = centerd_mean * self.LogAP2(centerd_mean, eps=0.001)
        return variance.mean([0])

    def forward(self, input):
        # determine the average factor and channel dimension
        exponential_average_factor = self.momentum
        channel_dim = input.shape[1]
        # calculate running estimates if the model is in training
        if self.training:
            mean = input.mean([0])
            var = self.get_var(input, mean, channel_dim)

            # Update the running stats in Batch Normalization Layer
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var + (1 - exponential_average_factor) * self.running_var
        # In testing the model directly uses the running stats
        else:
            mean = self.running_mean
            var = self.running_var

        # calculate the normalization factor
        div = self.LogAP2(1 / (torch.sqrt(var.reshape(1, channel_dim) + self.eps)))
        input = (input - mean.reshape(1, channel_dim)) * div  # normalize

        if self.affine:
            # affine transformation, note that the weights are converted to LogAP2
            weight = (self.weight.reshape(1, channel_dim))
            input = input * weight + self.bias.reshape(1, channel_dim)
        return input


class TinyModel(nn.Module):

    def __init__(self):
        super().__init__()
        # define network structures
        self.conv1 = BinLinear(28*28, 2048)
        self.bn1 = ShiftBatchNorm(2048)
        self.conv2 = BinLinear(2048, 2048)
        self.bn2 = ShiftBatchNorm(2048)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        # Standard Conv-BN-Act block, note that the
        # activation is not binarized for the first layer
        # according to Section 2.6 in the paper.
        x = x.view(x.size(0), -1)
        x = self.conv1(x)
        x = self.bn1(x)
        # Binarize activation
        x = binary_quantization(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = binary_quantization(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    """
    Training function for binary model
    :param model: the binary model that needs to be optimized
    :param device: usually GPU with cuda
    :param train_loader: data loader of training dataset
    :param optimizer: Adam optimizer
    :param epoch: current training epoch
    :return: trained model
    """
    model.train()
    # start training iterating
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # forward, compute loss, backward, update loop
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return


def test(model, device, test_loader):
    """
    Test function for binary model
    :param model: the binary model that needs to be tested
    :param device: usually GPU with cuda
    :param test_loader: data loader of testing dataset
    :return: 
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # no gradient signal during testing
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# set the random seed for reproducibility
seed = 1001
torch.manual_seed(seed)

device = torch.device("cuda")

train_kwargs = {'batch_size': 128}
test_kwargs = {'batch_size': 100}

cuda_kwargs = {'num_workers': 1,
               'pin_memory': True,
               'shuffle': True}
# --------------------------Preparing Dataset ----------------------
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('data', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('data', train=False,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
# -------------------------------- End ------------------------------

# prepare model
model = TinyModel()
model.cuda()           # we use cuda default
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# training for 30 epochs
epochs = 100
scheduler = CosineAnnealingLR(optimizer, T_max=epochs,)

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
