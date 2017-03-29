import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

# ============== Basic autograd example 1 ================== #

x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

y = w * x + b

y.backward()

print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1

# ============== Basic autograd example 2 ================== #

# Tensors
x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

# Linear layer
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# build loss and optimiser
criterion = nn.MSELoss()
optimiser = torch.optim.SGD(linear.parameters(), lr=0.01)

# forward propagation
pred = linear(x)

# compute loss
loss = criterion(pred, y)
print('loss: ', loss.data[0])

# back propagation
loss.backward()

# print gradient
print('dL/dW: ', linear.weight.grad)
print('dL/dB: ', linear.bias.grad)

# 1 step optimiser (descent)
optimiser.step()

# print loss after optimisation
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimisation: ', loss.data[0])

# ============== Loading data from numpy ================== #
a = np.array([[1, 2], [3, 4]])
b = torch.from_numpy(a)  # numpy to torch
c = b.numpy()

print(c)

# ============== Implement input pipline ================== #
train_dataset = dsets.CIFAR10(root='../data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

# select one data pair
image, lable = train_dataset[0]
print(image.size())
print(lable)

# Data Loader (this provides queue and thread in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)

# queue and thred start to load dataset from files
data_iter = iter(train_loader)

# Mini batch images and lables
images, lables = data_iter.next()

# usage of data Loader
for image, lable in train_loader:
    pass

# ============== input pipline for custom dataset ================== #
# build custom dataset


class CustomDataset(data.Dataset):
    def __init__(self):
        # TODO
        # Initialise the file path or list of file names
        pass

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass

    def __len__(self):
        return 0

# prebuilt torch's data Loader


custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=100, shuffle=True,
                                           num_workers=2)


# ============== use pretrained models ================== #
# download and load pretrained resnet
resnet = torchvision.models.resnet18(pretrained=True)

# fine tune top layer of the model
for param in resnet.parameters():
    param.requires_grad = False

# Replace top layer for finetuning
resnet.fc = nn.Linear(resnet.fc.in_features, 100)

# test
image = Variable(torch.randn(10, 3, 256, 256))
outputs = resnet(images)
print(outputs.size())


# ============== save and load the model ================== #
# save and load entire model
torch.save(resnet, 'model.pkl')
model = torch.load('model.pkl')

# save and load only the model parameters
torch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(torch.load('params.pkl'))
