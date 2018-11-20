import torch
from torchvision import datasets, transforms
import os
import torchvision
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

data_train = datasets.MNIST(root='../../data',
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root='../../data',
                            transform=transform,
                            train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                                shuffle = True,
                                                 num_workers=2)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = 64,
                                               shuffle = True,
                                                num_workers=2)

print(len(data_train))
images, labels = next(iter(data_loader_train))

# The shape of image is 64*1*28*28
print(images.shape)
print(labels)
print(len(labels))

# make_grid make a grid of images.
# The shape of input tensor is (B * C * H * W)
# B - batch_size, C - channel, H - Height, W - Weight
img = torchvision.utils.make_grid(images)

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Activation function: ReLU
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        # Fully-connected layer
        # https://stackoverflow.com/questions/44193270/how-to-calculate-the-output-size-after-convolving-and-pooling-to-the-input-image
        self.fc = torch.nn.Sequential(torch.nn.Linear(14*14*128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))
    def forward(self, x):
        x = self.conv1(x)
        # The output size after conv1 should be 14*14*128 for one image
        x = x.view(-1, 14*14*128)
        x = self.fc(x)
        return x

model = Net()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 1
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-"*10)
    for data in data_loader_train:
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _,pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(data_train),
                                                                                      100*running_correct/len(data_train),
                                                                                      100*testing_correct/len(data_test)))
torch.save(model.state_dict(), "model_parameter.pkl")