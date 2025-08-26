import torch
import torch.nn as nn
from torchvision.models import resnet18

class CIFAR10Net(torch.nn.Module):
  def __init__(self, cifar100=False):

    super().__init__()
    self._c1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self._b1 = torch.nn.BatchNorm2d(self._c1.out_channels)
    self._c2 = torch.nn.Conv2d(self._c1.out_channels, 64, kernel_size=3, padding=1)
    self._b2 = torch.nn.BatchNorm2d(self._c2.out_channels)
    self._m1 = torch.nn.MaxPool2d(2)
    self._d1 = torch.nn.Dropout(p=0.25)
    self._c3 = torch.nn.Conv2d(self._c2.out_channels, 128, kernel_size=3, padding=1)
    self._b3 = torch.nn.BatchNorm2d(self._c3.out_channels)
    self._c4 = torch.nn.Conv2d(self._c3.out_channels, 128, kernel_size=3, padding=1)
    self._b4 = torch.nn.BatchNorm2d(self._c4.out_channels)
    self._m2 = torch.nn.MaxPool2d(2)
    self._d2 = torch.nn.Dropout(p=0.25)
    self._d3 = torch.nn.Dropout(p=0.25)
    self._f1 = torch.nn.Linear(8192, 128)
    self._f2 = torch.nn.Linear(self._f1.out_features, 10)

  def forward(self, x):

    activation = torch.nn.functional.relu
    flatten    = lambda x: x.view(x.shape[0], -1)
    logsoftmax = torch.nn.functional.log_softmax
    x = self._c1(x)
    x = activation(x)
    x = self._b1(x)
    x = self._c2(x)
    x = activation(x)
    x = self._b2(x)
    x = self._m1(x)
    x = self._d1(x)
    x = self._c3(x)
    x = activation(x)
    x = self._b3(x)
    x = self._c4(x)
    x = activation(x)
    x = self._b4(x)
    x = self._m2(x)
    x = self._d2(x)
    x = flatten(x)
    x = self._f1(x)
    x = activation(x)
    x = self._d3(x)
    x = self._f2(x)
    x = logsoftmax(x, dim=1)
    return x

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet18 = resnet18()
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
            x = self.resnet18(x)
            return x

class FEMNISTNet(torch.nn.Module):

    def __init__(self):
        
        super().__init__()
     
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self, x):
        
        logsoftmax = torch.nn.functional.log_softmax
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
