from torch.nn import Module
from torch import nn
import numpy as np
import torch

#  VGG19 model
class VGG19(Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu8 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(2)

        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu12 = nn.ReLU()

        self.pool4 = nn.MaxPool2d(2)

        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu14 = nn.ReLU()
        self.conv15 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu16 = nn.ReLU()

        self.pool5 = nn.MaxPool2d(2)

        self.faltten = nn.Flatten()

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu17 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu18 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 1000)
        self.softmax = nn.Softmax()
            

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool1(y)

        y = self.conv3(y)
        y = self.relu3(y)
        y = self.conv4(y)
        y = self.relu4(y)
        y = self.pool2(y)

        y = self.conv5(y)
        y = self.relu5(y)
        y = self.conv6(y)
        y = self.relu6(y)
        y = self.conv7(y)
        y = self.relu7(y)
        y = self.conv8(y)
        y = self.relu8(y)
        y = self.pool3(y)

        y = self.conv9(y)
        y = self.relu9(y)
        y = self.conv10(y)
        y = self.relu10(y)
        y = self.conv11(y)
        y = self.relu11(y)
        y = self.conv12(y)
        y = self.relu12(y)
        y = self.pool4(y)

        y = self.conv13(y)
        y = self.relu13(y)
        y = self.conv14(y)
        y = self.relu14(y)
        y = self.conv15(y)
        y = self.relu15(y)
        y = self.conv16(y)
        y = self.relu16(y)
        y = self.pool5(y)

        y = self.faltten(y)
        y = self.fc1(y)
        y = self.relu17(y)
        y = self.fc2(y)
        y = self.relu18(y)
        y = self.fc3(y)
        y = self.softmax(y)
        return y
