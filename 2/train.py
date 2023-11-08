from model import Model, train
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim


if __name__ == '__main__':
    model_dir = "model"  
    model_name = "lenet-mnist.pkl"
    batch_size = 256
    epoches = 25
    train_dataset = mnist.MNIST(root='.', train=True, transform=ToTensor(), download= True)
    test_dataset = mnist.MNIST(root='.', train=False, transform=ToTensor(), download= True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    net = Model()
    optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=1e-5)
    creterion = nn.CrossEntropyLoss()
    train(net, optimizer, creterion, train_loader, test_loader, epoches)

    #  save model
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    torch.save(net, model_dir + "/" + model_name)
    print("model saved")
