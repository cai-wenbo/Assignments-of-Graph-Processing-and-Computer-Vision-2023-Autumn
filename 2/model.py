from torch.nn import Module
from torch import nn
import numpy as np
import torch

#  Lenet model
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim = 1)
            

    def forward(self, x):
        y = self.conv1(x)
        y = self.tanh1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.tanh2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.tanh3(y)
        y = self.fc2(y)
        y = self.tanh4(y)
        y = self.fc3(y)
        y = self.softmax(y)
        return y

#  training function
def train(net, optimizer, creterion, train_loader, test_loader, epoches):
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)


    for epoch in range(epoches):
        net.train()
        train_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = creterion(outputs, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= train_size
        #  print(train_loss)

        all_correct_num = 0
        test_loss = 0.0
        correct = 0
        net.eval()
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                outputs = net(inputs)

                test_loss += creterion(outputs, labels).item()

                predicted = torch.argmax(outputs, dim=-1)

                correct += (predicted == labels).sum().item()
                

            test_loss /= test_size
            test_acc = correct / test_size


        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.4f}')
        #  print('accuracy: {:.3f}'.format(acc), flush=True)
        #  if not os.path.isdir("models"):
        #      os.mkdir("models")
        #  torch.save(model, 'models/mnist_{:.3f}.pkl'.format(acc))
        #  if np.abs(acc - prev_acc) < 1e-4:
        #      break
    print("Model finished training")
