import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 3)
        self.fc1 = nn.Linear(32 * 23 * 17, 30)
        self.fc2 = nn.Linear(30, 4)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 23 * 17)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        m = torch.tensor([[640, 480, 640, 480]]).type(torch.DoubleTensor).cuda()
        x = x * m
        return x

     def sigmoid(self, z):
        return 1/(1+torch.exp(-z))
