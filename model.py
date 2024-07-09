import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config_dict


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5, padding=2)  # Reduced filters from 4 to 2
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)  # Reduced filters from 8 to 4
        self.fc1 = nn.Linear(4 * 5 * 5, 32)  # Adjust input features accordingly
        self.fc2 = nn.Linear(32, 16)  # Reduced nodes from 64 to 32, 32 to 16
        self.fc3 = nn.Linear(16, config_dict['num_classes'])

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 4 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

