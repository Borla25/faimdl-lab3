import torch
from torch import nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 224 -> pool -> 112 -> pool -> 56 -> pool -> 28 -> pool -> 14
        # Con 4 pool layer, la dimensione diventa 14x14
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 200)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 112x112
        x = self.pool(F.relu(self.conv2(x))) # 56x56
        x = self.pool(F.relu(self.conv3(x))) # 28x28
        x = self.pool(x) # Un ultimo pool per arrivare a 14x14
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x