import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from base import BaseModel


class CNN_Simple(BaseModel):
    def __init__(self, num_classes, drop_prob=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return F.log_softmax(x, dim=1)

class CNN_Complex(BaseModel):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 4, 5)
        self.max_pool_1 = nn.MaxPool2d(2)

        self.conv_2 = nn.Conv2d(4, 16, 5)
        self.max_pool_2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16*4*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.max_pool_1(x)

        x = F.relu(self.conv_2(x))
        x = self.max_pool_2(x)

        x = x.view(-1, 16*4*4)

        x = F.relu(self.fc1(x))
        x = F.selu(self.fc2(x))
        
        out = F.log_softmax(self.fc3(x), dim=1)

        return out


class Alexnet(BaseModel):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.alexnet(num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
