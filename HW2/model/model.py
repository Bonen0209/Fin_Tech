import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class DNN(BaseModel):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
