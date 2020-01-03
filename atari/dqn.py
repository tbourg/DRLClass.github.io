import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,in_size,out_size,hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(in_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,out_size)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x