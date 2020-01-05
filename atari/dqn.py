import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    
    
    def __init__(self,in_size,out_size,hidden_size=32):
        super().__init__()
        self.in_size = in_size
        self.conv1 = nn.Conv2d(4,4,kernel_size = 8, stride = 4, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        
        self.fc1 = nn.Linear(in_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,out_size)
        
    def forward(self, x):
        
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1,self.in_size)
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
