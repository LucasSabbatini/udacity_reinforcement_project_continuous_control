import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # What does the following lines do? 
        mean = torch.tanh(self.fc3(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std)) # get the sdistribution of the action so we
        return dist


import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim=(33), action_size=4):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.out = nn.Tanh() # Tanh already squeezes output to [-1, 1]

    def forward(self, state):
        x = F.Relu(self.fc1(state))
        x = F.Relu(self.fc2(x))
        # x = F.Tanh(self.fc3(x))
        x = self.out(self.fc3(x))
        return x        
        
from utils import device
# run your own policy!
policy=Policy().to(device)