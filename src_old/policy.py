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

"""THIS CODE IS AN ADAPTATION FROM https://github.com/bonniesjli/PPO-Reacher_UnityML/blob/master/model.py
"""
class PPOActor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_size)
        # self.std = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.mu(x))
        # std = F.softplus(self.std(x))
        # return mu, std
        return action
    
    
class PPOCritic(nn.Module):
    def __init__(self, state_size, value_size=1, hidden_size=64):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, value_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value
    
    
class PPOAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PPOAgent, self).__init__()
        self.actor = PPOActor(state_size, action_size, hidden_size)
        self.critic = PPOCritic(state_size, hidden_size)
        
        # NOTE: WHAT IS THIS?
        self.log_std = nn.Parameter(torch.zeros(action_size))
        
    def forward(self, state):
        value = self.critic(state)
        
        action = self.actor(state)
        std   = self.log_std.exp().expand_as(mu)
        dist  = torch.distributions.Normal(action, std)
        
        return dist, value
    
    def act(self, states):
        """Given state as per current policy model, returns action, log probabilities and estimated state values"""
        dist, values = self.forward(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=1, keepdim=True)

        return actions, log_probs, values