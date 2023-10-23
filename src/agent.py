

# Agent and models
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from conf import *


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
    
class Critic(nn.Module):
    def __init__(self, state_size, value_size=1, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, value_size)
        
    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, std=0.0):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, 1, hidden_size)
        
        self.log_std = nn.Parameter(torch.ones(1, action_size)*std)
        
    def forward(self, states): # TODO: LEARN WHAT THE FUCK THIS DOES
        obs = torch.FloatTensor(states)
        
        # actor and critic outputs
        mu = self.actor(obs)
        values = self.critic(obs)
        
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        
        return dist, values

class Agent():
    def __init__(self, num_agents, state_size, action_size):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=EPSILON)
        self.model.train()
        
    def act(self, states):
        """Remember: states are state vectors for each agent
        It is used when collecting trajectories
        """
        dist, values = self.model(states) # pass the state trough the network and get a distribution over actions and the value of the state
        actions = dist.sample() # sample an action from the distribution
        log_probs = dist.log_prob(actions) # calculate the log probability of that action
        log_probs = log_probs.sum(-1).unsqueeze(-1) # sum the log probabilities of all actions taken (in case of multiple actions) and reshape to (batch_size, 1)
        return actions, log_probs, values
    

    def batcher(self, BATCH_SIZE, states, actions, log_probs_old, returns, advantages):
        """Convert trajectories into learning batches."""
        # for _ in range(states.size(0) // BATCH_SIZE):
        rand_ids = np.random.randint(0, states.size(0), BATCH_SIZE)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs_old[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]

    def loss(self):
        pass
    
    def learn(self, states, actions, log_probs_old, returns, advantages, sgd_epochs=4):
        """ Performs a learning step given a batch of experiences
        
        Remmeber: in the PPO algorithm, we perform SGD_episodes (usually 4) weights update steps per batch
        using the proximal policy ratio clipped objective function
        """        

        for _ in range(sgd_epochs):
            # for _ in range(states.size(0) // BATCH_SIZE):
                
            for sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages in self.batcher(BATCH_SIZE, states, actions, log_probs_old, returns, advantages):

                dist, values = self.model(sampled_states)
                
                log_probs = dist.log_prob(sampled_actions)
                log_probs = torch.sum(log_probs, dim=1, keepdim=True)
                entropy = dist.entropy().mean()
                
                # r(θ) =  π(a|s) / π_old(a|s)
                ratio = (log_probs - sampled_log_probs_old).exp()
                
                # Surrogate Objctive : L_CPI(θ) = r(θ) * A
                obj = ratio * sampled_advantages
                
                # clip ( r(θ), 1-Ɛ, 1+Ɛ )*A
                obj_clipped = ratio.clamp(1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON) * sampled_advantages
                
                # L_CLIP(θ) = E { min[ r(θ)A, clip ( r(θ), 1-Ɛ, 1+Ɛ )*A ] - β * KL }
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - BETA * entropy.mean()
                
                # L_VF(θ) = ( V(s) - V_t )^2
                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()
               

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                self.optimizer.step() 
