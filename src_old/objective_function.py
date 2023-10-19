import numpy as np 
from pprint import pprint
import torch
from utils import device
    
def clipped_surrogate_lucas(policy, states, actions, rewards, discount=0.995, epsilon=0.2, beta=0.01):
    """THIS IS THE OBJECTIVE FUNCTION FOR THE PPO ALGORITHM

    Args:
        policy (_type_): _description_
        states (_type_): _description_
        actions (_type_): _description_
        rewards (_type_): _description_
        discount (float, optional): _description_. Defaults to 0.995.
        epsilon (float, optional): _description_. Defaults to 0.2.
        beta (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
        
    TODO: ADAPT TO MEET REACHER OBSERVATIONAL AND ACTION STATES
    """
    
    # NOTE: THIS SHOULD STILL HAPPEN
    # CONVERT TO FUTURE REWARDS AND DISCOUNT (SHORT VERSION)
    steps = len(states)
    discounts = np.asarray([discount]*len(rewards))**np.asarray(list(range(steps)))
    future_rewards = []
    rewards_array = np.asarray(rewards)
    for i in range(len(rewards)):
        future_rewards.append(sum(rewards_array[i:]*discounts[:steps-i][:, np.newaxis])) # FUTURE REWARD IS EQUIVALENT TO THE ADVENTAGE
    
    # NOTE: THIS SHOULD STILL HAPPEN
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    future_rewards = torch.tensor(future_rewards, dtype=torch.float, device=device, requires_grad=True)
    
    # NOTE: THIS SHOULD STILL HAPPEN
    # NORMALIZATION
    mean = torch.mean(future_rewards, dim=1, keepdim=True)
    std = torch.std(future_rewards, dim=1, keepdim=True)
    future_rewards = (future_rewards - mean)/(std + 1.0e-10)
    
    # NOTE: THIS NEEDS TO CHANGE: THE POLICY OUTPUT IS NOT THE PROBS FOR ACTIONS NOW, BUT THE VALUE FOR THE ACTION ITSELF
    # CONVERT PROBS FOR LEFT ACTIONS -> P(LEFT) = 1 - P(RIGHT)
    right = 4
    old_probs = torch.where(actions==right, old_probs, 1.0-old_probs) # means that wherever it is 4, choose from old_probs, otherwise choose from 1.0-old_probs
    
    # NEW PROBABILITIES TO GET THE RATIO BETWEEN THE NEW AND THE OLD POLICIES
    # new_probs = pong_utils.states_to_prob(policy, states)
    # TODO: GET THE NEW VALUES FROM THE POLICY
    new_probs = torch.where(actions==right, new_probs, 1.0-new_probs) # also convert probs for left actions
    ratios = new_probs/old_probs
    clip = torch.clamp(ratios, 1-epsilon, 1+epsilon) # never let the ratio be larger than 1+-epsilon
    clipped_surrogate = torch.min(ratios*future_rewards, clip*future_rewards)

    # ENTROPY TERM =-------------> WHAT IS THIS AND WHY??
    entropy = -(new_probs*torch.log(old_probs+1.e-10) + (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
    
    return torch.mean(clipped_surrogate + beta*entropy)