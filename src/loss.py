from conf import *
import torch
import torch.nn as nn

def ppo_loss(model, states, actions, log_probs_old, returns, advantages):
    dist, values = model(states)
    
    log_probs = dist.log_prob(actions)
    log_probs = torch.sum(log_probs, dim=1, keepdim=True)
    entropy = dist.entropy().mean()
    
    # r(θ) =  π(a|s) / π_old(a|s)
    ratio = (log_probs - log_probs_old).exp()
    
    # Surrogate Objctive : L_CPI(θ) = r(θ) * A
    obj = ratio * advantages
    
    # clip ( r(θ), 1-Ɛ, 1+Ɛ )*A
    obj_clipped = ratio.clamp(1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON) * advantages
    
    # L_CLIP(θ) = E { min[ r(θ)A, clip ( r(θ), 1-Ɛ, 1+Ɛ )*A ] - β * KL }
    policy_loss = -torch.min(obj, obj_clipped).mean(0) - BETA * entropy.mean()
    
    # L_VF(θ) = ( V(s) - V_t )^2
    value_loss = 0.5 * (returns - values).pow(2).mean()
    
    return policy_loss + value_loss