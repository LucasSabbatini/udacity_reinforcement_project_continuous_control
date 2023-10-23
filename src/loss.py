from conf import *
import torch
import torch.nn as nn

def loss_clipped_surrogate(agent, states, actions, log_probs_old, returns, advantages, sgd_epochs=4):
    """ Performs a learning step given a batch of experiences
    
    Remmeber: in the PPO algorithm, we perform SGD_episodes (usually 4) weights update steps per batch
    using the proximal policy ratio clipped objective function
    """        


    dist, values = agent.model(states)
    
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
    return policy_loss, value_loss
            
            
from trainer import *
import torch.optim as optim
def train(env, brain_name, agent, num_agents, n_episodes, max_t):
    print(f"Starting training...")
    env.info = env.reset(train_mode = True)[brain_name]
    all_scores = []
    all_scores_window = deque(maxlen=100)
    
    optimizer = optim.Adam(agent.model.parameters(), lr=LR, eps=EPSILON)
        
    for i_episode in range(n_episodes):
        # Each iteration, N parallel actors collect T time steps of data
        rollout, returns, episode_rewards, _ = collect_trajectories(env, brain_name, agent, max_t, num_agents)
        # print(f"Rollout: {len(rollout)}. Returns: {returns.shape}. Episode_rewards: {len(episode_rewards)}")
        
        states, actions, log_probs_old, returns, advantages = calculate_advantages(rollout, returns, num_agents)
        # print(f"States: {states.shape}. Actions: {actions.shape}. Log_probs_old: {log_probs_old.shape}. Returns: {returns.shape}. Advantages: {advantages.shape}")
        agent.learn(states, actions, log_probs_old, returns, advantages)
        for i in range(SGD_EPOCHS):
            policy_loss, value_loss = loss_clipped_surrogate(agent, states, actions, log_probs_old, returns, advantages, sgd_epochs=4)
            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(agent.model.parameters(), GRADIENT_CLIP)
            optimizer.step() 
        
        test_mean_reward = test_agent(env, agent, brain_name)

        all_scores.append(test_mean_reward)
        all_scores_window.append(test_mean_reward)

        if np.mean(all_scores_window) > 30.0:
            torch.save(agent.model.state_dict(), f"ppo_checkpoint.pth")
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(all_scores_window)))
            break       
        
        print('Episode {}, Total score this episode: {}, Last {} average: {}'.format(i_episode + 1, test_mean_reward, min(i_episode + 1, 100), np.mean(all_scores_window)) )