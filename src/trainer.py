# train the agent
import numpy as np
from collections import deque
import torch
from conf import *
from utils import test_agent


def collect_trajectories(env, brain_name, agent, max_t):
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations  
        
    rollout = []
    agents_rewards = np.zeros(num_agents)
    episode_rewards = []

    # Collecting trajectories
    for _ in range(max_t):
        actions, log_probs, values = agent.act(states)
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = np.array([1 if t else 0 for t in env_info.local_done])
        agents_rewards += rewards

        for j, done in enumerate(dones):
            if dones[j]:
                episode_rewards.append(agents_rewards[j])
                agents_rewards[j] = 0

        rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - dones])

        states = next_states

    pending_value = agent.model(states)[-1]
    returns = pending_value.detach()
    rollout.append([states, pending_value, None, None, None, None])
    
    return rollout, returns, episode_rewards, np.mean(episode_rewards)


def calculate_advantages(rollout, returns, num_agents):
    """ Given a rollout, calculates the advantages for each state
    """
    processed_rollout = [None] * (len(rollout) - 1)
    advantages = torch.Tensor(np.zeros((num_agents, 1)))

    for i in reversed(range(len(rollout) - 1)):
        states, value, actions, log_probs, rewards, dones = rollout[i]
        dones = torch.Tensor(dones).unsqueeze(1)
        rewards = torch.Tensor(rewards).unsqueeze(1)
        actions = torch.Tensor(actions)
        states = torch.Tensor(states)
        next_value = rollout[i + 1][1]
        
        # V(s) = r + γ * V(s')
        returns = rewards + GAMMA * dones * returns
        
        # L = r + γ*V(s') - V(s)
        td_error = rewards + GAMMA * dones * next_value.detach() - value.detach()
        
        advantages = advantages * TAU * GAMMA * dones + td_error
        processed_rollout[i] = [states, actions, log_probs, returns, advantages]

    states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
    advantages = (advantages - advantages.mean()) / advantages.std()
    
    
    return states, actions, log_probs_old, returns, advantages


def train(env, brain_name, agent, num_agents, n_episodes, max_t):
    print(f"Starting training...")
    env.info = env.reset(train_mode = True)[brain_name]
    all_scores = []
    all_scores_window = deque(maxlen=100)
        
    for i_episode in range(n_episodes):
        # Each iteration, N parallel actors collect T time steps of data
        rollout, returns, episode_rewards, _ = collect_trajectories(env, brain_name, agent, max_t, num_agents)
        # print(f"Rollout: {len(rollout)}. Returns: {returns.shape}. Episode_rewards: {len(episode_rewards)}")
        
        states, actions, log_probs_old, returns, advantages = calculate_advantages(rollout, returns, num_agents)
        # print(f"States: {states.shape}. Actions: {actions.shape}. Log_probs_old: {log_probs_old.shape}. Returns: {returns.shape}. Advantages: {advantages.shape}")
        agent.learn(states, actions, log_probs_old, returns, advantages)
        
        test_mean_reward = test_agent(env, agent, brain_name)

        all_scores.append(test_mean_reward)
        all_scores_window.append(test_mean_reward)

        if np.mean(all_scores_window) > 30.0:
            torch.save(agent.model.state_dict(), f"ppo_checkpoint.pth")
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(all_scores_window)))
            break       
        
        print('Episode {}, Total score this episode: {}, Last {} average: {}'.format(i_episode + 1, test_mean_reward, min(i_episode + 1, 100), np.mean(all_scores_window)) )
        
        

        
from loss import loss_clipped_surrogate
import torch.nn as nn
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
            policy_loss, value_loss = loss_clipped_surrogate(states, actions, log_probs_old, returns, advantages, sgd_epochs=4)
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