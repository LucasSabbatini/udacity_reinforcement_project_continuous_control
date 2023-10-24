# train the agent
import numpy as np
from collections import deque
import torch
from conf import *
from utils import test_agent
import os


def collect_trajectories(env, brain_name, agent, max_t):
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
        
    rollout = []
    agents_rewards = np.zeros(num_agents)
    episode_rewards = []

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
    returns = pending_value.detach() # Why is this called retuns? It's the value of the last state
    rollout.append([states, pending_value, None, None, None, None])
    
    return rollout, returns, episode_rewards, np.mean(episode_rewards)


def calculate_advantages(rollout, returns, num_agents):
    """ Given a rollout, calculates the advantages for each state """
    num_steps = len(rollout) - 1
    processed_rollout = [None] * num_steps
    advantages = torch.zeros((num_agents, 1))

    for i in reversed(range(num_steps)):
        states, value, actions, log_probs, rewards, dones = map(lambda x: torch.Tensor(x), rollout[i])
        next_value = rollout[i + 1][1]

        dones = dones.unsqueeze(1)
        rewards = rewards.unsqueeze(1)

        # Compute the updated returns
        returns = rewards + GAMMA * dones * returns

        # Compute temporal difference error
        td_error = rewards + GAMMA * dones * next_value.detach() - value.detach()
        
        advantages = advantages * TAU * GAMMA * dones + td_error
        processed_rollout[i] = [states, actions, log_probs, returns, advantages]

    # Concatenate along the appropriate dimension
    states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return states, actions, log_probs_old, returns, advantages


def train(env, brain_name, agent, num_agents, n_episodes, max_t, run_name="testing_01"):
    print(f"Starting training...")
    env.info = env.reset(train_mode = True)[brain_name]
    all_scores = []
    all_scores_window = deque(maxlen=100)
    best_so_far = 5.0
        
    for i_episode in range(n_episodes):
        rollout, returns, _, _ = collect_trajectories(env, brain_name, agent, max_t)
        
        states, actions, log_probs_old, returns, advantages = calculate_advantages(rollout, returns, num_agents)
        agent.learn(states, actions, log_probs_old, returns, advantages)
        
        test_mean_reward = test_agent(env, agent, brain_name)

        all_scores.append(test_mean_reward)
        all_scores_window.append(test_mean_reward)

        if np.mean(all_scores_window) > best_so_far:
            if not os.path.isdir(f"./ckpt/{run_name}/"):
                os.mkdir(f"./ckpt/{run_name}/")
            torch.save(agent.model.state_dict(), f"./ckpt/{run_name}/ppo_checkpoint_{np.mean(all_scores_window)}.ckpt")
            best_so_far = np.mean(all_scores_window)
            if np.mean(all_scores_window) > 30:
                
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(all_scores_window)))
                # break       
        
        print('Episode {}, Total score this episode: {}, Last {} average: {}'.format(i_episode + 1, test_mean_reward, min(i_episode + 1, 100), np.mean(all_scores_window)) )