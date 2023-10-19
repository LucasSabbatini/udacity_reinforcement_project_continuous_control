import time
import numpy as np
import torch
from unityagents import UnityEnvironment
from IPython.display import clear_output
from policy import PPOAgent


def test_agent(agebt, env):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    time.sleep(3)

    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    while True:
        actions, _, _= agent.act(states)
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break
    return np.mean(scores)


if __name__ == '__main__':
    # INTANTIATE ENV
    # env = UnityEnvironment(file_name=f'C:\Users\lucas\Documents\Udacity\DeepReinforcementLearningExpert\unity_ml_envs\Reacher_Windows_x86_64\Reacher.exe')
    env = UnityEnvironment(file_name='../unity_ml_envs/Reacher_Windows_x86_64/Reacher')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # GET ENV INFO
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    print ("Action Size is :", action_size)
    states = env_info.vector_observations
    state_size = states.shape[1]

    # INSTANTIATE AGENT
    agent = PPOAgent(num_agents, state_size, action_size)
    agent.load_state_dict(torch.load('./checkpoints/ppo_checkpoint.pth'))

    # TEST AGENT
    score = test_agent(agent, env)
    print(f"Final score: {score}")


