from unityagents import UnityEnvironment
import time
from agent import Agent
from trainer import train
from conf import *


if __name__=="__main__":
    # env = UnityEnvironment(file_name='../../unity_ml_envs/Reacher_Windows_x86_64/Reacher.exe')
    env = UnityEnvironment(file_name='../../PPO-Reacher_UnityML/Reacher_Windows_x86_64/Reacher.exe')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    time.sleep(2)
    print(f"Environment loaded.")

    # Environment variables
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size

    # Instantiate the agent
    agent = Agent(num_agents, state_size, action_size)
    print(f"Agent instantiated.")

    # Train the agent
    train(env, brain_name, agent, num_agents, EPISODES, MAX_T)
    env.close()