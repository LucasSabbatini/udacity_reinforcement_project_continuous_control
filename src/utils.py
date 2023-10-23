from numpy import np


def test_agent(env, agent, brain_name):
    env_info = env.reset(train_mode = True)[brain_name]
    num_agents = len(env_info.agents)
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
