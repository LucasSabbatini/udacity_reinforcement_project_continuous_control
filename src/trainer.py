import copy
import numpy as np
from objective_function import clipped_surrogate_lucas

def collect_single_trajectory(env, policy, tmax=500):
    """NOTE: SINGLE TRAJECTORY FOR ONE AGENT ONLY
    """
    # get the default brain
    brain_name = env.brain_names[0]
    # brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
    current_state = env_info.vector_observation          # getting only for one agent
    
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents: ', num_agents)
    assert num_agents==1, "There should only be one agent, for now."
    
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    state_list = []
    actions_list = []
    reward_list = []
    
    for i in range(tmax):
        action = policy(current_state).squeeze().cpu().detach().numpy()[:, np.newaxis]
                
        env_info = env.step(action)[brain_name]           # send all actions to tne environment
        next_state = env_info.vector_observations         # get next state (for each agent)
        reward = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        
        actions_list.append(copy.copy(action))
        state_list.append(copy.copy(current_state))
        reward_list.append(copy.copy(reward))
        
        current_state = next_state
        
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    
    return actions_list, state_list, reward_list

def train(policy, optimizer, env, timer,
          episodes=500,
          epsilon=0.1,
          beta=0.01,
          tmax=320,
          SGD_epoch=4,
          run_name="testing"):

    mean_rewards = []
    for e in range(episodes):

        # collect trajectories
        actions, states, rewards = collect_single_trajectory(env, policy, tmax=tmax)
        
        total_rewards = np.sum(rewards, axis=0)
        
        # gradient ascent step
        for _ in range(SGD_epoch): # NOTE: THIS WILL UPDATE THE POLICY SEVERAL TIMES ON THE SAME TRAJECTORY. This is possible due to the new policy and old policy ratio

            # uncomment to utilize your own clipped function!
            L = -clipped_surrogate_lucas(policy, states, actions, rewards, epsilon=epsilon, beta=beta)

    #         L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards,
    #                                           epsilon=epsilon, beta=beta)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

        # the clipping parameter reduces as time goes on
        epsilon*=.999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress every 20 iterations
        if (e+1)%20 ==0 :
            print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
            print(total_rewards)

        # update progress widget bar
        timer.update(e+1)

    timer.finish()
    torch.save(policy, f"PPO_{run_name}.policy")
    return 