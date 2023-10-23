


# Training Hyperparameters
EPISODES = 1000
MAX_T = 2048
SGD_EPOCHS = 4
# optimizer parameters
LR = 3e-4
EPSILON = 1e-5
GAMMA = 0.99            # Discount factor
TAU = 0.95              # GAE parameter

# Agent hyperparameters
BATCH_SIZE = 32         # minibatch size
BETA = 0.01             # entropy regularization parameter
PPO_CLIP_EPSILON = 0.2  # ppo clip parameter
GRADIENT_CLIP = 5       # gradient clipping parameter