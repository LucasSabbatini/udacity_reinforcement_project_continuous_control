import numpy as np

from unityagents import UnityEnvironment
import numpy as np
env = UnityEnvironment(file_name='../unity_ml_envs/Reacher_Windows_x86_64/Reacher')

# Optimizer
# we use the adam optimizer with learning rate 2e-4
import torch.optim as optim
from policy import policy
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

# training loop max iterations
episode = 500

# widget bar to display progress
import progressbar as pb
widget = ['training loop: ', pb.Percentage(), ' ', 
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

discount_rate = .99
epsilon = 0.1
beta = .01
tmax = 320
SGD_epoch = 4

from trainer import train

train(policy,
      optimizer,
      env,
      timer,
      episodes=episode,
      epsilon=epsilon,
      beta=beta,
      tmax=tmax,
      SGD_epoch=SGD_epoch,
      run_name="testing")