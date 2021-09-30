import gym
import numpy as np

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor

from env import Stock

from utils.callbacks import getBestRewardCallback, logDir

env = Stock(code="005930", verbose=False)
env = Monitor(env, logDir(), allow_early_resets=True)

bestRewardCallback = getBestRewardCallback()

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise)
model.learn(total_timesteps=20000, log_interval=100, callback=bestRewardCallback)
