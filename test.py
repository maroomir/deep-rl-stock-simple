import gym
import numpy as np
from glob import glob
import re

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.bench import Monitor

from env import Stock

from utils.callbacks import logDir

env = Stock(code="005930", verbose=False)

model_files = sorted(glob(logDir()+'*_best_model.pkl'))
model_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

best_model = model_files[-1]

model = DDPG.load(best_model)

test_episodes = 100

mean_value = 0
for i in range(test_episodes):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        if dones == True:
            mean_value += info['total_value']
            break

print("Mean value: ", mean_value / test_episodes)