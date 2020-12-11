#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:44:16 2020

@author: gyh
"""



import gym

#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines.common.evaluation import evaluate_policy

from gym import wrappers
from time import time # just to have timestamps in the files
from stable_baselines import PPO2, SAC

#model = PPO2.load(load_path = '/home/gyh/Desktop/Reacher', env='Hopper-v2', custom_objects = "PPO2_Reacher.zip")

env_id = 'Reacher-v2'
video_folder = 'videos/'
video_length = 1000

env = DummyVecEnv([lambda: gym.make(env_id)])

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, 
                       video_length=video_length,
                       name_prefix="random-agent-{}".format(env_id))

model = SAC(MlpPolicy, env, verbose=1)

# evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Before trainig: \n mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# when the timesteps is up to 2e6, it comes out overfitting, reward is -7.8
# when the timesteps is up to 1e5, agent gets to very nearly with the ball, reward -9
model.learn(total_timesteps=int(2e5))

# evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=500)
print(f"After trainig: \n mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
"""
for _ in range(video_length + 1):
  action = [env.action_space.sample()]
  obs, _, _, _ = env.step(action)
# Save the video
env.close()
"""


#python -m baselines.run --alg=ppo2 --env=Reacher-v2 --num_timesteps=2e4 --save_path=~/ppo2_with_Reacher
