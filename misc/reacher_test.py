#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:44:16 2020

@author: gyh
"""



import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines.common.evaluation import evaluate_policy

from stable_baselines import PPO2


env_id = 'Reacher-v2'
video_folder = 'videos/'
video_length = 20

env = DummyVecEnv([lambda: gym.make(env_id)])

obs = env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder, 
			record_video_trigger=lambda x: x == 0, 
			video_length=video_length, 
			name_prefix="random-agent-{}".format(env_id))

# build the model
model = PPO2(MlpPolicy, env, verbose=1)

# evaluate the model, before training
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=500)
print(f"Before trainig: \n mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# pirnt result : mean_reward:-11.87 +/- 4.80

# train the model
model.learn(total_timesteps=int(1e4))

# evaluate the model again, after training
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=500)
print(f"After trainig: \n mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# pirnt result : mean_reward:-26.12 +/- 1.92

env.reset()

### we can no that the reward doesn't increas at all. 
### Also the reward is negative.
### That leads to: the Reacher cannot reach the ball after training. (Even with more episodes)
 
### Next step: find the reason which causes the negative reward.

"""
for _ in range(video_length + 1):
  action = [env.action_space.sample()]
  obs, _, _, _ = env.step(action)
# Save the video
env.close()
"""
