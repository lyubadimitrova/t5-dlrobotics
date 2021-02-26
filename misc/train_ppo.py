import importlib
import gym
import numpy as np
import time
import sys

import stable_baselines3 as sb3 
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
#env = sb3.common.env_util.make_vec_env(cfg['env'], n_envs, wrapper_class=wrapper_class,


def load_config(filepath):
    """
    Loads the configuration file on filepath into a Python dictionary.
    """
    import yaml
    
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def make_vec_env(n_envs, hyperparams={}, base_seed=0):
    
    
    def make_env(rank):
        def _init():
            env = wrapper_class(gym.make(cfg['env']), **{})
            env.seed(base_seed + rank)
            return env
        return _init

    return DummyVecEnv(env_fns=[make_env(i) for i in range(n_envs)])


cfg = load_config('configs/reacher_ppo.yml')
env_params = cfg['env_params']

n_envs = 1 #hyperparams['n_envs']

n_timesteps = cfg['n_timesteps']

wrapper_str = env_params['wrappers'][0]

module, class_ = wrapper_str.rsplit('.', 1)

wrapper_module = importlib.import_module(module)
wrapper_class = getattr(wrapper_module, class_)

print(wrapper_class)

env = make_vec_env(n_envs)
print(env)

vecwrapper_str = env_params['wrappers'][1]

module, class_ = vecwrapper_str.rsplit('.', 1)

vecwrapper_module = importlib.import_module(module)
vecwrapper_class = getattr(vecwrapper_module, class_)


env = vecwrapper_class(env, gamma=cfg['algo_params']['gamma'])
print(env)

model = sb3.PPO(env=env, **cfg['algo_params'])

model.learn(total_timesteps=1000000)
model.save('trained_ppo')

sys.exit()
#obs = env.reset()
# env.envs[0].set_state(np.array([np.pi/2, np.pi/4, 0.1, -0.1]), env.envs[0].init_qvel)
# for i in range(100):
#     env.envs[0].render()
# sys.exit()
# action = np.array([0, 1])
# action = action[np.newaxis]
# print(np.array(action))

for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    time.sleep(0.2)

# reward, _ = evaluate_policy(self.model.algo, test_env,
#                             n_eval_episodes=100, render=True, deterministic=True, warn=False)

env.close()