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

    return DummyVecEnv(env_fns=[make_env(i) for i in range(n_envs)],
                      **hyperparams)


cfg = load_config('configs/example.yml')
hyperparams = cfg['hyperparameters']

n_envs = 1 #hyperparams['n_envs']
del hyperparams['n_envs']

n_timesteps = hyperparams['n_timesteps']
del hyperparams['n_timesteps']

wrapper_class = None

if hyperparams.get('env_wrapper', False):
    module, class_ = hyperparams['env_wrapper'].rsplit('.', 1)

    del hyperparams['env_wrapper']

    wrapper_module = importlib.import_module(module)
    wrapper_class = getattr(wrapper_module, class_)


print(wrapper_class)

#env = wrapper_class(gym.make(cfg['env']), **{})

#print(env)  

#env = sb3.common.env_util.make_vec_env(cfg['env'], n_envs, wrapper_class=wrapper_class,
                                                           #vec_env_kwargs={'gamma': hyperparams['gamma']}
                                                           #)

env = make_vec_env(n_envs)


if hyperparams.get('normalize', False):
    #print('HEREEE')
    env = sb3.common.vec_env.VecNormalize(env, gamma=hyperparams['gamma'])
    del hyperparams['normalize']


#del hyperparams['normalize']

#model = sb3.PPO(env=env, **hyperparams, verbose=1)

#model.learn(total_timesteps=10000)


obs = env.reset()
env.envs[0].set_state(np.array([np.pi/2, np.pi/4, 0.1, -0.1]), env.envs[0].init_qvel)
for i in range(100):
    env.envs[0].render()
sys.exit()
action = np.array([0, 1])
action = action[np.newaxis]
#print(np.array(action))

for i in range(1000):
    # print('body0 posvel', env.envs[0].sim.data.get_body_xvelp('body0'))
    # print('body0 rotvel', env.envs[0].sim.data.get_body_xvelr('body0'))
    # print('body1 posvel', env.envs[0].sim.data.get_body_xvelp('body1'))
    # print('body1 rotvel', env.envs[0].sim.data.get_body_xvelr('body1'))

    #action, _states = model.predict(obs)
    print('before body0 rotvel', env.envs[0].sim.data.get_body_xvelr('body0'))
    print('before body1 rotvel', env.envs[0].sim.data.get_body_xvelr('body1'))
    obs, rewards, dones, info = env.step(action)
    print('after body0 rotvel', env.envs[0].sim.data.get_body_xvelr('body0'))
    print('after body1 rotvel', env.envs[0].sim.data.get_body_xvelr('body1'))
    env.render()
    time.sleep(1)

env.close()