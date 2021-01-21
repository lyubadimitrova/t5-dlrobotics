import yaml
import gym
import importlib
from functools import partial

from stable_baselines3 import PPO
#from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv, _worker

import t5.dict_helpers as dict_helpers
import sys 



class Experiment:

    def __init__(self, config_path):
        self.cfg = load_config(config_path)  # self.cfg is a multilevel dictionary
        self.name = self.cfg['name']

        self.total_timesteps = self.cfg['n_timesteps']

        algo = self.cfg['algorithm'].lower()
        assert algo in {'ppo', 'cma'}

        self.algo = dict_helpers.ALGORITHMS[algo]

        self.env_name = self.cfg['env']
        self.env_params = self.cfg['env_params']

        self.env = gym.make(self.env_name)

        self.wrappers = self.env_params['wrappers']
        self.wrapper_params = self.env_params['wrapper_params']
        self.parse_wrappers()

        self.n_envs = self.env_params.get('n_envs', 1)
        
        vectorizer = dict_helpers.VECTORIZERS[algo]
        vecenv_params = self.env_params.get('vecenv_params', {})
        vecenv_params = parse_params(vecenv_params)

        self.env = make_vec_env(self.build_wrap_env, self.n_envs, 
                                vectorizer=vectorizer, hyperparams=vecenv_params)

        self.build_vec_wrappers()


        self.model = self.algo(env=self.env, **self.cfg['algo_params'])


        # TODO: logging (path & tensorboard)

    def run(self):
        self.model.learn(self.total_timesteps)
        test_env = self.env.env_fns[0]()
        test_env.rollout(self.model.algo.mean, render=True)


    def build_wrap_env(self):

        env = gym.make(self.env_name)


        for i in range(len(self.built_wrappers)):
            if 'vec' not in self.wrappers[i].lower():  # ignore vecenv wrappers for now
                env = self.built_wrappers[i](env, **self.built_wrapper_params[i])

        return env


    def parse_wrappers(self):
        self.built_wrappers = []
        self.built_wrapper_params = []
        for i in range(len(self.wrappers)):
            if self.wrappers[i] is not None:
                self.built_wrappers.append(resolve_import(self.wrappers[i]))
                self.built_wrapper_params.append(parse_to_dict(self.wrapper_params[i]))

    def build_vec_wrappers(self):

        for i in range(len(self.built_wrappers)):
            if 'vec' in self.wrappers[i].lower():  # only vec wrappers
                self.env = self.built_wrappers[i](self.env, self.built_wrapper_params[i])

   


def make_vec_env(creator, n_envs, hyperparams={}, vectorizer=DummyVecEnv, base_seed=0):
    
    def make_env(rank):
        def _init():
            env = creator()
            env.seed(base_seed + rank)
            return env
        return _init

    return vectorizer(env_fns=[make_env(i) for i in range(n_envs)], **hyperparams)


def parse_params(params_dict):
    for key in params_dict:
        if isinstance(params_dict[key], str):
            try:
                thing = resolve_import(params_dict[key])
                params_dict[key] = thing
            except:
                pass

    return params_dict


def parse_to_dict(params_str):
    return {k: eval(v) for pair in params_str.split() for k,v in [pair.split(':')]}


def resolve_import(str_):
    module, member = str_.rsplit('.', 1)

    module = importlib.import_module(module)
    member = getattr(module, member)

    return member


def load_config(filepath):
    """
    Loads the configuration file on filepath into a Python dictionary.
    """
    import yaml
    
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg



