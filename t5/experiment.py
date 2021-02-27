import yaml
import gym
import importlib
import time
from pathlib import Path
import shutil

import t5.dict_helpers as dict_helpers
from t5.algorithm import CMA, PPO

from stable_baselines3.common.vec_env import VecNormalize


class Experiment:

    def __init__(self, config_path):
        self.cfg_path = config_path
        self.cfg = load_config(config_path)  # self.cfg is a multilevel dictionary
        self.model_dir = None

        self.total_timesteps = self.cfg['n_timesteps']

        algo = self.cfg['algorithm'].lower()
        assert algo in {'ppo', 'cma'}, "Algorithm {} not supported.".format(algo)

        self.algo_class = dict_helpers.ALGORITHMS[algo]

        self.env_name = self.cfg['env']
        self.env_params = self.cfg['env_params']

        self.env = gym.make(self.env_name)

        self.wrappers = self.env_params['wrappers']
        self.wrapper_params = self.env_params['wrapper_params']

        self.built_wrappers = []
        self.built_wrapper_params = []
        self.parse_wrappers()

        self.n_envs = self.env_params.get('n_envs', 1)

        self.vectorizer = dict_helpers.VECTORIZERS[algo]
        vecenv_params = self.env_params.get('vecenv_params', {})
        vecenv_params = parse_params(vecenv_params)

        self.env = make_env(self.build_wrap_env, self.n_envs,
                            vectorizer=self.vectorizer, hyperparams=vecenv_params)

        self.build_vec_wrappers()

        self.model = self.algo_class(env=self.env, hyperparams=self.cfg['algo_params'])

        # self.eval_function =

        # TODO: logging (path & tensorboard)

    def load(self, model_path):
        self.model_dir = Path(model_path)
        self.model.load_model(self.model_dir)

    def run(self):
        self.model.learn(self.total_timesteps)
        # self.env.close()

    def save(self):
        self.model_dir = make_experiment_dir(self.cfg)
        shutil.copyfile(self.cfg_path, self.model_dir / 'config.yml')

        self.model.save_model(self.model_dir)
        if isinstance(self.env, VecNormalize):
            self.env.save(self.model_dir / 'vecnormalize.pkl')

    def test_learned(self):
        test_env = make_test_env(self.model, self.build_wrap_env, self.vectorizer, self.model_dir)
        reward = self.model.evaluate_model(test_env)
        test_env.close()
        print('Episode reward: ', reward)
        return reward

    def build_wrap_env(self):
        env = gym.make(self.env_name)
        for i in range(len(self.built_wrappers)):
            if 'vec' not in self.wrappers[i].lower():  # ignore vecenv wrappers for now
                env = self.built_wrappers[i](env, **self.built_wrapper_params[i])
        return env

    def parse_wrappers(self):
        for i in range(len(self.wrappers)):
            if self.wrappers[i] is not None:
                self.built_wrappers.append(resolve_import(self.wrappers[i]))
                self.built_wrapper_params.append(parse_to_dict(self.wrapper_params[i]))

    def build_vec_wrappers(self):
        for i in range(len(self.built_wrappers)):
            if 'vec' in self.wrappers[i].lower():  # only vec wrappers
                self.env = self.built_wrappers[i](self.env, self.built_wrapper_params[i])


def make_env(creator, n_envs=1, hyperparams={}, vectorizer=None, base_seed=0):
    def maker(rank):
        def _init():
            env = creator()
            env.seed(base_seed + rank)
            return env

        return _init

    if vectorizer is None:
        return maker(0)

    return vectorizer(env_fns=[maker(i) for i in range(n_envs)], **hyperparams)


def make_test_env(model, creator, vectorizer=None, model_dir=None):
    if isinstance(model, CMA):
        return make_env(creator)()
    elif isinstance(model, PPO):
        test_env = make_env(creator, vectorizer=vectorizer)
        try:
            test_env = VecNormalize.load(model_dir / "vecnormalize.pkl", test_env)
            test_env.training = False
            test_env.norm_reward = False
        except FileNotFoundError:
            pass
        return test_env


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
    return {k: eval(v) for pair in params_str.split() for k, v in [pair.split(':')]}


def resolve_import(str_):
    module, member = str_.rsplit('.', 1)

    module = importlib.import_module(module)
    member = getattr(module, member)

    return member


def load_config(filepath):
    """
    Loads the configuration file on filepath into a Python dictionary.
    """
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def make_experiment_dir(cfg):

    exist_ok = cfg.get("overwrite", False)
    model_dir = Path(cfg["save_path"]) / (cfg["name"] + time.strftime("%Y%m%d-%H%M%S"))
    model_dir.mkdir(parents=True, exist_ok=exist_ok)

    return model_dir
