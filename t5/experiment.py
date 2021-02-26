import yaml
import gym
import importlib
import time
from pathlib import Path

import t5.dict_helpers as dict_helpers
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize


class Experiment:

    def __init__(self, config_path):
        self.cfg = load_config(config_path)  # self.cfg is a multilevel dictionary
        self.save_dir = make_experiment_dir(self.cfg)
        shutil.copyfile(config_path, self.save_dir / 'config.yml')

        self.total_timesteps = self.cfg['n_timesteps']

        algo = self.cfg['algorithm'].lower()
        assert algo in {'ppo', 'cma'}

        self.algo = dict_helpers.ALGORITHMS[algo]

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

        self.model = self.algo(env=self.env, hyperparams=self.cfg['algo_params'])

        # self.eval_function =

        # TODO: logging (path & tensorboard)

    def run(self):
        self.model.learn(self.total_timesteps)
        self.env.close()

    def save(self):
        self.model.save_model(self.save_dir)
        if isinstance(self.env, VecNormalize):
            self.env.save(self.save_dir / 'vecnormalize.pkl')

    def test_learned(self):
        test_env = make_env(self.build_wrap_env)()
        if hasattr(test_env, 'rollout'):
            reward, _ = test_env.rollout(self.model.algo.mean, render=True)
        else:
            test_env = make_env(self.build_wrap_env, vectorizer=self.vectorizer)
            test_env = VecNormalize.load(self.save_dir / "vecnormalize.pkl", test_env)
            test_env.training = False
            test_env.norm_reward = False
            reward, _ = evaluate_policy(self.model.algo, test_env,
                                        n_eval_episodes=10, render=True, deterministic=True, warn=False)

        print('Episode reward: ', reward)
        test_env.close()
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
