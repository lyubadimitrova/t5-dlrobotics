import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from cma import CMAEvolutionStrategy


class Algo:
    def learn(self, total_timesteps):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def evaluate_model(self, env):
        raise NotImplementedError


class CMA(Algo):

    def __init__(self, env, hyperparams):

        self.env = env
        self.init_sigma = hyperparams.get('init_sigma', 0.1)

        if hyperparams.get('x_start', False):
            self.x_start = hyperparams['x_start']
        else:
            self.x_start = self._get_xstart()

        self.inopts = hyperparams.get('inopts', None)

        self.algo = CMAEvolutionStrategy(x0=self.x_start, sigma0=self.init_sigma, inopts=self.inopts)
        self.verbose = hyperparams.get('verbose', True)

        self.opts = []

    def learn(self, total_timesteps):
        t = 1
        opt = 1e10
        print_freq = 10

        while t <= total_timesteps and opt > 1e-8:

            if self.verbose and t % print_freq == 0:
                print("Iteration {:>3}/{} ----------- Result: {}".format(t, total_timesteps, opt))

            # sample parameters to test
            solutions = self.algo.ask()
            # collect rollouts with parameters, need to negate because cma-es minimizes
            fitness = -self.env(np.vstack(solutions))[
                0]  # fitness is the reward, [0] because rollout now returns infos as well
            # update search distribution
            self.algo.tell(solutions, fitness)

            opt = -self.env(self.algo.mean)[0][0]
            self.opts.append(opt)

            t += 1

    def _get_xstart(self):
        dummy_env = self.env.env_fns[0]()
        x_start = self.init_sigma * np.random.randn(dummy_env.dim)
        if dummy_env.dim > dummy_env.num_basis * dummy_env.num_dof:
            x_start[-dummy_env.num_dof] = np.pi / 2
            for i in range(-dummy_env.num_dof + 1, 0):
                x_start[i] = -np.pi / 4

        return x_start

    def load_model(self, path):
        self.algo.mean = np.load(path / 'algo_mean.npy')
        self.opts = np.load(path / 'opts.npy')

    def save_model(self, path):
        np.save(path / 'algo_mean.npy', self.algo.mean)
        np.save(path / 'opts.npy', self.opts)

    def evaluate_model(self, test_env):
        reward, _ = test_env.rollout(self.algo.mean, render=True)
        return reward

    def plot_curve(self, max_value=None, min_value=None, save=None):
        plt.figure()
        plt.title("Learning curve")

        opts = self.opts
        # filters for outliers, so that the plot is pretty
        if max_value is not None:
            opts = [opt for opt in opts if opt < max_value]
        if min_value is not None:
            opts = [opt for opt in opts if opt > min_value]

        plt.plot(opts)
        plt.xlabel("Iteration")
        plt.ylabel("opt")

        if save is not None:
            plt.savefig(save)


class PPO(Algo):

    def __init__(self, env, hyperparams):
        self.env = env
        self.algo = sb3.PPO(env=env, **hyperparams)

    def learn(self, total_timesteps):
        self.algo.learn(total_timesteps=total_timesteps)

    def load_model(self, path):
        self.algo = sb3.PPO.load(path / 'model')

    def save_model(self, path):
        self.algo.save(path / 'model')

    def evaluate_model(self, test_env):
        reward, _ = evaluate_policy(self.algo, test_env,
                                    n_eval_episodes=10, render=True, deterministic=True, warn=False)
        return reward