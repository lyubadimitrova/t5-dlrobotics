from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from cma import CMAEvolutionStrategy


class CMA:

    def __init__(self, env, inopts, init_sigma, x_start=None, verbose=True):

        self.env = env
        self.init_sigma = init_sigma

        if x_start is not None:
            self.x_start = x_start
        else:
            self.x_start = self._get_xstart()

        self.inopts = inopts

        self.algo = CMAEvolutionStrategy(x0=self.x_start, sigma0=self.init_sigma, inopts=self.inopts)
        self.verbose = verbose

        self.opts = []

    def learn(self, total_timesteps):
        t = 1
        opt = 1e10
        print_freq = 10

        while t <= total_timesteps and opt > 1e-8:

            if self.verbose and t % print_freq == 0:
                print("Iteration {}/{} ----------- Result: {}".format(t, total_timesteps, opt))

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

    def save_model(self, path):
        np.save(path / 'algo_mean.npy', self.algo.mean)
        np.save(path / 'opts.npy', self.opts)

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


class PPO:

    def __init__(self, env, hyperparams):
        # policy='MlpPolicy',
        #         batch_size=256,
        #         n_steps=1024,
        #         gamma=0.99,
        #         gae_lambda=0.98,
        #         n_epochs=10,
        #         ent_coef=0.0,
        #         learning_rate=8.e-4,
        #         clip_range=0.1,
        #         max_grad_norm=0.9,
        #         vf_coef=0.7):

        self.env = env
        self.algo = PPO(env=env, **hyperparams)

    def learn(self):
        pass
