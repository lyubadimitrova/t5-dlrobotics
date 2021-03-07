import time
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common import logger
from torch.utils.tensorboard import SummaryWriter
from cma import CMAEvolutionStrategy


class Algo:
    def learn(self, total_timesteps, tb_dir):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def evaluate_model(self, env):
        raise NotImplementedError


class CMA(Algo):

    def __init__(self, env, env_name, hyperparams):

        self.env = env
        self.env_name = env_name
        self.init_sigma = hyperparams.get('init_sigma', 0.1)

        if hyperparams.get('x_start', False):
            self.x_start = hyperparams['x_start']
        else:
            self.x_start = self._get_xstart()

        self.inopts = hyperparams.get('inopts', None)
        self.popsize = self.inopts['popsize']

        self.algo = CMAEvolutionStrategy(x0=self.x_start, sigma0=self.init_sigma, inopts=self.inopts)
        self.verbose = hyperparams.get('verbose', True)

        self.opts = []

    def learn(self, total_timesteps, tb_dir):
        t = 1
        opt = 1e10
        print_freq = 10

        tb_writer = SummaryWriter(tb_dir)
        print("Logging to", tb_dir)

        env_steps_per_iteration = self.popsize * self.env.env_fns[0]().spec.max_episode_steps * self.env.num_envs

        # training the model
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

            # plotting the total rewards and last step reward
            tb_writer.add_scalar("iterations/reward_last_step", -opt, t*env_steps_per_iteration)
            tb_writer.add_scalar("iterations/iterations", t, t*env_steps_per_iteration)

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

    def __init__(self, env, env_name, hyperparams):
        self.env = env
        self.env_name = env_name
        self.algo = sb3.PPO(env=env, **hyperparams)

    def learn(self, total_timesteps, tb_dir):

        self.algo.tensorboard_log = tb_dir
        
        # training the model
        iteration = 0
        total_timesteps = int(total_timesteps)
        log_interval = 1
    
        total_timesteps, callback = self.algo._setup_learn(total_timesteps,
                                                           callback=None,
                                                           eval_env=None,
                                                           eval_freq=-1,
                                                           n_eval_episodes=5,
                                                           tb_log_name='tb',
                                                           reset_num_timesteps=True,)
        
        callback.on_training_start(locals(), globals())
        
        while self.algo.num_timesteps < total_timesteps:

            continue_training = self.algo.collect_rollouts(self.algo.env, callback, self.algo.rollout_buffer,
                                                           n_rollout_steps=self.algo.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self.algo._update_current_progress_remaining(self.algo.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.algo.num_timesteps / (time.time() - self.algo.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.algo.ep_info_buffer) > 0 and len(self.algo.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.algo.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.algo.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.algo.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.algo.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.algo.num_timesteps)
                
                # custom logging
                logger.record("iterations/reward_last_step", np.mean(self.algo.env.get_attr('reward_last')))
                logger.record("iterations/iteration", iteration)

            self.algo.train()
        callback.on_training_end()

    def load_model(self, path):
        self.algo = sb3.PPO.load(path / 'model')

    def save_model(self, path):
        self.algo.save(path / 'model')

    def evaluate_model(self, test_env):
        reward, _ = evaluate_policy(self.algo, test_env,
                                    n_eval_episodes=1, render=True, deterministic=True, warn=False)
        return reward
