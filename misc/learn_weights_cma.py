import cma
import numpy as np
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv, _worker
from alr_envs.utils.dmp_env_wrapper import DmpEnvWrapperVel
from alr_envs.classic_control.hole_reacher import HoleReacher

from matplotlib import patches
import matplotlib.pyplot as plt
import os
import sys

if __name__ == "__main__":

    def make_env(rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the initial seed for RNG
        :param rank: (int) index of the subprocess
        :returns a function that generates an environment
        """

        def _init():
            env = HoleReacher(num_links=5,
                              allow_self_collision=False,
                              allow_wall_collision=False,
                              hole_width=0.15,
                              hole_depth=1,
                              hole_x=1,
                              via_point=(2,2))
                              
            env = DmpEnvWrapperVel(env,
                                   num_dof=5,
                                   num_basis=5,
                                   duration=2,
                                   dt=env._dt,
                                   learn_goal=True)
            env.seed(seed + rank)
            return env

        return _init
        
        
    

    n_cpu = 8
    n_samples = 14
    #print('make_env', make_env(1))
    env_fns = [make_env(i) for i in range(n_cpu)]
    objective = DmpAsyncVectorEnv(env_fns=env_fns,
                                  n_samples=n_samples,
                                  context="spawn",
                                  shared_memory=False,
                                  worker=_worker)


    # how to generate an environment without multiprocessing
    # objective = make_env(0, 0)()

    n = 30
    init_sigma = 0.1
    x_start = 0.1 * np.random.randn(n, 1)
    x_start[-5] = np.pi / 2
    x_start[-4] = -np.pi / 4
    x_start[-3] = -np.pi / 4
    x_start[-2] = -np.pi / 4
    x_start[-1] = -np.pi / 4  


    # create an instance of the CMA-ES algorithm
    algo = cma.CMAEvolutionStrategy(x0=x_start, sigma0=init_sigma, inopts={"popsize": n_samples})
    
    

    t = 0
    opt = 1e10
    opts = []
    while t < 10 and opt > 1e-8:
        print("----------iter {} -----------".format(t))

        # sample parameters to test
        solutions = algo.ask()   # solutions Dimensioin: [14,30]
        # collect rollouts with parameters, need to negate because cma-es minimizes
        #print('aszd')
        # print('solutions', (np.vstack(solutions).shape))
        fitness = -objective(np.vstack(solutions))[0] # fitness is the reward, [0] because rollout now returns infos as well
        # update search distribution
        #print('fitness',fitness)
        algo.tell(solutions, fitness)
        #print('dfdsghz')
        opt = -objective(algo.mean)[0]
    
        opts.append(opt)


        print(opt)

        t += 1
    
    ### TODO: Plot the performance of the opt
    plt.figure()
    plt.title("Performance")
    opts_select = [opts[a] for a in range(len(opts)) if opts[a] < 10] # to delete the opt greater than 10
    plt.plot(opts_select)
    plt.xlabel("Iteration")
    plt.ylabel("opt")

    os.mkdir('cma_data')
    plt.savefig('cma_data/perfomance')
    ### TODO: Finish
    

    ### TODO: save the model data
    np.save(os.path.join('cma_data', 'algo_mean.npy'), algo.mean)
    np.save(os.path.join('cma_data', 'opts.npy'), opts)
    #algo.mean= np.load("algo_mean.npy")
    #opts= np.load("opts.npy")
    ### TODO: Finish

    objective.close()
    test_env = make_env(0, 0)()
    test_env.rollout(algo.mean, render=True)
