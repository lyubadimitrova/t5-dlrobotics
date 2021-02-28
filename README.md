# T5-dlrobotics

Comparing Episodic vs Step-Based Policy Representations

Authors: Dimitrova, Lyubomira & Gong, Yuhe 

### Overview

This repository houses a framework for comparing two types of policy representations:

- Neural network policies as used in standard reinforcement learning (RL) algorithms like 
  Proximal Policy Optimization (PPO)
- Low-dimensional policies like dynamic movement primitives (DMPs) that can be learned with 
  stochastic search algorithms like CMA-ES

The way to learn such policies is different, depending on what type of reward signal is available. 
While e.g. PPO is usually applied to RL problems with reward signal at 
every step, it might have trouble if the reward is sparse (cite). On the other hand, a DMP
policy learns a trajectory and as such can deal well with a sparse reward signal.

With the framework, we experiment with different policies and reward signals for solving two
environments: the [MuJoCo Reacher](https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py) 
and the [ALR HoleReacher](https://github.com/ALRhub/alr_envs/blob/dmp_env_dev/alr_envs/classic_control/hole_reacher.py).
Both environments needed modifications for at least one of the examined policies, either in the form of a gym 
Wrapper class or as a new environment:

- [CmaReacherWrapper](./t5/wrappers.py)  for solving the MuJoCo Reacher with DMP policy
- [FixedTargetReacherEnv](https://github.com/lyubadimitrova/alr_envs/blob/dmp_env_dev/alr_envs/mujoco/fixed_goal_reacher/reacher.py) Enables
  comparison between policies, since DMPs represent a single trajectory, and so a single goal.
- [InformedHoleReacher](https://github.com/lyubadimitrova/alr_envs/blob/dmp_env_dev/alr_envs/classic_control/hole_reacher_informed.py) Same
  environment, but with a reward at every timestep.



### Training 

In order to train an agent, you need a config file. Several are provided in `./configs`.

For example, to solve the MuJoCo Reacher with CMA-ES, run:

```python -m t5 -c configs/reacher_cma.yml```

After training is done, all training output plus the configuration file are saved in 
the `experiments` directory, under a timestamped name. You can change output path is the
config file.

### Testing

After training, the learned policy is evaluated on a single episode and rendered 
automatically. If you want to evaluate a trained policy separately, you can run:

```python -m t5 -m experiments/cma-reacher20210228-105448```


### TODOs

- Saving videos
- Reward wrappers for all envs, something as generic as possible
- Tensorboard