import stable_baselines3
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import t5.algorithm

ALGORITHMS = {'ppo': stable_baselines3.PPO, 'cma': t5.algorithm.CMA}

VECTORIZERS = {'ppo':DummyVecEnv, 'cma':DmpAsyncVectorEnv}
