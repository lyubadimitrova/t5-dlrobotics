import gym

from stable_baselines.common.policies import MlpPolicy as PPO_mlp
from stable_baselines.sac.policies import MlpPolicy as SAC_mlp
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, SAC

# TODO: hyperparameter setting, for example RL zoo
env = gym.make('Reacher-v2')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

model = PPO2(PPO_mlp, env) #, verbose=1, n_steps=2048, ent_coef=0.0, learning_rate=2.5e-4, 
						   #			  nminibatches=64, noptepochs=10, cliprange_vf=-1) 
#model = SAC(SAC_mlp, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(100000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()