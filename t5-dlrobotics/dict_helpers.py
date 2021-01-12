from stable_baselines3 import PPO
from cma import CMAEvolutionStrategy

ALGORITHMS = {'ppo': PPO, 'cma': CMAEvolutionStrategy}