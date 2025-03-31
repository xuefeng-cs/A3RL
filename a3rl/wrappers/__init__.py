import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from a3rl.wrappers.single_precision import SinglePrecision
from a3rl.wrappers.universal_seed import UniversalSeed

def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env
