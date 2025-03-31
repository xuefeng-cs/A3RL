from typing import Dict

import gymnasium as gym
import jax.numpy as jnp

def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, num_episodes)

    for i in range(num_episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.eval_actions(observation)
            observation, reward, terminated, truncated, info = env.step(action)
    return {"return": jnp.mean(jnp.array(env.return_queue)), "length": jnp.mean(jnp.array(env.length_queue))}
