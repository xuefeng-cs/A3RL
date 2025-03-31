#! /usr/bin/env python
import os
import pickle

import gymnasium as gym
import tqdm
from absl import app, flags

import jax
import jax.numpy as jnp
import numpy as np

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from a3rl.agents import SACLearner
from a3rl.data import ReplayBuffer
from a3rl.data.minari_dataset import MinariDataset

from a3rl.evaluation import evaluate
from a3rl.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "a3rl", "wandb project name.")
flags.DEFINE_string("group_name", "default_group_name", "wandb group name.")
flags.DEFINE_string("run_name", "default_run_name", "wandb run name.")
flags.DEFINE_string("env_name", "mujoco/halfcheetah/expert-v0", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("max_steps", 300000, "Number of training steps.")
flags.DEFINE_integer("start_training", 5000, "Number of training steps to start training.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save agent replay buffer on evaluation.")

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def combine(one_dict, other_dict):
    """
    interweaves 2 dictionaries of equal size
    """
    combined = {}
    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp
    return combined


def main(_):
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0
    wandb.init(entity="hung-active-rlpd",
               project=FLAGS.project_name, 
               name=FLAGS.run_name,
               group=FLAGS.group_name,
               id=FLAGS.run_name,
               resume="allow")
    wandb.config.update(FLAGS, allow_val_change=True)
    
    x = jnp.ones((1000, 1000))
    print(f'Starting run {FLAGS.run_name} in group {FLAGS.group_name} in project {FLAGS.project_name} with seed {FLAGS.seed} with device {x.device}')
    print(f'Some flags: critic_layer_norm: {FLAGS.config.critic_layer_norm}, num_min_qs: {FLAGS.config.num_min_qs}, num_qs: {FLAGS.config.num_qs}')
    exp_prefix = FLAGS.run_name

    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)


    ds = MinariDataset(FLAGS.env_name)
    env = ds.minari_dataset.recover_environment()
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, 1)
    env.reset(seed=FLAGS.seed)
    
    eval_env = ds.minari_dataset.recover_environment(True)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.reset(seed=FLAGS.seed + 42)
    
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    observation, _ = env.reset(seed=FLAGS.seed)
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        mask = 1.0 - float(terminated)

        replay_buffer.insert(
            dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=reward,
                masks=mask,
            )
        )
        observation = next_observation

        if terminated or truncated:
            observation, _ = env.reset()
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i)

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            offline_batch = ds.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
            )

            batch = combine(offline_batch, online_batch)

            if "antmaze" in FLAGS.env_name:
                batch["rewards"] -= 1

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

if __name__ == "__main__":
    app.run(main)
