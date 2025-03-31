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
from a3rl.agents import SACLearnerPriority
from a3rl.data import ReplayBuffer
from a3rl.data.minari_dataset import MinariDataset

from a3rl.evaluation import evaluate
from a3rl.wrappers import wrap_gym
import time

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "a3rl", "wandb project name.")
flags.DEFINE_string("group_name", "default_group_name", "wandb group name.")
flags.DEFINE_string("run_name", "default_run_name", "wandb run name.")
flags.DEFINE_string("env_name", "mujoco/halfcheetah/expert-v0", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_float("h_beta_0", 0.4, "initial beta_0 for bias annealing")
flags.DEFINE_float("h_alpha", 0.3, "priority exponentiation")
flags.DEFINE_float("h_lambda", 1, "advantage weight")
flags.DEFINE_float("epsilon", 1e-10, "epsilon pad")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("max_steps", 300000, "Number of training steps.")
flags.DEFINE_integer("start_training", 5000, "Number of training steps to start training.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", True, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean("checkpoint_buffer", True, "Save agent replay buffer on evaluation.")
flags.DEFINE_boolean("use_advantage", False, "whether to use advantage or TD error")

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
    print(f'Some flags: critic_layer_norm: {FLAGS.config.critic_layer_norm}, num_min_qs: {FLAGS.config.num_min_qs}, num_qs: {FLAGS.config.num_qs}, use_advantage: {FLAGS.use_advantage}, beta_0: {FLAGS.h_beta_0}, alpha: {FLAGS.h_alpha}, lambda: {FLAGS.h_lambda}, seed: {FLAGS.seed}, utd_ratio: {FLAGS.utd_ratio}, batch_size: {FLAGS.batch_size}')
    
    print('***\n'*3 + f'env_name: {FLAGS.env_name}\n' + '***\n'*3)
    exp_prefix = FLAGS.run_name

    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    N = int(FLAGS.batch_size * FLAGS.utd_ratio)
    
    print(f'log dir: {log_dir}')
    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)
        print(f'Model checkpoint directory: {chkpt_dir}')

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)
        print(f'Buffer checkpoint directory: {buffer_dir}')
        

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
    
    training_start_step = 0
    try:
        chkpt_data = checkpoints.restore_checkpoint(chkpt_dir, {
                    "critic": agent.critic,
                    "target_critic": agent.target_critic,
                    "temp": agent.temp,
                    "density": agent.density,
                    "actor": agent.actor,
                    "step": 0
                })
        agent = agent.replace(
            critic=chkpt_data["critic"],
            target_critic=chkpt_data["target_critic"],
            temp=chkpt_data["temp"],
            density=chkpt_data["density"],
            actor=chkpt_data["actor"]
        )
        training_start_step = chkpt_data["step"]
        print('Successfully restored from checkpoint')
        print(f'training_start_step: {training_start_step}')
        print(f'Density_step: {agent.density.step}')
        print(f'Critic_step: {agent.critic.step}')
        print(f'Target_critic_step: {agent.target_critic.step}')
        print(f'Temp_step: {agent.temp.step}')
        print(f'Actor_step: {agent.actor.step}')
    except Exception as e:
        print(f"Could not load model checkpoint: {e}")
        # traceback.print_exc()

    try:
        replay_buffer_chkpt_data = checkpoints.restore_checkpoint(buffer_dir, {
                    "dataset_dict": replay_buffer.dataset_dict,
                    "size": replay_buffer._size,
                    "insert_index": replay_buffer._insert_index
                })
        replay_buffer.dataset_dict["observations"] = replay_buffer_chkpt_data["dataset_dict"]["observations"].copy()
        replay_buffer.dataset_dict["next_observations"] = replay_buffer_chkpt_data["dataset_dict"]["next_observations"].copy()
        replay_buffer.dataset_dict["actions"] = replay_buffer_chkpt_data["dataset_dict"]["actions"].copy()
        replay_buffer.dataset_dict["rewards"] = replay_buffer_chkpt_data["dataset_dict"]["rewards"].copy()
        replay_buffer.dataset_dict["masks"] = replay_buffer_chkpt_data["dataset_dict"]["masks"].copy()

        replay_buffer._size = replay_buffer_chkpt_data["size"]
        replay_buffer._insert_index = replay_buffer_chkpt_data["insert_index"]
        print(f'The loaded replay buffer has size: {replay_buffer._size}, insert_index: {replay_buffer._insert_index} and capacity (should agree from the start): {replay_buffer._capacity}')
        for key in replay_buffer.dataset_dict:
            replay_buffer.dataset_dict[key].setflags(write=True)
            
    except Exception as e:
        print(f'Could not load buffer checkpoint: {e}')
        # traceback.print_exc()
    
    
    rng = jax.random.key(FLAGS.seed)
    observation, _ = env.reset(seed=FLAGS.seed)
    for i in tqdm.tqdm(
        range(training_start_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
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
                int(N * (1 - FLAGS.offline_ratio))
            )
            offline_batch = ds.sample(
                int(N * FLAGS.offline_ratio)
            )

            batch = combine(offline_batch, online_batch)

            if "antmaze" in FLAGS.env_name:
                batch["rewards"] -= 1

            offline_dens = agent.get_density(offline_batch)
            log_offline_dens = jnp.log(offline_dens + FLAGS.epsilon)

            all_log_dens = combine({"dens": log_offline_dens}, {"dens": jnp.zeros(log_offline_dens.shape)})["dens"]
            
            heuristics = None
            if FLAGS.use_advantage:
                heuristics = agent.get_advantage(batch)
            else:
                heuristics = agent.get_td(batch)

            # using PER notation
            log_priority =  all_log_dens + FLAGS.h_lambda * heuristics
            Prob = jax.nn.softmax(FLAGS.h_alpha * log_priority)
            
            
            # annealing
            beta = FLAGS.h_beta_0 + i/FLAGS.max_steps * (1 - FLAGS.h_beta_0)
            importance_sampling_weights = (1/N * 1/Prob) ** beta
            
            # sum to utd_ratio
            importance_sampling_weights *= FLAGS.utd_ratio / jnp.sum(importance_sampling_weights)

            rng, key = jax.random.split(rng)
            sampled_indices = jax.random.choice(key, 
                                                N,
                                                shape=(N,),
                                                p=Prob, replace=True)

            batch = {k: v[sampled_indices] for k, v in batch.items()}
            
            if "antmaze" in FLAGS.env_name:
                batch["rewards"] -= 1
            
            kl = -jnp.log(N) - jnp.sum(jnp.log(Prob))/N
            chosen_heuristics = heuristics[sampled_indices]
            
            log_info = {
                "hrts_max": jnp.max(chosen_heuristics),
                "hrts_min": jnp.min(chosen_heuristics),
                "hrts_75": jnp.percentile(chosen_heuristics, 75),
                "hrts_median": jnp.median(chosen_heuristics),
                "hrts_mean": jnp.mean(chosen_heuristics),
                "hrts_25": jnp.percentile(chosen_heuristics, 25),
                "N*p_max": jnp.max(Prob) * N,
                "N*p_min": jnp.min(Prob) * N,
                "kl_to_uniform": kl,
                "log_off_dens_max": jnp.max(log_offline_dens),
                "log_off_dens_min": jnp.min(log_offline_dens),
                "log_off_dens_75": jnp.percentile(log_offline_dens, 75),
                "log_off_dens_median": jnp.median(log_offline_dens),
                "log_off_dens_mean": jnp.mean(log_offline_dens),
                "log_off_dens_25": jnp.percentile(log_offline_dens, 25)

            }

            agent, update_info = agent.update(batch, FLAGS.utd_ratio, importance_sampling_weights)
            agent, density_update_info = agent.split_update(offline_batch, online_batch, FLAGS.utd_ratio)
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)
                for k, v in log_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            start_time = time.time()
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
            )
            print(f'evaluation: ', time.time() - start_time)
            start_time = time.time()
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)
                
            if FLAGS.checkpoint_model:
                try:
                    model_chkpt_data = {
                        "critic": agent.critic,
                        "target_critic": agent.target_critic,
                        "temp": agent.temp,
                        "density": agent.density,
                        "actor": agent.actor,
                        "step": i
                    }
                    checkpoints.save_checkpoint(
                        chkpt_dir, model_chkpt_data, step= i, keep=2, overwrite=True
                    )
                    print('Saved model checkpoint')
                except Exception as e:
                    print(f"Could not save model checkpoint: {e}")

            if FLAGS.checkpoint_buffer:
                try:
                    replay_buffer_chkpt_data = {
                        "dataset_dict": replay_buffer.dataset_dict,
                        "size": replay_buffer._size,
                        "insert_index": replay_buffer._insert_index
                    }
                    checkpoints.save_checkpoint(
                        buffer_dir, replay_buffer_chkpt_data, step = i, keep = 2, overwrite = True
                    )
                    print('Saved buffer checkpoint')
                except Exception as e:
                    print(f"Could not save buffer checkpoint: {e}")


if __name__ == "__main__":
    app.run(main)
