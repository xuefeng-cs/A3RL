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
from a3rl.agents import SACLearner, SACLearnerPriority
from a3rl.data import ReplayBuffer
from a3rl.data.minari_dataset import MinariDataset

from a3rl.evaluation import evaluate
from a3rl.wrappers import wrap_gym
from a3rl.envs import ENVS_LIST

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "a3rl", "wandb project name.")
flags.DEFINE_string("group_name", "default_group_name", "wandb group name.")
flags.DEFINE_string("run_name", "default_run_name", "wandb run name.")
flags.DEFINE_string("env_core", "halfcheetah", "Environment name")
flags.DEFINE_string("env_name", "halfcheetah_1_1_1", "Name of mixture of datasets")

flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_float("h_beta_0", 0.4, "initial beta_0 for bias annealing")
flags.DEFINE_float("h_alpha", 0.3, "priority exponentiation")
flags.DEFINE_float("h_alpha_final_p", 1.0, "final priority exponentiation proportional constant")
flags.DEFINE_float("h_lambda", 1.0, "advantage weight")
flags.DEFINE_float("epsilon", 1e-5, "epsilon pad")
flags.DEFINE_string("heuristics", "adv", "none, adv, td")
flags.DEFINE_boolean("use_density", True, "density correction")
flags.DEFINE_boolean("use_interweave", False, "Interweave A3RL with RLPD")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("interweave_a3rl", 5, "Chunk of interweave for a3rl")
flags.DEFINE_integer("interweave_rlpd", 5, "Chunk of interweave for rlpd")

flags.DEFINE_float("p_simple", 0.5, "Chunk of interweave for a3rl")
flags.DEFINE_float("p_medium", 0.1, "Chunk of interweave for a3rl")
flags.DEFINE_float("p_expert", 0.1, "Chunk of interweave for a3rl")


flags.DEFINE_integer("eval_episodes", 20, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("max_steps", 300000, "Number of training steps.")
flags.DEFINE_integer("pretrain_steps", 20000, "Number of offline updates.")
flags.DEFINE_integer("start_a3rl", 100000, "Number of training steps to start running a3rl.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", True, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean("checkpoint_buffer", True, "Save agent replay buffer on evaluation.")

config_flags.DEFINE_config_file(
    "config",
    "configs/a3rl_config.py",
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
    if (FLAGS.use_density):
        assert FLAGS.offline_ratio == 0.5

    wandb.init(entity="hung-active-rlpd",
               project=FLAGS.project_name, 
               name=FLAGS.run_name,
               group=FLAGS.group_name,
               id=FLAGS.run_name,
               resume="allow")
    config_dict = {key: round(FLAGS[key].value, 6) if isinstance(FLAGS[key].value, float) else FLAGS[key].value for key in FLAGS}

    wandb.config.update(FLAGS, allow_val_change=True)
    
    x = jnp.ones((1000, 1000))
    exp_prefix = FLAGS.run_name
    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)
    
    print('***\n'*3 + f'Starting run {FLAGS.run_name} in group {FLAGS.group_name} in project {FLAGS.project_name} with seed {FLAGS.seed} with device {x.device}' + '***\n'*3)
    print(f'Some flags: critic_layer_norm: {FLAGS.config.critic_layer_norm}, num_min_qs: {FLAGS.config.num_min_qs}, num_qs: {FLAGS.config.num_qs}, offline_ratio: {FLAGS.offline_ratio}, heuristics: {FLAGS.heuristics}, use_density: {FLAGS.use_density}, beta_0: {FLAGS.h_beta_0}, alpha: {FLAGS.h_alpha}, lambda: {FLAGS.h_lambda}, seed: {FLAGS.seed}, utd_ratio: {FLAGS.utd_ratio}, batch_size: {FLAGS.batch_size}' + '***\n'*3)
    print(f'env_name: {FLAGS.env_name}, log_dir: {log_dir}\n' + '***\n'*3)
    
    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)
        print(f'Model checkpoint directory: {chkpt_dir}')

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)
        print(f'Buffer checkpoint directory: {buffer_dir}')
    
    rng = jax.random.key(FLAGS.seed)
    key, rng = jax.random.split(rng, 2)
    ds = MinariDataset(ENVS_LIST[FLAGS.env_core], FLAGS.p_simple, FLAGS.p_medium, FLAGS.p_expert, key)
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
    N = int(FLAGS.batch_size * FLAGS.utd_ratio)
    
    observation, _ = env.reset(seed=FLAGS.seed)


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
    
    # * Build replay buffer
    if (len(replay_buffer) < N):
        for i in range(N - len(replay_buffer)):
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
                
    # * Training loop            
    for i in tqdm.tqdm(
        range(training_start_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
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

        # * offline, online and density
        batch = {}
        offline_batch = {}
        online_batch = {}
        density_correction = jnp.zeros((N,))
        if FLAGS.offline_ratio == 0.0:
            # * fully offline learning
            batch = ds.sample(N)
        elif FLAGS.offline_ratio == 1.0:
            # * fully online learning
            batch = replay_buffer.sample(N)
        else:
            # * mixed
            offline_batch = ds.sample(
                int(N * FLAGS.offline_ratio)
            )
            online_batch = replay_buffer.sample(
                int(N * (1 - FLAGS.offline_ratio))
            )
            batch = combine(offline_batch, online_batch)
        if "antmaze" in FLAGS.env_core:
            batch["rewards"] -= 1

        
        log_info = {}
        interweave_interval = FLAGS.interweave_a3rl + FLAGS.interweave_rlpd
        use_interweave_and_in_a3rl = FLAGS.use_interweave and (i % interweave_interval >= 0) and (i % interweave_interval <= FLAGS.interweave_a3rl - 1)
        if ((i > FLAGS.start_a3rl) and (FLAGS.use_density or FLAGS.heuristics != "none") and (use_interweave_and_in_a3rl or (not FLAGS.use_interweave))):
            # run A3RL: with density or some heuristics. can't run without both.
            if FLAGS.use_density:
                offline_dens = agent.get_density(offline_batch)
                log_offline_dens = jnp.log(offline_dens + 1e-5)
                log_online_dens = jnp.log(1e-5)
                density_correction = combine({"dens": log_offline_dens}, {"dens": jnp.zeros(log_online_dens.shape)})["dens"]
                log_info.update({
                    "log_off_dens_max": jnp.max(log_offline_dens),
                    "log_off_dens_min": jnp.min(log_offline_dens),
                    "log_off_dens_mean": jnp.mean(log_offline_dens),
                    "log_off_dens_75": jnp.percentile(log_offline_dens, 75),
                    "log_off_dens_50": jnp.median(log_offline_dens),
                    "log_off_dens_25": jnp.percentile(log_offline_dens, 25),
                })
            # * heuristics options
            heuristics = jnp.zeros((N,))
            if FLAGS.heuristics == "adv":
                heuristics = agent.get_advantage(batch)
            elif FLAGS.heuristics == "td":
                heuristics = agent.get_td(batch)
            
            log_info.update({
                "hrts_max": jnp.max(heuristics),
                "hrts_min": jnp.min(heuristics),
                "hrts_mean": jnp.mean(heuristics),
                "hrts_75": jnp.percentile(heuristics, 75),
                "hrts_50": jnp.median(heuristics),
                "hrts_25": jnp.percentile(heuristics, 25),
            })
            # * combining density and heuristics, using PER notation
            log_priority =  density_correction + FLAGS.h_lambda * heuristics
            alpha = FLAGS.h_alpha + (i - FLAGS.start_a3rl)/(FLAGS.max_steps - FLAGS.start_a3rl) * (FLAGS.h_alpha * FLAGS.h_alpha_final_p - FLAGS.h_alpha)
            score = alpha * log_priority
            Prob = jax.nn.softmax(score)
            kl = -jnp.log(N) - jnp.sum(jnp.log(Prob))/N
            
            # * annealing
            beta = FLAGS.h_beta_0 + i/FLAGS.max_steps * (1 - FLAGS.h_beta_0)
            importance_sampling_weights = (1/N * 1/Prob) ** beta
            log_info.update({
                "score_max": jnp.max(score),
                "score_min": jnp.min(score),
                "score_mean": jnp.mean(score),
                "score_75": jnp.percentile(score, 75),
                "score_50": jnp.median(score),
                "score_25": jnp.percentile(score, 25),
                "N*p_max": jnp.max(Prob) * N,
                "N*p_min": jnp.min(Prob) * N,
                "kl_to_uniform": kl,
            })
            rng, key = jax.random.split(rng)
            sampled_indices = jax.random.choice(key, 
                                                N,
                                                shape=(N,),
                                                p=Prob, replace=True)

            batch = {k: v[sampled_indices] for k, v in batch.items()}
            importance_sampling_weights = importance_sampling_weights[sampled_indices]
            # sum to utd_ratio
            importance_sampling_weights *= FLAGS.utd_ratio / jnp.sum(importance_sampling_weights)
        else:
            # regular RLPD
            importance_sampling_weights = jnp.ones((N,)) * 1/N * FLAGS.utd_ratio

        log_info.update({
            "isweights_max": jnp.max(importance_sampling_weights),
            "isweights_min": jnp.min(importance_sampling_weights),
            "isweights_75": jnp.percentile(importance_sampling_weights, 75),
            "isweights_50": jnp.median(importance_sampling_weights),
            "isweights_mean": jnp.mean(importance_sampling_weights),
            "isweights_25": jnp.percentile(importance_sampling_weights, 25),
        })
        agent, update_info = agent.update(batch, FLAGS.utd_ratio, importance_sampling_weights)
        
        if FLAGS.use_density:
            agent, density_update_info = agent.split_update(offline_batch, online_batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i)
            for k, v in log_info.items():
                wandb.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            print(f'****\n****\n****\nEvaluating at {i}****\n****\n****\n')
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
            )
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
