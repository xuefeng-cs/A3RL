"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from a3rl.agents.agent import Agent
from a3rl.agents.sac.temperature import Temperature
from a3rl.data.dataset import DatasetDict
from a3rl.distributions import TanhNormal
from a3rl.networks import (
    MLP,
    Ensemble,
    StateActionValue,
    subsample_ensemble,
    StateActionValueDensity
)

import time

# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


class SACLearnerPriority(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    density: TrainState
    actor: TrainState
    tau: float = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    target_entropy: float =  struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        density_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.key(seed)
        rng, actor_key, critic_key, temp_key, density_key = jax.random.split(rng, 5)

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        
        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
            use_pnorm=use_pnorm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )
        
        density_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
        )
        density_def = StateActionValueDensity(base_cls=density_base_cls)
        density_params = density_def.init(density_key, observations, actions)["params"]
        density = TrainState.create(
            apply_fn=density_def.apply,
            params=density_params,
            tx=optax.adam(learning_rate=density_lr)
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            density=density,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
        )

    def update_actor(self, batch: DatasetDict, importance_sampling_weights: jnp.ndarray) -> Tuple[Agent, Dict[str, float]]:
        key, key2, rng = jax.random.split(self.rng, 3)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            )@importance_sampling_weights
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs@importance_sampling_weights}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    @jax.jit
    def get_advantage(self, batch: DatasetDict) -> jnp.ndarray:
        key, key2, rng = jax.random.split(self.rng, 3)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )
        # print(f'shape of batch_observations: {batch["observations"].shape}, shape of batch_actions: {batch["actions"].shape}')
        qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["observations"],
            batch["actions"],
        )
        # * take min over ensemble
        qs = qs.min(axis = 0)

        # * now calculate value    
        policy_dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["observations"]
        )
        sampled_actions = policy_dist.sample(sample_shape=(100,), seed=key2)
        sampled_qs = jax.vmap(
            lambda batch_actions: self.target_critic.apply_fn(
                {"params": target_params},
                batch["observations"],
                batch_actions,
            ),
            in_axes = 0,
            out_axes = 0
        )(sampled_actions)
        # * sampled_qs.shape = (20, 2, 5120), 2 from ensemble
        # * take min over ensemble as consistent
        # * then average over num_samples
        sampled_qs_ensembled = sampled_qs.min(axis = 1)
        sampled_vs = sampled_qs_ensembled.mean(axis = 0)        
        # * now calculate advantage
        return qs - sampled_vs

    @jax.jit
    def get_td(self, batch: DatasetDict) -> jnp.ndarray:
        key, rng = jax.random.split(self.rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )
        next_actions = dist.sample(seed=key, sample_shape=(100,))
        sampled_qs = jax.vmap(
            lambda batch_actions: self.target_critic.apply_fn(
                {"params": target_params},
                batch["next_observations"],
                batch_actions,
            )
        )(next_actions)
        sampled_qs_ensembled = sampled_qs.min(axis = 1)
        # if self.backup_entropy:
        #     next_log_probs = dist.log_prob(next_actions)
        #     print(f'Shape of next_log_probs: {next_log_probs.shape}')
        #     sampled_qs_ensembled -= self.temp.apply_fn(
        #         {"params" : self.temp.params}
        #     ) * next_log_probs
        #     print(f'Shape of sampled_qs_ensembled post modification: {sampled_qs_ensembled.shape}')
        
        next_q = sampled_qs_ensembled.mean(axis = 0)
        
        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q
        
        key, rng = jax.random.split(rng)
        
        q = self.critic.apply_fn(
            {"params": self.critic.params},
            batch["observations"],
            batch["actions"]
        )
        
        q = q.mean(axis = 0)

        return target_q - q

    def update_critic(self, batch: DatasetDict, importance_sampling_weights: jnp.ndarray) -> Tuple[TrainState, Dict[str, float]]:
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )
        key, rng = jax.random.split(self.rng)
        next_actions = dist.sample(seed=key)

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((((qs - target_q) ** 2)) @ importance_sampling_weights).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": (qs @ importance_sampling_weights).mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def get_density(self, batch: DatasetDict) -> jnp.ndarray:
        return self.density.apply_fn(
            {"params": self.density.params},
            batch["observations"],
            batch["actions"]
        )

    def update_density(self, offline_minibatch: DatasetDict, online_minibatch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        def density_loss_fn(density_params) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, float]]:
            # from line 58 to 83 of off2on/networks.py
            offline_weight = self.density.apply_fn(
                {"params": density_params},
                offline_minibatch["observations"],
                offline_minibatch["actions"]
            )
        
            offline_f_star = -jnp.log(2.0 / (offline_weight + 1) + 1e-10)
            
            online_weight = self.density.apply_fn(
                {"params": density_params},
                online_minibatch["observations"],
                online_minibatch["actions"]
            )
            
            online_f_prime = jnp.log(2 * online_weight / (online_weight + 1) + 1e-10)
            
            weight_loss = jnp.mean(offline_f_star - online_f_prime)
            
            metrics = {
                "offline_weight": jnp.mean(offline_weight),
                "online_weight": jnp.mean(online_weight),
            }

            return (weight_loss, metrics)
        
        grads, density_info = jax.grad(density_loss_fn, has_aux=True)(self.density.params)
        density = self.density.apply_gradients(grads = grads)
        
        return self.replace(density=density), density_info

    
    @partial(jax.jit, static_argnames="utd_ratio")
    def split_update(self, offline_batch: DatasetDict, online_batch: DatasetDict, utd_ratio: int):
        new_agent = self
        for i in range(utd_ratio):
            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i+1)]

            offline_minibatch = jax.tree_util.tree_map(slice, offline_batch)
            online_minibatch = jax.tree_util.tree_map(slice, online_batch)

            new_agent, density_info = new_agent.update_density(offline_minibatch, online_minibatch)
        
        return new_agent, density_info
        
    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int, importance_sampling_weights: jnp.ndarray):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            mini_importance_sampling_weights = slice(importance_sampling_weights)
            new_agent, critic_info = new_agent.update_critic(mini_batch, mini_importance_sampling_weights)

        new_agent, actor_info = new_agent.update_actor(mini_batch, mini_importance_sampling_weights)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}