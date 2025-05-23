from typing import Dict, Optional, Tuple, Union, List

import jax.numpy as jnp
from jax import random
import minari

from a3rl.data.dataset import Dataset

def flatten_observation(obs, obs_space):
    flattened_obs = np.concatenate([
        np.asarray(obs[key]).ravel() 
        for key in obs_space.spaces.keys()
    ])
    print(flattened_obs.shape)  # Check the shape of the flattened observation

class MinariDataset(Dataset):
    """
    Minari dataset (with action space (-1, 1))
    """
    minari_dataset: minari.MinariDataset
    def __init__(self, dataset_ids: List[str], p_simple , p_medium , p_expert, rng, clip_to_eps: bool = True, eps: float = 1e-5):
        """
        Convert a Minari dataset (episode-based) to interaction-based.
        
        The class takes in a list of environments, and concatenates all of them.
        """
        all_observations = []
        all_next_observations = []
        all_actions = []
        all_rewards = []
        all_terminations = []
        all_truncations = []
        for dataset_id in dataset_ids:
            dataset_observations = []
            dataset_next_observations = []
            dataset_actions = []
            dataset_rewards = []
            dataset_terminations = []
            dataset_truncations = []

            key, rng = random.split(rng)
            try:
                self.minari_dataset = minari.load_dataset(dataset_id, download=True)
            except Exception as e:
                raise ValueError(f"Failed to load Minari dataset {dataset_id}: {e}") from e

            for episode in self.minari_dataset:
                dataset_observations.extend(episode.observations[:-1])
                dataset_next_observations.extend(episode.observations[1:])
                dataset_actions.extend(episode.actions)
                dataset_rewards.extend(episode.rewards)
                dataset_terminations.extend(episode.terminations)
                dataset_truncations.extend(episode.truncations)
            
            p = 1    
            if ("simple" in dataset_id or "human" in dataset_id):
                p = p_simple
            elif ("medium" in dataset_id or "cloned" in dataset_id):
                p = p_medium
            elif "expert" in dataset_id:
                p = p_expert
            
            # choose a random fraction of dataset
            dataset_indices = random.choice(key, len(dataset_observations), shape=(int(p * len(dataset_observations)),), replace=False)
            
            print(f"Getting {int(p * len(dataset_observations))} from dataset {dataset_id}")
            
            all_observations.extend([dataset_observations[i] for i in dataset_indices])
            all_next_observations.extend([dataset_next_observations[i] for i in dataset_indices])
            all_actions.extend([dataset_actions[i] for i in dataset_indices])
            all_rewards.extend([dataset_rewards[i] for i in dataset_indices])
            all_terminations.extend([dataset_terminations[i] for i in dataset_indices])
            all_truncations.extend([dataset_truncations[i] for i in dataset_indices])

        
        print(f"Number of observations: {len(all_observations)}")
        print(f"Number of next observations: {len(all_next_observations)}")
        print(f"Number of actions: {len(all_actions)}")
        print(f"Number of rewards: {len(all_rewards)}")
        print(f"Number of terminations: {len(all_terminations)}")
        print(f"Number of truncations: {len(all_truncations)}")

        # mix everything together
        key, rng = random.split(key)
        indices = random.permutation(key, len(all_observations))

        all_observations = [all_observations[i] for i in indices]
        all_next_observations = [all_next_observations[i] for i in indices]
        all_actions = [all_actions[i] for i in indices]
        all_rewards = [all_rewards[i] for i in indices]
        all_terminations = [all_terminations[i] for i in indices]
        all_truncations = [all_truncations[i] for i in indices]
        

        dataset_dict = {
            "observations": jnp.array(all_observations),
            "next_observations": jnp.array(all_next_observations),
            "actions": jnp.array(all_actions),
            "rewards": jnp.array(all_rewards),
            # "terminations": jnp.array(all_terminations),
            # "truncations": jnp.array(all_truncations)
        }
        
        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = jnp.clip(dataset_dict["actions"], -lim, lim)

        dataset_dict["masks"] = 1.0 - jnp.array(all_terminations).astype(jnp.float32)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(jnp.float32)

        super().__init__(dataset_dict)
            