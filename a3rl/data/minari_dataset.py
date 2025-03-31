from typing import Dict, Optional, Tuple, Union, List

import numpy as np
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
    def __init__(self, dataset_id = str, clip_to_eps: bool = True, eps: float = 1e-5):
        """
        Convert a Minari dataset (episode-based) to interaction-based.
        """
        try:
            self.minari_dataset = minari.load_dataset(dataset_id, download=True)
        except Exception as e:
            raise ValueError(f"Failed to load Minari dataset {dataset_id}: {e}") from e

        all_observations = []
        all_next_observations = []
        all_actions = []
        all_rewards = []
        all_terminations = []
        all_truncations = []
        
        for episode in self.minari_dataset:
            all_observations.extend(episode.observations[:-1])
            all_next_observations.extend(episode.observations[1:])
            all_actions.extend(episode.actions)
            all_rewards.extend(episode.rewards)
            all_terminations.extend(episode.terminations)
            all_truncations.extend(episode.truncations)

        print(f"Number of observations: {len(all_observations)}")
        print(f"Number of next observations: {len(all_next_observations)}")
        print(f"Number of actions: {len(all_actions)}")
        print(f"Number of rewards: {len(all_rewards)}")
        print(f"Number of terminations: {len(all_terminations)}")
        print(f"Number of truncations: {len(all_truncations)}")
        
        dataset_dict = {
            "observations": np.array(all_observations),
            "next_observations": np.array(all_next_observations),
            "actions": np.array(all_actions),
            "rewards": np.array(all_rewards),
            # "terminations": np.array(all_terminations),
            # "truncations": np.array(all_truncations)
        }
        
        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dataset_dict["masks"] = 1.0 - np.array(all_terminations).astype(np.float32)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        super().__init__(dataset_dict)
            