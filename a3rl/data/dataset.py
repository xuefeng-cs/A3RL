from functools import partial
from random import sample
from typing import Dict, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict
from gymnasium.utils import seeding

from a3rl.types import DataType

DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, jnp.ndarray) or isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: jnp.ndarray) -> DatasetDict:
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, jnp.ndarray):
            new_v = v[index]
        else:
            raise TypeError("Unsupported type.")
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(
    dataset_dict: Union[jnp.ndarray, DatasetDict], indx: jnp.ndarray
) -> DatasetDict:
    if isinstance(dataset_dict, jnp.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        # Seeding similar to OpenAI Gym:
        # https://github.com/openai/gym/blob/master/gym/spaces/space.py#L46
        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, self._seed = seeding.np_random(seed)

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[jnp.ndarray] = None,
    ) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

    def sample_jax(self, batch_size: int, keys: Optional[Iterable[str]] = None):
        if not hasattr(self, "rng"):
            self.rng = jax.random.key(self._seed or 42)

            if keys is None:
                keys = self.dataset_dict.keys()

            jax_dataset_dict = {k: self.dataset_dict[k] for k in keys}
            jax_dataset_dict = jax.device_put(jax_dataset_dict)

            @jax.jit
            def _sample_jax(rng):
                key, rng = jax.random.split(rng)
                indx = jax.random.randint(
                    key, (batch_size,), minval=0, maxval=len(self)
                )
                return rng, jax.tree_map(
                    lambda d: jnp.take(d, indx, axis=0), jax_dataset_dict
                )

            self._sample_jax = _sample_jax

        self.rng, sample = self._sample_jax(self.rng)
        return sample

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        assert 0 < ratio and ratio < 1
        index = jnp.arange(len(self), dtype=jnp.int32)
        self.np_random.shuffle(index)
        train_index = index[: int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio) :]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)

    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        episode_starts = [0]
        episode_ends = []

        episode_return = 0
        episode_returns = []

        for i in range(len(self)):
            episode_return += self.dataset_dict["rewards"][i]

            if self.dataset_dict["dones"][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns

    def filter(
        self, take_top: Optional[float] = None, threshold: Optional[float] = None
    ):
        assert (take_top is None and threshold is not None) or (
            take_top is not None and threshold is None
        )

        (
            episode_starts,
            episode_ends,
            episode_returns,
        ) = self._trajectory_boundaries_and_returns()

        if take_top is not None:
            threshold = jnp.percentile(episode_returns, 100 - take_top)

        bool_indx = jnp.full((len(self),), False, dtype=bool)

        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_indx[episode_starts[i] : episode_ends[i]] = True

        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)

        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000):
        (_, _, episode_returns) = self._trajectory_boundaries_and_returns()
        self.dataset_dict["rewards"] /= jnp.max(episode_returns) - jnp.min(
            episode_returns
        )
        self.dataset_dict["rewards"] *= scaling
