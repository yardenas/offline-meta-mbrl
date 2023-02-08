from typing import Iterator, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
from tensorflow import data as tfd


class Trajectory(NamedTuple):
    o: npt.NDArray[np.float32]
    a: npt.NDArray[np.float32]
    r: npt.NDArray[np.float32]


class ReplayBuffer:
    def __init__(
        self,
        seed: int,
        observation_shape: Tuple[int],
        action_shape: Tuple[int],
        num_tasks: int,
        num_episodes: int,
        max_length: int,
        meta_batch_size: int,
        batch_size: int,
        sequence_length: int,
        observations: Optional[npt.NDArray[np.float32]] = None,
        actions: Optional[npt.NDArray[np.float32]] = None,
        rewards: Optional[npt.NDArray[np.float32]] = None,
    ):
        self.idx = 0
        self.episode_id = 0
        self.observation = (
            observations
            if observations is not None
            else np.zeros(
                (
                    num_tasks,
                    num_episodes,
                    max_length + 1,
                )
                + observation_shape,
                dtype=np.float32,
            )
        )
        self.action = (
            actions
            if actions is not None
            else np.zeros(
                (
                    num_tasks,
                    num_episodes,
                    max_length,
                )
                + action_shape,
                dtype=np.float32,
            )
        )
        self.reward = (
            rewards
            if rewards is not None
            else np.zeros(
                (
                    num_tasks,
                    num_episodes,
                    max_length,
                ),
                dtype=np.float32,
            )
        )
        self._valid_tasks = observations.shape[0] if observations is not None else 0
        self._valid_episodes = observations.shape[1] if observations is not None else 0
        self.rs = np.random.RandomState(seed)
        example = next(
            iter(self._sample_batch(meta_batch_size, batch_size, sequence_length))
        )
        self._generator = lambda: self._sample_batch(
            meta_batch_size, batch_size, sequence_length
        )
        self._dataset = _make_dataset(self._generator, example)

    def _sample_batch(
        self,
        meta_batch_size: int,
        batch_size: int,
        sequence_length: int,
    ) -> Iterator[Trajectory]:
        time_limit = self.observation.shape[2]
        valid_tasks = (
            self._valid_tasks if self._valid_tasks > 0 else self.observation.shape[0]
        )
        valid_episodes = (
            self._valid_episodes
            if self._valid_episodes > 0
            else self.observation.shape[1]
        )
        assert (
            time_limit > sequence_length
            and valid_tasks >= meta_batch_size
            and valid_episodes >= batch_size
        )
        while True:
            if (time_limit - sequence_length - 1) > 0:
                low = self.rs.choice(time_limit - sequence_length - 1, batch_size)
            else:
                low = np.zeros(batch_size, dtype=np.int32)
            timestep_ids = low[..., None] + np.tile(
                np.arange(sequence_length + 1),
                (meta_batch_size, batch_size, 1),
            )
            episode_ids = self.rs.choice(valid_episodes, size=batch_size)
            task_ids = self.rs.choice(valid_tasks, size=meta_batch_size)
            # Sample a sequence of length H for the actions, rewards and costs,
            # and a length of H + 1 for the observations (which is needed for
            # bootstrapping)
            a, r = [
                x[
                    task_ids[:, None, None],
                    episode_ids[None, :, None],
                    timestep_ids[..., :-1],
                ]
                for x in (
                    self.action,
                    self.reward,
                )
            ]
            o = self.observation[
                task_ids[:, None, None],
                episode_ids[None, :, None],
                timestep_ids,
            ]
            yield Trajectory(o, a, r)

    def sample(self, n_batches: int) -> Iterator[Trajectory]:
        if self.empty:
            return
        for batch in self._dataset.take(n_batches):
            a = Trajectory(*map(lambda x: x.numpy(), batch))  # type: ignore
            yield a

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_dataset"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        example = next(iter(self._generator()))
        self._dataset = _make_dataset(self._generator, example)

    @property
    def empty(self):
        return self._valid_episodes == 0


def _make_dataset(generator, example):
    dataset = tfd.Dataset.from_generator(
        generator,
        *zip(*tuple((v.dtype, v.shape) for v in example)),
    )
    dataset = dataset.prefetch(10)
    return dataset
