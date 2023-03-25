import numpy as np
from gymnasium import spaces
from numpy import typing as npt
from omegaconf import DictConfig

from sadam import metrics as m
from sadam.logging import TrainingLogger
from sadam.replay_buffer import ReplayBuffer
from sadam.trajectory import TrajectoryData

FloatArray = npt.NDArray[np.float32]


class SAdaM:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        self.config = config
        self.logger = logger
        self.obs_normalizer = m.MetricsAccumulator()
        self.replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit,
            seed=config.training.seed,
            precision=config.training.precision,
            **config.sadam.replay_buffer,
        )

    def __call__(
        self,
        observation: FloatArray,
    ) -> FloatArray:
        """
        Compute the next action based on the observation, update internal state
        as needed.
        """
        normalized_obs = _normalize(
            observation, self.obs_normalizer.result.mean, self.obs_normalizer.result.std
        )
        return np.ones_like(normalized_obs)

    def observe(self, trajectory: TrajectoryData):
        """
        Observe a trajectory, update internal state as needed.
        """
        self.obs_normalizer.update_state(
            np.concatenate(
                [trajectory.observation, trajectory.next_observation[:, -1:]]
            )
        )

    def __getstate__(self):
        """
        Define how the agent should be pickled.
        """
        state = self.__dict__.copy()
        del state["logger"]
        return state

    def __setstate__(self, state):
        """
        Define how the agent should be loaded.
        """
        self.__dict__.update(state)
        self.logger = TrainingLogger(self.config.log_dir)


def _normalize(observation: FloatArray, mean: float, std: float) -> FloatArray:
    diff = observation - mean
    return diff / (std + 1e-8)
