from types import SimpleNamespace

import numpy as np
from numpy import typing as npt

from sadam.logging import TrainingLogger
from sadam.trajectory import Trajectory


class SAdaM:
    def __init__(self, config: SimpleNamespace, logger: TrainingLogger):
        self.config = config
        self.logger = logger

    def __call__(
        self,
        observation: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """
        Compute the next action based on the observation, update internal state
        as needed.
        """
        return np.ones_like(observation)

    def observe(self, transition: Trajectory):
        """
        Observe a trajectory, update internal state as needed.
        """
        pass

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
