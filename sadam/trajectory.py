from dataclasses import dataclass, field
from typing import NamedTuple, TypeAlias

import numpy as np
from numpy import typing as npt


class Transition(NamedTuple):
    observation: npt.NDArray[np.float32 | np.float64]
    next_observation: npt.NDArray[np.float32 | np.float64]
    action: npt.NDArray[np.float32 | np.float64]
    reward: npt.NDArray[np.float32 | np.float64]
    cost: npt.NDArray[np.float32 | np.float64]


TrajectoryData: TypeAlias = Transition


@dataclass
class Trajectory:
    transitions: list[Transition] = field(default_factory=list)
    frames: list[npt.NDArray[np.float32 | np.int8]] = field(default_factory=list)

    def __len__(self):
        return len(self.transitions)

    def as_numpy(self) -> TrajectoryData:
        # Transpose list of tuples to a tuple of lists,
        # this magic is possible since transition is a named tuple.
        # This allows us make lists of observations, actions, rewards, etc.,
        # instead of list of transitions.
        o, next_o, a, r, c = zip(*self.transitions)
        # Stack on axis=1 to keep batch dimension first, and time axis second.
        stack = lambda x: np.stack(x, axis=1)
        data = TrajectoryData(stack(o), stack(next_o), stack(a), stack(r), stack(c))
        return data