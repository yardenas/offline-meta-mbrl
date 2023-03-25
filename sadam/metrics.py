from typing import Any, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt


class Metrics(NamedTuple):
    mean: npt.NDArray[Any]
    std: npt.NDArray[Any]
    min: npt.NDArray[Any]
    max: npt.NDArray[Any]


class MetricsAccumulator:
    def __init__(self):
        self._state = None
        self._count = 0
        self._m2 = None

    def update_state(
        self, sample: float | npt.NDArray[Any], axis: int | Sequence[int] = 0
    ):
        if isinstance(sample, float):
            sample = np.array([sample])
        new_mean = sample.mean(axis=axis)
        new_std = sample.std(axis=axis)
        new_min = sample.min(axis=axis)
        new_max = sample.max(axis=axis)
        if self._state is None:
            self._state = Metrics(new_mean, new_std, new_min, new_max)
        else:
            delta = new_mean - self._state.mean
            updated_mean = self._state.mean + delta / self._count
            delta2 = new_mean - updated_mean
            self._m2 = (
                self._m2 + delta2 * delta2 if self._m2 is not None else delta2**2
            )
            self._state = Metrics(
                updated_mean,
                self._m2 / (self._count - 1) if self._count > 1 else new_std,
                np.minimum(self._state.min, new_min),
                np.maximum(self._state.max, new_max),
            )
        self._count += 1

    @property
    def result(self):
        if self._state is not None:
            state = self._state
        else:
            state = Metrics(
                np.zeros((1,)),
                np.ones((1,)),
                -np.inf * np.ones((1,)),
                np.inf * np.ones((1,)),
            )
        return state

    def reset_states(self):
        self._state = None
        self._count = 0
        self._m2 = None
