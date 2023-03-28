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
        self._state = Metrics(
            np.zeros((1,)),
            np.ones((1,)),
            np.empty((1,)),
            np.empty((1,)),
        )
        self._count = 0
        self._m2 = 0.0

    def update_state(
        self, sample: float | npt.NDArray[Any], axis: int | Sequence[int] = 0
    ):
        if isinstance(sample, float) or sample.ndim == 0:
            sample = np.array(
                [
                    sample,
                ]
            )
        batch_mean = sample.mean(axis=axis)
        batch_var = sample.std(axis=axis)
        batch_min = sample.min(axis=axis)
        batch_max = sample.max(axis=axis)
        self._count += 1
        delta = batch_mean - self._state.mean
        new_mean = self._state.mean + delta / self._count
        delta2 = batch_mean - new_mean
        self._m2 = self._m2 + delta * delta2
        new_stddev = np.sqrt(
            self._m2 / (self._count - 1) if self._count > 1 else batch_var
        )
        self._state = Metrics(
            new_mean,
            new_stddev,
            np.minimum(self._state.min if self._count > 1 else batch_min, batch_min),
            np.maximum(self._state.max if self._count > 1 else batch_max, batch_max),
        )

    @property
    def result(self):
        return self._state

    def reset_states(self):
        self._state = Metrics(
            np.zeros((1,)),
            np.ones((1,)),
            np.zeros((1,)),
            np.zeros((1,)),
        )
        self._count = 0
        self._m2 = 0.0
