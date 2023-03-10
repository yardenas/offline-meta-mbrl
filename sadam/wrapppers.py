# type: ignore
from typing import Callable, Optional

import numpy as np
from gymnasium.core import Wrapper

from sadam.episodic_async_env import EpisodicAsync
from sadam.logging import TrainingLogger
from sadam.trajectory import Trajectory, Transition


class ActionRepeat(Wrapper):
    def __init__(self, env, repeat):
        assert repeat >= 1, "Expects at least one repeat."
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0.0
        total_cost = 0.0
        current_step = 0
        info = {"steps": 0}
        while current_step < self.repeat and not done:
            obs, reward, terminal, truncated, info = self.env.step(action)
            total_reward += reward
            total_cost += info.get("cost", 0.0)
            current_step += 1
        info["steps"] = current_step
        info["cost"] = total_cost
        return obs, total_reward, terminal, truncated, info


class LoggingWrapper:
    def __init__(
        self,
        env: EpisodicAsync,
        logger: TrainingLogger,
        callbacks: list[Callable[[Trajectory, TrainingLogger], None]],
    ):
        self.env = env
        self.logger = logger
        self.callbacks = callbacks
        self.steps = 0
        self.trajectory = None
        self.prev_obs = None

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        self.step += sum(i.get("steps", 1) for i in info.values())
        cost = np.array([info.get("cost", 0) for info in info])
        self.trajectory.append(Transition(self.prev_obs, obs, action, reward, cost))
        if terminal or truncated:
            for callback in self.callbacks:
                callback(self.trajectory, self.logger)
            self.trajectory = Trajectory()
        self.prev_obs = obs
        return obs, reward, terminal, truncated, info

    def reset(
        self, *, seed: Optional[int | list[int]] = None, options: Optional[dict] = None
    ):
        obs = self.env.reset(seed=seed, options=options)
        self.prev_obs = obs
        self.trajectory = Trajectory()
        return obs
