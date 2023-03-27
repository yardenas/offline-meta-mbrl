from typing import Union

import equinox as eqx
import jax
import numpy as np
from gymnasium import spaces
from numpy import typing as npt
from omegaconf import DictConfig
from optax import OptState

from sadam import metrics as m
from sadam.logging import TrainingLogger
from sadam.models import Model
from sadam.replay_buffer import ReplayBuffer
from sadam.trajectory import TrajectoryData
from sadam.utils import Learner

FloatArray = npt.NDArray[Union[np.float32, np.float64]]


class SAdaM:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        self.prng = jax.random.PRNGKey(config.training.seed)
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
        self.model = Model(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            sequence_length=config.sadam.replay_buffer.sequence_length,
            key=self.prng,
            **config.sadam.model,
        )
        self.model_learner = Learner(self.model, config.sadam.model_optimizer)

    def __call__(
        self,
        observation: FloatArray,
    ) -> FloatArray:
        """
        Compute the next action based on the observation, update internal state
        as needed.
        """
        # normalized_obs = _normalize(
        #     observation, self.obs_normalizer.result.mean,
        # self.obs_normalizer.result.std
        # )
        return (
            np.ones((observation.shape[0], self.replay_buffer.action.shape[-1])) * 5.0
        )

    def observe(self, trajectory: TrajectoryData):
        """
        Observe a trajectory, update internal state as needed.
        """
        self.obs_normalizer.update_state(
            np.concatenate(
                [trajectory.observation, trajectory.next_observation[:, -1:]], axis=1
            ),
            axis=(0, 1),
        )
        standardized_obs = _normalize(
            trajectory.observation,
            self.obs_normalizer.result.mean,
            self.obs_normalizer.result.std,
        )
        standardized_next_obs = _normalize(
            trajectory.next_observation,
            self.obs_normalizer.result.mean,
            self.obs_normalizer.result.std,
        )
        self.replay_buffer.add(
            TrajectoryData(
                standardized_obs,
                standardized_next_obs,
                trajectory.action,
                trajectory.reward,
                trajectory.cost,
            )
        )
        self.train()

    def train(self):
        for batch in self.replay_buffer.sample(self.config.sadam.update_steps):
            print("dddds")
            loss, self.model, self.model_learner.state = self.update_model(
                self.model, self.model_learner.state, batch
            )

    @eqx.filter_jit
    def update_model(self, model: Model, opt_state: OptState, batch: TrajectoryData):
        def loss(
            model,
            state_sequence,
            action_sequence,
            next_state_sequence,
            reward_sequence,
        ):
            preds = jax.vmap(lambda s, a: model(s, a, convolve=True))(
                state_sequence, action_sequence
            )[1]
            state_loss = (preds.next_state - next_state_sequence) ** 2
            reward_loss = (preds.reward.squeeze(-1) - reward_sequence) ** 2
            return 0.5 * (state_loss.mean() + reward_loss.mean())

        loss_fn = eqx.filter_value_and_grad(loss)
        loss, grads = loss_fn(
            model, batch.observation, batch.action, batch.next_observation, batch.reward
        )
        new_model, new_opt_state = self.model_learner.grad_step(model, grads, opt_state)
        return loss, new_model, new_opt_state

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


def _normalize(
    observation: FloatArray,
    mean: FloatArray,
    std: FloatArray,
) -> FloatArray:
    diff = observation - mean
    return diff / (std + 1e-8)
