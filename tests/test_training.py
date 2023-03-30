from typing import Iterable, Optional

from hydra import compose, initialize

from sadam import training
from sadam.trainer import Trainer


def test_training():
    with initialize(version_base=None, config_path="../sadam/"):
        cfg = compose(
            config_name="config",
            overrides=[
                "training.time_limit=32",
                "training.episodes_per_task=1",
                "training.task_batch_size=5",
                "training.parallel_envs=5",
                "sadam.model.n_layers=1",
                "sadam.model.hidden_size=32",
                "sadam.model.hippo_n=8",
                "sadam.update_steps=1",
                "sadam.replay_buffer.sequence_length=16",
            ],
        )
    if not cfg.training.jit:
        from jax.config import config as jax_config  # pyright: ignore

        jax_config.update("jax_disable_jit", True)

    def make_env():
        import gymnasium as gym

        env = gym.make("Pendulum-v1")
        env._max_episode_steps = cfg.training.time_limit  # type: ignore

        return env

    def task_sampler(dummy: int, dummy2: Optional[bool] = False) -> Iterable[int]:
        for _ in range(cfg.training.task_batch_size):
            yield 1

    with Trainer(cfg, make_env, task_sampler) as trainer:
        trainer.train(epochs=1)


def test_model_learning():
    import jax
    import numpy as np

    with initialize(version_base=None, config_path="../sadam/"):
        cfg = compose(
            config_name="config",
            overrides=[
                "training.time_limit=32",
                "training.episodes_per_task=1",
                "training.task_batch_size=5",
                "training.parallel_envs=5",
                "training.render_episodes=0",
                "sadam.model.n_layers=2",
                "sadam.model.hidden_size=64",
                "sadam.model.hippo_n=16",
                "sadam.update_steps=100",
                "sadam.replay_buffer.sequence_length=30",
            ],
        )
    if not cfg.training.jit:
        from jax.config import config as jax_config  # pyright: ignore

        jax_config.update("jax_disable_jit", True)

    def make_env():
        import gymnasium as gym

        env = gym.make("Pendulum-v1")
        env._max_episode_steps = cfg.training.time_limit  # type: ignore

        return env

    def task_sampler(dummy: int, dummy2: Optional[bool] = False) -> Iterable[int]:
        for _ in range(cfg.training.task_batch_size):
            yield 1

    with Trainer(cfg, make_env, task_sampler) as trainer:
        assert trainer.agent is not None and trainer.env is not None
        from sadam.sadam import SAdaM

        SAdaM.__call__ = lambda self, observation: np.tile(
            trainer.env.action_space.sample(),  # type: ignore
            (
                cfg.training.task_batch_size,
                1,
            ),
        )
        trainer.train(epochs=5)
    agent = trainer.agent
    assert agent is not None
    trajectories = training.interact(agent, trainer.env, 1, False)[0].as_numpy()
    _, one_step_predictions = jax.vmap(agent.model)(
        trajectories.observation, trajectories.action
    )
    reward_mse = np.mean(
        (one_step_predictions.reward.squeeze(-1) - trajectories.reward) ** 2
    )
    obs_mse = np.mean(
        (one_step_predictions.next_state - trajectories.next_observation) ** 2
    )
    print(f"Reward MSE: {reward_mse}")
    print(f"Observation MSE: {obs_mse}")
