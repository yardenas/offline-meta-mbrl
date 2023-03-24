from typing import Iterable, Optional

from hydra import compose, initialize

from sadam.trainer import Trainer


def test_training():
    with initialize(version_base=None, config_path="../sadam/"):
        cfg = compose(
            config_name="config",
        )
    if not cfg.training.jit:
        from jax.config import config as jax_config  # pyright: ignore

        jax_config.update("jax_disable_jit", True)

    def make_env():
        import gymnasium as gym

        env = gym.make("Pendulum-v1")
        return env

    def task_sampler(dummy: int, dummy2: Optional[bool] = False) -> Iterable[int]:
        while True:
            yield 1

    with Trainer(cfg, make_env, task_sampler) as trainer:
        trainer.train(epochs=1)
