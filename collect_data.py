import argparse
import datetime
import functools

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.wrappers.rescale_action import RescaleAction

EPISODE_STEPS = 100


def controller():
    """God help me, why not just use a class for this?"""
    action = 0.0
    while True:
        keys = pygame.key.get_pressed()
        action += (keys[pygame.K_RIGHT]) * 0.25
        action -= (keys[pygame.K_LEFT]) * 0.25
        action = np.clip(action, -1.0, 1.0)
        yield action


class Buffer:
    """A buffer of past transitions."""

    def __init__(self, trials, num_episodes, observation_shape, action_shape):
        self.observations = np.empty(
            (trials, num_episodes, EPISODE_STEPS, *observation_shape), dtype=np.float32
        )
        self.actions = np.empty(
            (trials, num_episodes, EPISODE_STEPS, *action_shape), dtype=np.float32
        )
        self.rewards = np.empty((trials, num_episodes, EPISODE_STEPS), dtype=np.float32)

    def append(self, trial, episode, step, observation, action, reward):
        self.observations[trial, episode, step] = observation
        self.actions[trial, episode, step] = action
        self.rewards[trial, episode, step] = reward


def trial(num_episodes, rod_length, append_fn):
    """Collects episodes for a given rod length variable."""
    env = gym.make(
        "Pendulum-v1",
        g=rod_length,
        render_mode="human",
        max_episode_steps=EPISODE_STEPS,
    )
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env.unwrapped.l = rod_length  # type: ignore  # noqa: E741
    episode_count = 0
    step = 0
    obs, _ = env.reset()
    gen = controller()
    while episode_count < num_episodes:
        action = next(gen)
        next_obs, reward, _, truncated, _ = env.step(action)
        append_fn(episode_count, step, obs, action, reward)
        step += 1
        obs = next_obs.copy()
        if truncated:
            episode_count += 1
            step = 0
            obs, _ = env.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    dummy = gym.make("Pendulum-v1")
    data = Buffer(
        args.num_trials,
        args.num_episodes,
        dummy.observation_space.shape,
        dummy.action_space.shape,
    )
    rng = np.random.default_rng(args.seed)
    for trial_id, gravity in enumerate(rng.uniform(0.1, 2, args.num_trials)):
        append = functools.partial(data.append, trial_id)
        trial(args.num_episodes, gravity, append)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    with open(
        f"data-{args.num_trials}-{args.num_episodes}-{now_str}.npz",
        "wb",
    ) as f:
        np.savez(
            f, observation=data.observations, action=data.actions, reward=data.rewards
        )


if __name__ == "__main__":
    main()
