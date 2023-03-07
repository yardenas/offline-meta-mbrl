import argparse
import datetime
import functools

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.classic_control import pendulum
from gymnasium.wrappers.rescale_action import RescaleAction
from gymnasium.wrappers.time_limit import TimeLimit

EPISODE_STEPS = 200


def controller():
    """God help me, why not just use a class for this?"""
    action = 0.0
    while True:
        keys = pygame.key.get_pressed()
        action += (keys[pygame.K_RIGHT]) * 0.25
        action -= (keys[pygame.K_LEFT]) * 0.25
        action = np.clip(action, -1.0, 1.0)
        yield action


class RotatedPendulum(pendulum.PendulumEnv):
    def __init__(self, render_mode="human", g=10.0, angle=0.0):
        super().__init__(render_mode, g)
        self.angle = angle

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l  # noqa E741
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = (
            pendulum.angle_normalize(th + self.angle) ** 2
            + 0.1 * thdot**2
            + 0.001 * (u**2)
        )

        newthdot = (
            thdot
            + (3 * g / (2 * l) * np.sin(th + self.angle) + 3.0 / (m * l**2) * u) * dt
        )
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, False, False, {}


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


def trial(num_episodes, gravity_angle, append_fn):
    """Collects episodes for a given rod length variable."""
    env = RotatedPendulum(angle=gravity_angle)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)  # type: ignore
    env = TimeLimit(env, max_episode_steps=EPISODE_STEPS)  # type: ignore
    episode_count = 0
    step = 0
    obs, _ = env.reset()
    gen = controller()
    while episode_count < num_episodes:
        action = next(gen)
        next_obs, reward, _, truncated, _ = env.step(action)
        append_fn(episode_count, step, obs, action, reward)
        step += 1
        obs = next_obs
        if truncated:
            episode_count += 1
            step = 0
            obs, _ = env.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1)
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
    for trial_id, gravity in enumerate(rng.uniform(-np.pi, np.pi, args.num_trials)):
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
