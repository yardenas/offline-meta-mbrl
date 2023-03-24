from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from sadam import utils
from sadam.episodic_async_env import EpisodicAsync
from sadam.iteration_summary import IterationSummary
from sadam.sadam import SAdaM
from sadam.trajectory import Trajectory, Transition


def interact(
    agent: SAdaM,
    environment: EpisodicAsync,
    num_episodes: int,
    train: bool,
    render_episodes: int = 0,
    render_mode: str = "rgb_array",
):
    observations = environment.reset()
    episode_count = 0
    episodes: list[Trajectory] = []
    trajectory = Trajectory()
    with tqdm(total=num_episodes) as pbar:
        while episode_count < num_episodes:
            if render_episodes:
                trajectory.frames.append(environment.render(render_mode))
            actions = agent(observations)
            next_observations, rewards, done, infos = environment.step(actions)
            costs = np.array([info.get("cost", 0) for info in infos])
            transition = Transition(
                observations,
                next_observations,
                actions,
                rewards,
                costs,
            )
            trajectory.transitions.append(transition)
            observations = next_observations
            if done.all():
                agent.observe(trajectory)
                # on_episode_end(episodes[-1], train, adapt)
                render_episodes = max(render_episodes - 1, 0)
                observations = environment.reset()
                episodes.append(trajectory)
                trajectory = Trajectory()
                pbar.update(1)
                episode_count += 1
    return episodes


def epoch(
    agent: SAdaM,
    env: EpisodicAsync,
    tasks: Iterable[Any],
    episodes_per_task: int,
    train: bool,
    render_episodes: int = 0,
) -> IterationSummary:
    summary = IterationSummary()
    batches = list(utils.grouper(tasks, env.num_envs))
    for batch in batches:
        assert len(batch) == env.num_envs
        tasks = zip(*batch)
        env.reset(options={"task": tasks})
        samples = interact(
            agent,
            env,
            episodes_per_task,
            train=train,
            render_episodes=render_episodes,
        )
        summary.extend(samples)
    return summary
