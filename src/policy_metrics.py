"""Monte-Carlo evaluation metrics + a step-by-step visualizer.

step() now returns an ActionResult (prob=None = already sampled).
"""

import time

import numpy as np

from models import Environment, Policy


def rollouts(env: Environment, pi: Policy, n_episodes: int = 100,
             max_steps: int = 200, seed: int | None = None) -> list[tuple[float, int]]:
    """Run n_episodes under pi; return [(total_return, final_state), ...].

    If the env exposes `seed()`, it is reseeded first so the same seed
    always replays the same episode stream.
    """
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)

    data = []
    for _ in range(n_episodes):
        state = env.reset()
        total, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            result = env.step(pi(state))          # ActionResult
            state = result.next_state
            total += result.reward
            done = result.done
            steps += 1
        data.append((total, state))
    return data


def mean_return(env: Environment, pi: Policy, n_episodes: int = 100,
                max_steps: int = 200, seed: int | None = None) -> float:
    data = rollouts(env, pi, n_episodes, max_steps, seed)
    return float(np.mean([r for r, _ in data])) if data else 0.0


def std_return(env: Environment, pi: Policy, n_episodes: int = 100,
               max_steps: int = 200, seed: int | None = None) -> float:
    data = rollouts(env, pi, n_episodes, max_steps, seed)
    return float(np.std([r for r, _ in data])) if len(data) > 1 else 0.0


def probability_success(env: Environment, pi: Policy, goal_state: int | None = None,
                        n_episodes: int = 100, max_steps: int = 200,
                        seed: int | None = None) -> float:
    if goal_state is None:
        goal_state = getattr(env, "goal", None)
        if goal_state is None:
            raise TypeError("goal_state required, or env must expose .goal")
    data = rollouts(env, pi, n_episodes, max_steps, seed)
    return sum(1 for _, s in data if s == goal_state) / len(data) if data else 0.0


def play_episode(env: Environment, pi: Policy, max_steps: int = 200,
                 delay: float = 0.3) -> tuple[float, int]:
    """Run ONE episode, redrawing the board after every action.

    Uses env.display(), so you can watch the agent move (and slip).
    Returns (total_return, final_state).
    """
    env.reset()
    done, steps, total = False, 0, 0.0
    print("reset:")
    env.display(show_legend=True)

    while not done and steps < max_steps:
        a = pi(env.state)
        result = env.step(a)
        total += result.reward
        steps += 1
        done = result.done
        print(f"step {steps}: take {env.actions[a].name}  "
              f"-> reward={result.reward}, done={result.done}")
        env.display(show_indices=False, show_legend=False)
        if delay:
            time.sleep(delay)

    print(f"episode finished: return = {total}, final state = {env.state}\n")
    return total, env.state