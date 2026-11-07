"""Monte Carlo control: learns Q(s, a) with an epsilon-greedy policy."""

import numpy as np

from core.models import Environment
from utils.policy_estimation.schedules import decay_schedule

from .exploration import epsilon_greedy


def mc_control(env: Environment,
               gamma: float = 1.0,
               init_alpha: float = 0.5,
               min_alpha: float = 0.01,
               alpha_decay_ratio: float = 0.5,
               init_epsilon: float = 1.0,
               min_epsilon: float = 0.1,
               epsilon_decay_ratio: float = 0.9,
               n_episodes: int = 1000,
               max_steps: int = 200,
               first_visit: bool = True):
    """Returns (Q, V_track, pi, Q_track, pi_track)."""
    n_states = env.n_states
    n_actions = len(env.actions)

    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    pi_track = []

    Q = np.zeros((n_states, n_actions))
    Q_track = np.zeros((n_episodes, n_states, n_actions))

    for episode in range(n_episodes):
        # Generate an episode; restart if it is truncated before reaching a terminal state
        done = False
        while not done:
            state, trajectory = env.reset(), []
            for _ in range(max_steps):
                action = epsilon_greedy(state, Q, epsilons[episode])
                result = env.step(action)
                trajectory.append((state, action, result.reward))
                done = result.done
                state = result.next_state
                if done:
                    break

        rewards = np.array([reward for _, _, reward in trajectory])

        visited = np.zeros((n_states, n_actions), dtype=bool)
        for t, (state, action, _) in enumerate(trajectory):
            if visited[state, action] and first_visit:
                continue
            visited[state, action] = True

            n_steps = len(trajectory) - t
            G = np.sum(discounts[:n_steps] * rewards[t:])

            Q[state, action] += alphas[episode] * (G - Q[state, action])

        Q_track[episode] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V_track = np.max(Q_track, axis=2)  # (n_episodes, n_states): V(s) = max_a Q(s, a)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V_track, pi, Q_track, pi_track
