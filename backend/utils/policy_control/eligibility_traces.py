"""TD control with eligibility traces: Sarsa(lambda) and Watkins's Q(lambda)."""

import numpy as np

from core.models import Environment
from utils.policy_estimation.schedules import decay_schedule

from .exploration import epsilon_greedy


def sarsa_lambda(env: Environment,
                 gamma: float = 1.0,
                 init_alpha: float = 0.5,
                 min_alpha: float = 0.01,
                 alpha_decay_ratio: float = 0.5,
                 init_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 epsilon_decay_ratio: float = 0.9,
                 n_episodes: int = 1000,
                 max_steps: int = 200,
                 lambda_: float = 0.5,
                 replacing_traces: bool = True):
    """Sarsa(lambda) with accumulating or replacing (clipped to 1) traces.

    Returns (Q, V_track, pi, Q_track, pi_track).
    """
    n_states = env.n_states
    n_actions = len(env.actions)

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    pi_track = []

    Q = np.zeros((n_states, n_actions))
    Q_track = np.zeros((n_episodes, n_states, n_actions))
    E = np.zeros((n_states, n_actions))  # Eligibility traces

    for e in range(n_episodes):
        E.fill(0)
        state, done = env.reset(), False
        action = epsilon_greedy(state, Q, epsilons[e])

        step = 0
        while not done and step < max_steps:
            result = env.step(action)
            next_action = epsilon_greedy(result.next_state, Q, epsilons[e])

            td_target = result.reward + gamma * Q[result.next_state, next_action] * (not result.done)
            td_error = td_target - Q[state, action]

            E[state, action] += 1
            if replacing_traces:
                E.clip(0, 1, out=E)

            Q += alphas[e] * td_error * E
            E *= gamma * lambda_

            state, action, done = result.next_state, next_action, result.done
            step += 1

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V_track = np.max(Q_track, axis=2)  # (n_episodes, n_states): V(s) = max_a Q(s, a)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V_track, pi, Q_track, pi_track


def q_lambda(env: Environment,
             gamma: float = 1.0,
             init_alpha: float = 0.5,
             min_alpha: float = 0.01,
             alpha_decay_ratio: float = 0.5,
             init_epsilon: float = 1.0,
             min_epsilon: float = 0.1,
             epsilon_decay_ratio: float = 0.9,
             n_episodes: int = 1000,
             max_steps: int = 200,
             lambda_: float = 0.5,
             replacing_traces: bool = True):
    """Watkins's Q(lambda): traces are cut whenever a non-greedy action is taken.

    Returns (Q, V_track, pi, Q_track, pi_track).
    """
    n_states = env.n_states
    n_actions = len(env.actions)

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    pi_track = []

    Q = np.zeros((n_states, n_actions))
    Q_track = np.zeros((n_episodes, n_states, n_actions))
    E = np.zeros((n_states, n_actions))  # Eligibility traces

    for e in range(n_episodes):
        E.fill(0)
        state, done = env.reset(), False
        action = epsilon_greedy(state, Q, epsilons[e])

        step = 0
        while not done and step < max_steps:
            result = env.step(action)
            next_action = epsilon_greedy(result.next_state, Q, epsilons[e])
            next_action_is_greedy = Q[result.next_state, next_action] == Q[result.next_state].max()

            td_target = result.reward + gamma * Q[result.next_state].max() * (not result.done)
            td_error = td_target - Q[state, action]

            if replacing_traces:
                E[state].fill(0)
            E[state, action] += 1

            Q += alphas[e] * td_error * E

            if next_action_is_greedy:
                E *= gamma * lambda_
            else:
                E.fill(0)

            state, action, done = result.next_state, next_action, result.done
            step += 1

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V_track = np.max(Q_track, axis=2)  # (n_episodes, n_states): V(s) = max_a Q(s, a)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V_track, pi, Q_track, pi_track
