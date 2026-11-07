"""Temporal-difference control: Sarsa, Q-learning and Double Q-learning."""

import numpy as np

from core.models import Environment
from utils.policy_estimation.schedules import decay_schedule

from .exploration import epsilon_greedy


def sarsa(env: Environment,
          gamma: float = 1.0,
          init_alpha: float = 0.5,
          min_alpha: float = 0.01,
          alpha_decay_ratio: float = 0.5,
          init_epsilon: float = 1.0,
          min_epsilon: float = 0.1,
          epsilon_decay_ratio: float = 0.9,
          n_episodes: int = 1000,
          max_steps: int = 200):
    """On-policy TD control. Returns (Q, V_track, pi, Q_track, pi_track)."""
    n_states = env.n_states
    n_actions = len(env.actions)

    pi_track = []

    Q = np.zeros((n_states, n_actions))
    Q_track = np.zeros((n_episodes, n_states, n_actions))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in range(n_episodes):
        state, done = env.reset(), False
        action = epsilon_greedy(state, Q, epsilons[e])

        step = 0
        while not done and step < max_steps:
            result = env.step(action)
            next_action = epsilon_greedy(result.next_state, Q, epsilons[e])

            td_target = result.reward + gamma * Q[result.next_state, next_action] * (not result.done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alphas[e] * td_error

            state, action, done = result.next_state, next_action, result.done
            step += 1

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V_track = np.max(Q_track, axis=2)  # (n_episodes, n_states): V(s) = max_a Q(s, a)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V_track, pi, Q_track, pi_track


def q_learning(env: Environment,
               gamma: float = 1.0,
               init_alpha: float = 0.5,
               min_alpha: float = 0.01,
               alpha_decay_ratio: float = 0.5,
               init_epsilon: float = 1.0,
               min_epsilon: float = 0.1,
               epsilon_decay_ratio: float = 0.9,
               n_episodes: int = 1000,
               max_steps: int = 200):
    """Off-policy TD control. Returns (Q, V_track, pi, Q_track, pi_track)."""
    n_states = env.n_states
    n_actions = len(env.actions)

    pi_track = []

    Q = np.zeros((n_states, n_actions))
    Q_track = np.zeros((n_episodes, n_states, n_actions))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in range(n_episodes):
        state, done = env.reset(), False

        step = 0
        while not done and step < max_steps:
            action = epsilon_greedy(state, Q, epsilons[e])
            result = env.step(action)

            q_learning_target = result.reward + gamma * np.max(Q[result.next_state]) * (not result.done)
            q_learning_error = q_learning_target - Q[state, action]
            Q[state, action] += alphas[e] * q_learning_error

            state, done = result.next_state, result.done
            step += 1

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V_track = np.max(Q_track, axis=2)  # (n_episodes, n_states): V(s) = max_a Q(s, a)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V_track, pi, Q_track, pi_track


def double_q_learning(env: Environment,
                      gamma: float = 1.0,
                      init_alpha: float = 0.5,
                      min_alpha: float = 0.01,
                      alpha_decay_ratio: float = 0.5,
                      init_epsilon: float = 1.0,
                      min_epsilon: float = 0.1,
                      epsilon_decay_ratio: float = 0.9,
                      n_episodes: int = 1000,
                      max_steps: int = 200):
    """Double Q-learning; Q is the mean of Q1 and Q2.

    Returns (Q, V_track, pi, Q_track, pi_track).
    """
    n_states = env.n_states
    n_actions = len(env.actions)

    pi_track = []

    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))
    Q1_track = np.zeros((n_episodes, n_states, n_actions))
    Q2_track = np.zeros((n_episodes, n_states, n_actions))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in range(n_episodes):
        state, done = env.reset(), False

        step = 0
        while not done and step < max_steps:
            action = epsilon_greedy(state, (Q1 + Q2) / 2, epsilons[e])
            result = env.step(action)

            # Pick the greedy next action with one table, evaluate it with the other
            if np.random.randint(2):
                argmax_Q1 = np.argmax(Q1[result.next_state])
                q_learning_target = result.reward + gamma * Q2[result.next_state, argmax_Q1] * (not result.done)
                q_learning_error = q_learning_target - Q1[state, action]
                Q1[state, action] += alphas[e] * q_learning_error
            else:
                argmax_Q2 = np.argmax(Q2[result.next_state])
                q_learning_target = result.reward + gamma * Q1[result.next_state, argmax_Q2] * (not result.done)
                q_learning_error = q_learning_target - Q2[state, action]
                Q2[state, action] += alphas[e] * q_learning_error

            state, done = result.next_state, result.done
            step += 1

        Q1_track[e] = Q1
        Q2_track[e] = Q2
        pi_track.append(np.argmax((Q1 + Q2) / 2, axis=1))

    Q_track = (Q1_track + Q2_track) / 2
    V_track = np.max(Q_track, axis=2)  # (n_episodes, n_states): V(s) = max_a Q(s, a)
    Q = (Q1 + Q2) / 2
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V_track, pi, Q_track, pi_track
