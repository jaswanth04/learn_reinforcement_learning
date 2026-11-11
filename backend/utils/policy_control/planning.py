"""Model-based (planning) control: Dyna-Q and trajectory sampling.

Both learn a tabular model of the environment from real experience
(transition counts and mean rewards) and use it for extra simulated
Q-learning updates between real steps.

The per-step building blocks (update_model, q_learning_update, dyna_planning,
trajectory_planning) are shared with the step-at-a-time agents in
online_planning.py, which the API server drives one transition per request.
"""

import numpy as np

from core.models import Environment, QTable
from utils.policy_estimation.schedules import decay_schedule

from .exploration import epsilon_greedy, epsilon_greedy_over


def _sample_next_state(counts: np.ndarray, rng=None) -> int:
    """Sample an index with probability proportional to counts.

    Draws the same single uniform as np.random.choice(len(counts), p=counts / counts.sum()),
    but skips choice()'s per-call validation, which dominates the planning loops.
    """
    rng = np.random if rng is None else rng
    cdf = counts.cumsum()
    return int(cdf.searchsorted(rng.random_sample() * cdf[-1], side='right'))


# ---------------------------------------------------------------- per-step building blocks
def update_model(T_count: np.ndarray, R_model: np.ndarray,
                 state: int, action: int, reward: float, next_state: int) -> None:
    """Count the transition and update the running mean reward of (state, action, next_state)."""
    T_count[state, action, next_state] += 1
    r_diff = reward - R_model[state, action, next_state]
    R_model[state, action, next_state] += r_diff / T_count[state, action, next_state]


def q_learning_update(Q: QTable, state: int, action: int, reward: float, next_state: int,
                      done: bool, gamma: float, alpha: float) -> None:
    """One Q-learning update from a real transition.

    done must mean "terminated" (hole or goal). An episode cut off by max_steps
    is not terminal, so it passes done=False and still bootstraps from next_state.
    """
    td_target = reward + gamma * Q[next_state].max() * (not done)
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error


def dyna_planning(Q: QTable, T_count: np.ndarray, R_model: np.ndarray,
                  visited_states: np.ndarray, n_planning: int,
                  gamma: float, alpha: float, rng=None) -> None:
    """n_planning simulated updates from uniformly sampled, previously visited (state, action) pairs."""
    rng = np.random if rng is None else rng
    for _ in range(n_planning):
        if not Q.any():
            break

        p_state = rng.choice(visited_states)
        actions_taken = np.where(np.sum(T_count[p_state], axis=1) > 0)[0]
        p_action = rng.choice(actions_taken)

        p_next_state = _sample_next_state(T_count[p_state, p_action], rng)
        p_reward = R_model[p_state, p_action, p_next_state]

        # Terminal states are never acted from, so their Q stays 0 and needs no done mask
        td_target = p_reward + gamma * Q[p_next_state].max()
        td_error = td_target - Q[p_state, p_action]
        Q[p_state, p_action] += alpha * td_error


def trajectory_planning(Q: QTable, T_count: np.ndarray, R_model: np.ndarray,
                        start_state: int, max_trajectory_depth: int,
                        gamma: float, alpha: float, epsilon: float, rng=None) -> None:
    """Simulated updates along an epsilon-greedy trajectory through the model,
    starting at start_state (Sutton & Barto §8.4, Fig. 8.4).

    Two details carry the algorithm, both measured on FrozenLake 8x8 (2000
    episodes, depth 100, mean of 4 seeds — learned-policy success / goal
    reaches):

    * the rollout starts where the agent actually ended up (pass the real
      next_state, not the state it acted from; Fig. 8.4 writes `S <- S'`) and
      only picks among the actions the model has seen in a state, otherwise it
      dies at the first unmodelled pair and wastes the trajectory budget;
    * actions come from the behaviour policy, not the greedy one - a greedy
      rollout updates only one action per state along a single model path.

        epsilon-greedy over all actions, old state        0.32 / 171
        epsilon-greedy over modelled, next state (this)   0.44 / 220
        greedy over all actions, old state                0.15 /  48   (4x4, gamma=1)

    Terminal states need no special case: they are never acted from, so a
    rollout reaching one stops immediately.
    """
    rng = np.random if rng is None else rng
    p_state = start_state
    for _ in range(max_trajectory_depth):
        if not Q.any():
            break

        modelled = np.flatnonzero(T_count[p_state].sum(axis=1) > 0)
        # Stops at states the model never saw an action from, terminals included
        if modelled.size == 0:
            break

        p_action = epsilon_greedy_over(p_state, Q, epsilon, modelled, rng)
        counts = T_count[p_state, p_action]

        p_next_state = _sample_next_state(counts, rng)
        p_reward = R_model[p_state, p_action, p_next_state]

        td_target = p_reward + gamma * Q[p_next_state].max()
        td_error = td_target - Q[p_state, p_action]
        Q[p_state, p_action] += alpha * td_error

        p_state = p_next_state


# ---------------------------------------------------------------- full training loops
def dyna_q(env: Environment,
           gamma: float = 1.0,
           init_alpha: float = 0.5,
           min_alpha: float = 0.01,
           alpha_decay_ratio: float = 0.5,
           init_epsilon: float = 1.0,
           min_epsilon: float = 0.1,
           epsilon_decay_ratio: float = 0.9,
           n_episodes: int = 1000,
           n_planning: int = 3,
           max_steps: int = 100):
    """Q-learning plus n_planning simulated updates per real step.

    Planning updates use uniformly sampled previously visited (state, action)
    pairs. Returns (Q, V_track, pi, Q_track, pi_track).
    """
    n_states = env.n_states
    n_actions = len(env.actions)

    pi_track = []

    Q = np.zeros((n_states, n_actions))
    Q_track = np.zeros((n_episodes, n_states, n_actions))

    # Learned model of the environment
    T_count = np.zeros((n_states, n_actions, n_states))  # Transition counts
    R_model = np.zeros((n_states, n_actions, n_states))  # Mean reward per transition
    acted_from = np.zeros(n_states, dtype=bool)          # States with at least one modelled action

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in range(n_episodes):
        state, done = env.reset(), False

        step = 0
        while not done and step < max_steps:
            action = epsilon_greedy(state, Q, epsilons[e])
            result = env.step(action)
            next_state, done = result.next_state, result.done

            update_model(T_count, R_model, state, action, result.reward, next_state)
            acted_from[state] = True

            # Q-learning update from the real transition
            q_learning_update(Q, state, action, result.reward, next_state, done, gamma, alphas[e])

            # Planning: replay simulated transitions sampled from the model
            dyna_planning(Q, T_count, R_model, np.flatnonzero(acted_from), n_planning,
                          gamma, alphas[e])

            state = next_state
            step += 1

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V_track = np.max(Q_track, axis=2)  # (n_episodes, n_states): V(s) = max_a Q(s, a)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V_track, pi, Q_track, pi_track


def trajectory_sampling(env: Environment,
                        gamma: float = 1.0,
                        init_alpha: float = 0.5,
                        min_alpha: float = 0.01,
                        alpha_decay_ratio: float = 0.5,
                        init_epsilon: float = 1.0,
                        min_epsilon: float = 0.1,
                        epsilon_decay_ratio: float = 0.9,
                        n_episodes: int = 1000,
                        max_trajectory_depth: int = 100,
                        max_steps: int = 100):
    """Q-learning plus a simulated epsilon-greedy trajectory after every real step.

    Unlike Dyna-Q's uniform sampling, planning walks a trajectory through the
    learned model, starting from the state just acted from, for up to
    max_trajectory_depth steps; actions along the way come from the same
    epsilon-greedy behaviour policy as acting. Returns (Q, V_track, pi, Q_track, pi_track).
    """
    n_states = env.n_states
    n_actions = len(env.actions)

    pi_track = []

    Q = np.zeros((n_states, n_actions))
    Q_track = np.zeros((n_episodes, n_states, n_actions))

    # Learned model of the environment
    T_count = np.zeros((n_states, n_actions, n_states))  # Transition counts
    R_model = np.zeros((n_states, n_actions, n_states))  # Mean reward per transition

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in range(n_episodes):
        state, done = env.reset(), False

        step = 0
        while not done and step < max_steps:
            action = epsilon_greedy(state, Q, epsilons[e])
            result = env.step(action)
            next_state, done = result.next_state, result.done

            update_model(T_count, R_model, state, action, result.reward, next_state)

            # Q-learning update from the real transition
            q_learning_update(Q, state, action, result.reward, next_state, done, gamma, alphas[e])

            # Planning: an epsilon-greedy rollout through the model, started from
            # the state the agent actually landed in (Fig. 8.4's `S <- S'`)
            trajectory_planning(Q, T_count, R_model, next_state, max_trajectory_depth,
                                gamma, alphas[e], epsilons[e])

            state = next_state
            step += 1

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V_track = np.max(Q_track, axis=2)  # (n_episodes, n_states): V(s) = max_a Q(s, a)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V_track, pi, Q_track, pi_track
