"""Step-at-a-time Dyna-Q and trajectory sampling.

planning.py runs the whole training loop in one call, with the environment in
the same process. These agents run the same algorithm when the environment lives
somewhere else (the browser game): the caller passes in one real transition at
a time and asks for the next action.

    agent = OnlineDynaQ(n_states=16, n_actions=4, n_episodes=30_000, gamma=0.99)
    action = agent.select_action(state, episode)                          # step 0
    agent.learn(state, action, reward, next_state, terminated, episode)   # each step
    action = agent.select_action(next_state, episode)

Calling learn() then select_action() for every step reproduces planning.py's
update order: model update, Q-learning update, planning, then the next action.
"""

from abc import ABC, abstractmethod

import numpy as np

from core.models import QTable
from utils.policy_estimation.schedules import decay_schedule

from .exploration import epsilon_greedy, greedy_action
from .planning import dyna_planning, q_learning_update, trajectory_planning, update_model


class OnlinePlanningAgent(ABC):
    """Q table, learned model (T_count, R_model) and per-episode alpha/epsilon schedules."""

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 n_episodes: int,
                 gamma: float = 1.0,
                 init_alpha: float = 0.5,
                 min_alpha: float = 0.01,
                 alpha_decay_ratio: float = 0.5,
                 init_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 epsilon_decay_ratio: float = 0.9,
                 seed: int | None = None):
        if n_states < 1 or n_actions < 1 or n_episodes < 1:
            raise ValueError("n_states, n_actions and n_episodes must be positive")

        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.n_episodes = int(n_episodes)
        self.gamma = float(gamma)

        self.Q: QTable = np.zeros((self.n_states, self.n_actions))
        self.T_count = np.zeros((self.n_states, self.n_actions, self.n_states))  # Transition counts
        self.R_model = np.zeros((self.n_states, self.n_actions, self.n_states))  # Mean reward per transition

        self.alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, self.n_episodes)
        self.epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, self.n_episodes)

        # Own random state so concurrent sessions never share one random stream
        self.rng = np.random.RandomState(seed)

    def alpha(self, episode: int) -> float:
        return float(self.alphas[self._episode_index(episode)])

    def epsilon(self, episode: int) -> float:
        return float(self.epsilons[self._episode_index(episode)])

    def select_action(self, state: int, episode: int) -> int:
        """Training action: epsilon-greedy with the episode's epsilon."""
        return epsilon_greedy(state, self.Q, self.epsilon(episode), self.rng)

    def greedy_action(self, state: int) -> int:
        """Prediction action: greedy, ties broken at random."""
        return greedy_action(state, self.Q, self.rng)

    def learn(self, state: int, action: int, reward: float, next_state: int,
              terminated: bool, episode: int) -> None:
        """Learn from one real transition: model update, Q-learning update, then planning.

        terminated is True only for a hole or the goal. A transition that hits
        max_steps passes False so the update still bootstraps from next_state.
        """
        alpha = self.alpha(episode)
        update_model(self.T_count, self.R_model, state, action, reward, next_state)
        self._on_model_updated(state)
        q_learning_update(self.Q, state, action, reward, next_state, terminated, self.gamma, alpha)
        self._plan(state, next_state, alpha, self.epsilon(episode))

    def _on_model_updated(self, state: int) -> None:
        """Hook for agents that track extra model statistics."""

    @abstractmethod
    def _plan(self, state: int, next_state: int, alpha: float, epsilon: float) -> None:
        """Simulated updates after the real one: `state` is the state just acted
        from, `next_state` where the agent actually ended up."""

    def _episode_index(self, episode: int) -> int:
        if not 0 <= episode < self.n_episodes:
            raise ValueError(f"episode must be in 0..{self.n_episodes - 1}, got {episode}")
        return episode


class OnlineDynaQ(OnlinePlanningAgent):
    """Dyna-Q: n_planning updates from uniformly sampled visited (state, action) pairs."""

    def __init__(self, *args, n_planning: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_planning = int(n_planning)
        self.acted_from = np.zeros(self.n_states, dtype=bool)  # States with at least one modelled action

    def _on_model_updated(self, state: int) -> None:
        self.acted_from[state] = True

    def _plan(self, state: int, next_state: int, alpha: float, epsilon: float = 0.0) -> None:
        dyna_planning(self.Q, self.T_count, self.R_model, np.flatnonzero(self.acted_from),
                      self.n_planning, self.gamma, alpha, self.rng)


class OnlineTrajectorySampling(OnlinePlanningAgent):
    """Trajectory sampling: an epsilon-greedy rollout through the model, starting from
    the state the agent actually landed in. Both follow Sutton & Barto Fig. 8.4 — a
    greedy rollout (or one that dies at the first unmodelled action) learns almost
    nothing on a big sparse board."""

    def __init__(self, *args, max_trajectory_depth: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_trajectory_depth = int(max_trajectory_depth)

    def _plan(self, state: int, next_state: int, alpha: float, epsilon: float) -> None:
        trajectory_planning(self.Q, self.T_count, self.R_model, next_state,
                            self.max_trajectory_depth, self.gamma, alpha, epsilon, self.rng)
