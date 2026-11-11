"""Action-selection strategies shared by the control algorithms."""

from typing import Any

import numpy as np

from core.models import QTable


def greedy_action(state: int, Q: QTable, rng: Any = None) -> int:
    """Action with the highest Q value; ties are broken uniformly at random.

    rng is anything with numpy's legacy sampling methods (np.random.RandomState
    or the np.random module). None uses the global np.random state.
    """
    if rng is None:
        rng = np.random
    q = Q[state]
    best = np.flatnonzero(q == q.max())
    if len(best) == 1:
        return int(best[0])
    return int(best[rng.randint(len(best))])


def epsilon_greedy(state: int, Q: QTable, epsilon: float, rng: Any = None) -> int:
    """Greedy action w.p. 1 - epsilon, otherwise a uniformly random action.

    Ties between equally valued greedy actions are broken uniformly at random.
    Always taking the first maximum (plain np.argmax) means that every state whose
    Q values are still all zero gets the same action, which can stop the agent
    from ever reaching a distant, sparse reward (e.g. FrozenLake 8x8).

    rng: see greedy_action. None uses the global np.random state.
    """
    if rng is None:
        rng = np.random
    if rng.rand() > epsilon:
        return greedy_action(state, Q, rng)
    return int(rng.randint(Q.shape[1]))


def epsilon_greedy_over(state: int, Q: QTable, epsilon: float, actions: Any, rng: Any = None) -> int:
    """epsilon-greedy restricted to a subset of actions.

    Used by model-based planning, where only state-action pairs the model has
    actually seen can be simulated: picking from all actions would stop the
    rollout at the first unmodelled pair and waste the whole trajectory budget.
    `actions` must be non-empty (the caller decides what to do with dead ends).
    Ties are broken uniformly at random, as in greedy_action.

    rng: see greedy_action. None uses the global np.random state.
    """
    if rng is None:
        rng = np.random
    actions = np.asarray(actions)
    if rng.rand() > epsilon:
        q = Q[state, actions]
        best = np.flatnonzero(q == q.max())
        return int(actions[best[rng.randint(len(best))]])
    return int(actions[rng.randint(len(actions))])
