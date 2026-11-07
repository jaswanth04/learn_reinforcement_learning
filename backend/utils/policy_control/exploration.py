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
