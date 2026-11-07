"""Step-size (alpha) schedules shared by the prediction algorithms."""

from collections.abc import Callable

import numpy as np

# alpha argument of the prediction algorithms:
#   'decay' -> decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
#   float   -> constant step size
#   callable(episode, n_episodes) -> float, a custom per-episode step size
#   None    -> algorithm default (sample mean for MC, 0.1 for TD)
type AlphaSpec = str | float | Callable[[int, int], float] | None


def decay_schedule(init_value, min_value, decay_ratio, max_steps=5000, log_start=-2, log_base=10):
    decay_steps = max(min(int(max_steps * decay_ratio), max_steps), 2)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    return np.pad(values, (0, rem_steps), 'edge')


def _make_alphas(alpha: AlphaSpec, init_alpha, min_alpha, alpha_decay_ratio, n_episodes):
    """Shared step-size schedule: 'decay' | callable | float (default 0.1)."""
    if isinstance(alpha, str) and alpha == 'decay':
        return decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    if callable(alpha):
        return np.array([float(alpha(e, n_episodes)) for e in range(n_episodes)])
    if alpha is None:
        alpha = 0.1
    return np.full(n_episodes, float(alpha))
